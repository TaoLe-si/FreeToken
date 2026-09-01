# E1 - llama.cpp MTP 算法 Port 到 iGPU D3D12 HLSL 的设计文档

**状态**: 调研文档（不实现）
**目标**: 把 MTP head 整层（FC + attention + MoE）从「dGPU torch attn/MoE + iGPU D3D12 FC」融合到「iGPU D3D12 HLSL 一站式 MTP head」，参考 llama.cpp PR #22673 / PR #22400 的算法思路。
**作者**: 调研 subagent
**对应代码**:
- E:/FreeToken/python/freetoken/models/qwen3_5_moe/mtp.py（当前 MTP head，FC 已 iGPU）
- E:/FreeToken/benchmarks/cpu_moe_microbench/t_mxfp4_gemv_v3_server.cpp（现有 iGPU D3D12 FC server）
- E:/FreeToken/benchmarks/cpu_moe_microbench/A3B_MTP_ANALYSIS.md（已有高层场景估算，本文档补算法 + port 细节）

---

## 0. 关键发现 TL;DR

1. **llama.cpp PR #22673 本身没有引入任何 MTP-specific 的新 fused kernel。** MTP head 在 GGML 里的 graph_mtp 直接复用了 GGML_OP_MUL_MAT + GGML_OP_FLASH_ATTN_EXT + GGML_OP_RMS_NORM + GGML_OP_ROPE + GGML_OP_SOFT_MAX + GGML_OP_MUL + GGML_OP_ADD 这些已有的 op。所谓的「MTP 优化」=「把 MTP head 走 GGML graph + 利用现有 flash-attn/mul-mat kernels」，不是「发明新 fused MTP kernel」。
2. PR #22400 的 n_rs_seq（recurrent state rollback）**只针对 GatedDeltaNet 主干层**，MTP head 是 dense attention + dense FFN（**不是 MoE！见 llama.cpp qwen35.cpp graph_mtp**），跟 GDN rollback 没关系。但仍然受益于「checkpoint on device」避免 D2H copy（这就是 PR #22673 后续 #22679 commit 做的事）。
3. FreeToken MTP head 当前架构 = **Qwen3-Next MTP head**，**和 llama.cpp 的 Qwen3.5/3.6 MTP head 几乎一致**（都是 concat(rmsnorm(embed), rmsnorm(prev_hidden)) @ eh_proj -> attn_norm -> QKV proj + qk_norm + RoPE + GQA attn + sigmoid(gate) + o_proj -> post_norm -> SwiGLU FFN -> lm_head_norm -> LM head）。差异只在 MoE 上：FreeToken 用 256/8 routed + 1 shared 的 MoE，llama.cpp Qwen3.5 主干层用 **dense SwiGLU FFN**（不是 MoE，见 qwen35.cpp:415 GGML_ASSERT(model.layers[il].ffn_gate_inp == nullptr)）。这个差异决定了 **必须 port MoE kernel**。
4. **推荐 port 顺序: FC+MoE -> FC+MoE+Attn -> 完整 head**。理由见 §5。**不建议** 反过来先做 attn，因为当前 MTP head 在 CPU 上跑 MoE 是绝对瓶颈（A3B_MTP_ANALYSIS.md 已测 70ms/draft vs 7ms 主模型）。
5. **HLSL DXIL 关键约束**: RDNA3+ iGPU wavefront=32（不是 64），LDS（group shared memory）默认 64KB/CU，bf16 (uint16_t) 在 SM 6.2+ DXIL 原生支持（Strix Halo / Phoenix 是 SM 6.6）。这些都跟 NVIDIA 不同，port 时要 explicit。

---

## 1. llama.cpp MTP 算法概要

### 1.1 PR 关系图

PR #22400 (am17an, merged May 16 2026) -> PR #22673 (am17an, merged May 16 2026)
  "llama: allow partial seq_rm for         "llama + spec: MTP Support"
   GDN models for speculative decoding"      (本任务的主要参考)

PR #22673 的 commit list（按时间顺序）揭示了 port 路径:

1. spec: support MTP (9b996f0) -- 主 commit，引入 MTP class、checkpoint、ubatch hook
2. vulkan: add GDN partial rollback (2ef737a) -- Vulkan 后端适配 GDN
3. metal: add GDN partial rollback (19be81c) -- Metal 后端适配
4. llama-memory: enable checkpointing with partial rollback (cddbb7f) -- 主干修改
5. server: avoid checkpoint data host copies (22558，PR 22673 之前已 merge) -- 这是真正让 MTP 性能起飞的关键 commit: **checkpoint 数据全程留在 device 上不 D2H copy**

**关键点**: PR #22673 在 main-model 侧的修改 = **GDN 部分 seq_rm rollback + device-side checkpoint**。**MTP head 自身完全复用已有 GGML ops**。

### 1.2 graph_mtp 的 GGML 操作流（llama.cpp src/models/qwen35.cpp:graph_mtp）

```cpp
// 1. Embed + prev_hidden 双 RMSNorm
ggml_tensor * h_norm = build_norm(h_embd, layer.nextn.hnorm, nullptr, LLM_NORM_RMS, il);
ggml_tensor * e_norm = build_norm(tok_embd, layer.nextn.enorm, nullptr, LLM_NORM_RMS, il);

// 2. Concat 2*n_embd -> eh_proj -> n_embd
ggml_tensor * concat = ggml_concat(ctx0, e_norm, h_norm, /*dim=*/ 0);
ggml_tensor * cur = build_lora_mm(layer.nextn.eh_proj, concat, layer.nextn.eh_proj_s);  // = FC

// 3. attn_norm -> QKV proj + q/k norm + RoPE + flash-attn + sigmoid(gate)*attn + o_proj
cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
Qcur_full = build_lora_mm(layer.wq, cur);  // 输出 (n_embd_head * 2) * n_head 列
Qcur = ggml_view_3d(... n_embd_head, n_head ...);  // 前半 = Q
Qcur = build_norm(Qcur, layer.attn_q_norm, ..., RMS_NORM);  // Q-norm
gate = ggml_view_3d(... n_embd_head, n_head ... offset n_embd_head);  // 后半 = gate
Kcur = build_lora_mm(layer.wk, cur); Kcur = build_norm(Kcur, layer.attn_k_norm);  // K+K-norm
Vcur = build_lora_mm(layer.wv, cur);  // V 不做 norm
Qcur = ggml_rope_multi(...);
Kcur = ggml_rope_multi(...);
cur = build_attn(... Qcur, Kcur, Vcur, kq_scale);  // flash-attn
cur = ggml_mul(cur, ggml_sigmoid(gate));  // gated MHA
cur = build_lora_mm(layer.wo, cur);  // o_proj
cur = ggml_add(cur, inpSA);  // + residual

// 4. post_norm -> SwiGLU FFN + + residual (Qwen3.5 MTP 是 DENSE FFN, 不是 MoE!)
cur = build_norm(cur, layer.attn_post_norm, ..., RMS_NORM);
cur = build_ffn(cur, ffn_up, ffn_gate, ffn_down, ..., LLM_FFN_SILU, LLM_FFN_PAR);
cur = ggml_add(cur, ffn_residual);

// 5. shared head norm + LM head
cur = build_norm(cur, head_norm_w, ..., RMS_NORM);
cur = build_lora_mm(head_w, cur);  // LM head (可能 tied with model.output)
```

**每一步都是 GGML op fuse 过的标准 kernel**:
- RMSNorm -> GGML_OP_RMS_NORM（CUDA f16/f32 fused，Vulkan 同理）
- QKV proj -> GGML_OP_MUL_MAT（= bf16/fp16 GEMM，行优先 row-major）
- Q/K norm -> GGML_OP_RMS_NORM
- RoPE -> GGML_OP_ROPE（partial rotary 25% 也支持）
- Flash attn -> GGML_OP_FLASH_ATTN_EXT（GQA、因果、causal mask 都内置）
- Gated MHA -> GGML_OP_MUL（element-wise）
- O proj -> GGML_OP_MUL_MAT
- SwiGLU -> GGML_OP_SILU + GGML_OP_MUL（或 GGML_OP_GELU/FFN_*）

### 1.3 FreeToken MTP head 对照（mtp.py）

| FreeToken mtp.py | llama.cpp qwen35 graph_mtp | 匹配度 |
|------------------|---------------------------|--------|
| MtpHeadAttention._project QKV proj + q/k norm + RoPE | build_lora_mm(wq/wk/wv) + build_norm(q/k_norm) + rope_multi | 完全一致 |
| MtpHeadAttention.forward GQA 16/2 repeat + softmax + V + sigmoid(gate) + o_proj | build_attn (flash) + sigmoid(gate) * attn + lora_mm(wo) | 完全一致 |
| MtpHeadMoe 256 routed top-8 + 1 shared + SwiGLU per expert + per-token sigmoid gate | llama.cpp Qwen3.5 MTP 用 **dense SwiGLU FFN**，不是 MoE | **差异!** |
| Qwen3_5MtpHead.forward_with_state pre_fc_norm + concat + eh_proj + +residual + input_layernorm + attn + +residual + post_attention_layernorm + MoE + +residual + mtp.norm + LM head | graph_mtp 同序 | 完全一致 |
| 数值精度 | llama.cpp 默认 fp16/bf16 weights, fp32 compute on attention logits | 一致 |

**结论**: FreeToken MTP head = llama.cpp Qwen3.5 MTP head + 一个额外的 MoE 块。**Port 工作的 95% = port llama.cpp 已有 GGML op 行为到 D3D12 HLSL compute shader。** 只有 MoE 是新东西（llama.cpp 当前没有 MoE-in-MTP，因为 Qwen3.5 主干不是 MoE）。

### 1.4 关键设计: device-side checkpoint（PR #22558，#22679）

llama.cpp 让 MTP 性能起飞的核心不是新 kernel，而是 **避免 host-device copy**:

```
checkpoint 数据（每 step 的 logits / top-k probs）保持在 device 上，
verify 后 rollback 直接 device-side 操作，不回 host。
```

这与 FreeToken 现状: MTP head 的 input 是 dGPU torch tensor（prev_hidden, prev_token_id），FC 走 iGPU 意味着 **每一 draft step 都要 H2D 一次 prev_hidden 4KB** + **D2H 一次 fc output 8KB**。port 到完整 iGPU head 后这个 12KB/step × 3 draft/step = 36KB/step 的开销消失。

---

## 2. 算法要点: 哪些 kernel 怎么 fused

### 2.1 Fused Attention Kernel（QKV proj + q/k norm + RoPE + GQA + flash + gate + o_proj）

**llama.cpp 的 fusion 粒度**: 每个子步骤是独立 GGML op（mul_mat -> norm -> rope -> flash_attn -> mul -> mul_mat），**不跨子步骤 fuse**。优点是灵活、可后端复用; 缺点是中间 tensor 需要落 memory（即使是 on-device，对小 head 也是开销）。

**FreeToken iGPU 上 fused 的可行性**:

| 子步骤 | 是否可 fuse 进同一个 CS | 备注 |
|-------|------------------------|------|
| QKV proj (3 × bf16 matmul) | 可 | M=2048 (cat), N=(q+2*kv)=5120 (16+2+2)*256, K=2048。一次 dispatch 三个 wq/wk/wv 各 GEMM。Fusion 在 matmul 内部（shared mem tiling）。 |
| Q reshape + Q-norm | 可 fuse 进 QKV-proj 之后同一 CS | 2 个独立 RMSNorm，都是 [N=1, H=2048] -> [1, 16, 256]，norm 完直接吐到 group shared，**不落 memory** |
| K reshape + K-norm | 同上，但 [1, 2, 256] | 同上 |
| RoPE | **必须单独一个 CS**（因为需要 inp_pos 即 rope 位置，每个 token 不同） | RoPE 频率向量 = constant buffer，不需要重计算; 只需 ID + base（=10000.0）。256 dim only 25% rotary（=64 dim），所以只对前 64 维算 cos/sin。 |
| Flash attn (GQA 16/2, causal) | 单独 CS | N=1（单 token drafting）, M=cache_len。**关键路径**: cache 长 1024~8192，单 token query 不需要 tiling，分母 exp 和归一化直接 one-pass。 |
| sigmoid(gate) * attn_out | 跟 flash attn fuse 或紧随其后 | element-wise，无内存压力 |
| O proj | 单独 GEMM | M=1, N=2048, K=4096 (16 heads × 256 dim)，bf16 |

**推荐 fusion 边界**:
- **CS#1 = qkv_proj_qknorm_rope**: 吃 x[1, 2048] -> 写 Q[16, 256] roped, K[2, 256] roped, V[2, 256], gate[16, 256]。**一次 dispatch 完成 QKV proj + Q/K norm + RoPE**。这消除了 QKV 输出 intermediate buffer 落 memory 的开销（5120 × 2 = 10KB）。
- **CS#2 = flash_attn_gqa_o_proj**: 吃 Q/K/V、KV cache、inp_pos -> 写 attn_out[1, 2048]。Flash attention 内部用 LDS tiles for K/V，Q=1 不需要。

中间 gate sigmoid 可在 CS#2 头部 inline（一次 sigmoid per element），**不单独 dispatch**。

### 2.2 Fused MoE Kernel（routing + 8 expert GEMV + combine）

**当前 FreeToken MoE = mtp.py:255-287**:
```
gate_logits = self.gate(x)        # [N, 256] bf16 GEMV (1, 2048) @ (2048, 256)
router_probs = softmax(gate_logits, dim=-1)   # [N, 256] fp32
top_w, top_idx = topk(router_probs, 8)       # [N, 8] fp32, int
top_w = top_w / sum(top_w, keepdim=True)     # norm_topk_prob
# 8 GEMV loop:
for k in 0..7:
    eidx = top_idx[:, k]
    w = top_w[:, k]
    sg = switch_gate[eidx] @ x                # [N, 512] GEMV (1, 2048) @ (2048, 512)
    su = switch_up[eidx] @ x                  # [N, 512] GEMV
    sh = silu(sg) * su                        # SwiGLU (single silu)
    sd = switch_down[eidx] @ sh               # [N, 2048] GEMV (1, 512) @ (512, 2048)
    routed += sd * w                          # weighted add

# shared expert:
ssg = silu(shared_gate(x))   # [N, 512] GEMV
ssu = shared_up(x)            # [N, 512] GEMV
shared = shared_down(ssg*ssu) # [N, 2048] GEMV
shared = shared * sigmoid(shared_expert_gate(x))   # scalar per token

return routed + shared
```

**Fusion 策略**:

```
CS#moe_gate_topk
  吃: x[1, 2048]
  算: gate_logits = x @ gate_weight    (bf16 GEMV, M=1, N=256, K=2048)
      router_probs = softmax(gate_logits)
      top_w, top_idx = topk(router_probs, 8)
      top_w = top_w / sum
  写: top_w[1, 8] (fp32), top_idx[1, 8] (uint32)
  写: shared_expert_gate_scalar = sigmoid(x @ shared_expert_gate)  (1 number)
```

```
CS#moe_routed_loop (or BATCH_ALL):
  吃: x[1, 2048], top_idx[1, 8], top_w[1, 8]
  对 8 个 expert e:
    sg = x @ switch_gate[e]         (1, 2048) @ (2048, 512) -> [1, 512]
    su = x @ switch_up[e]           (1, 2048) @ (2048, 512) -> [1, 512]
    sh = silu(sg) * su              (1, 512) elemwise
    sd = sh @ switch_down[e]        (1, 512) @ (512, 2048) -> [1, 2048]
    acc += sd * top_w[k]
  写: routed[1, 2048]
```

```
CS#moe_shared:
  ssg = silu(x @ shared_gate)        (1, 2048) @ (2048, 512) -> [1, 512]
  ssu = x @ shared_up                (1, 2048) @ (2048, 512) -> [1, 512]
  shared = (ssg * ssu) @ shared_down (1, 512) @ (512, 2048) -> [1, 2048]
  shared = shared * sigmoid_scalar   (1, 2048) elemwise
  写: shared[1, 2048]
```

```
CS#moe_combine (trivial):
  out = routed + shared + prev_hidden_residual
  写: out[1, 2048]
```

**BATCH_ALL 优化**（参照 A3B_MTP_ANALYSIS.md 场景 2）: 把 8 个 routed expert GEMV 打包成一次 Dispatch，dim.x = 8 × M_per_expert。这是现有 v3_server BATCH_ALL 路径的延续（MTP FC 层已经验证过），不需要新 shader，只需要 server 端 recv 8 个 expert ID + 8 个 weight handle。

**是不是要把 routing 跟 expert 算 fuse 进同一个 CS?** **不建议**。routing 完了才知道哪些 expert 要算 top-8，data-dependent control flow 在 CS 里会破坏 SIMD alignment。保持 routing 输出 -> CPU 或 dispatch-side 决定 expert ID -> 触发 expert 计算。

### 2.3 FC (eh_proj) Kernel

mtp.py:7-8: FC = 4096 -> 2048 (MXFP4 uint4-affine packed) **已被 v3_server 覆盖**。本任务的 port 工作 **不再重写 FC kernel**，只复用现有 t_mxfp4_gemv_v3_server.cpp 路径（sticky FC_LOAD + FC_CALL）。

### 2.4 RMSNorm + LM head

- RMSNorm: 在 fused attn CS 内部 inline（每个 layer 一次），不单独 dispatch。
- LM head: E:/FreeToken/python/freetoken/models/qwen3_5_moe/mtp.py:317 lm_head = tied with embed_table。bf16 [1, 2048] @ [2048, 248320] GEMV。**M=1, N=248320, K=2048** = 507M FLOPs / token。**FC 的 1/4 工作量**。值得做 iGPU（用 v3_server path, bf16 weights 转一次）。

---

## 3. 数据 Layout（weight + activation）

### 3.1 FreeToken 现状（来自 mtp.py:64-87 _dequant_mxfp4_affine）

| Tensor | shape | dtype | layout | 用途 |
|--------|-------|-------|--------|------|
| fc.weight (eh_proj) | (2048, 4096) | MXFP4 packed (2048, 512) uint32 (K/8) + (2048, 128) fp32 scales (K/32) + (2048, 128) fp32 biases (K/32) | row-major (each row 2048 weights = 256 uint32 packed) | FC, 走 v3_server |
| attn.qkv_proj.weight | (5120, 2048) bf16 | bf16 | row-major | QKV proj（FC 后 cat 出来） |
| attn.q_norm / k_norm | (256,) fp32 (1+weight from checkpoint) | fp32 | 1D | Q/K norm |
| attn.o_proj.weight | (2048, 4096) bf16 | bf16 | row-major | O proj |
| mlp.switch_gate/up | (256, 512, 2048) bf16 | bf16 | 行=expert, col=intermediate, in=hidden (row-major) | MoE |
| mlp.switch_down | (256, 2048, 512) bf16 | bf16 | 行=expert, out=hidden, in=intermediate (row-major) | MoE |
| mlp.shared_gate/up | (512, 2048) bf16 | bf16 | row-major | Shared expert |
| mlp.shared_down | (2048, 512) bf16 | bf16 | row-major | Shared expert |
| mlp.shared_expert_gate | (1, 2048) bf16 | bf16 | row-major | per-token scalar gate |
| mlp.gate | (256, 2048) bf16 | bf16 | row-major | routing |
| KV cache | (C, 2, 256) bf16 | bf16 | dim 0 = sequence pos, dim 1 = kv heads, dim 2 = head_dim | persistent |
| embed_table | (248320, 2048) bf16 | bf16 | row-major | tied with lm_head |

### 3.2 HLSL D3D12 Buffer layout 映射

D3D12 StructuredBuffer / ByteAddressBuffer 都用 row-major（HLSL 默认）。t_mxfp4_gemv_v3_server.cpp 已经是 row-major 上传（packed[base+b], scales[base+b], act[abase+j] 全是按 row 排列）。**新 attn/MoE kernel 沿用同样的 row-major 约定，不需要转置。**

### 3.3 关键的 weight packing 决策

- **bf16 权重不打包**（不像 MXFP4）。每个 element = 2 bytes，行=512 或 2048 个元素对齐到 4-byte boundary (ByteAddressBuffer 可)。
- **MoE expert 排列**: **保留 contiguous expert layout**（switch_gate[256, 512, 2048]），dispatch 时 [expert_id] 索引出对应的 (512, 2048) slice。这是现有 dequantized buffer 的现成格式。**不要按 expert 重新打包**——会破坏 mtp.py:301 的 eidx 索引逻辑，也会破坏 v3_server BATCH_ALL。
- **KV cache layout**: dim 0 = seq_pos。这样 attention CS 可以一次扫 [seq_pos, head_dim] 完整 KV slice，无需 gather。

---

## 4. port 到 D3D12 HLSL 的挑战

### 4.1 CUDA C vs HLSL DXIL 差异（影响 port 思路）

| 项 | CUDA | HLSL DXIL (SM 6.6) | port 注意 |
|---|---|---|---|
| Kernel model | __global__ void kernel(...) | [numthreads(N,M,K)] void main(...) | 一对一 |
| Thread ID | threadIdx.x, blockIdx.x | SV_DispatchThreadID, SV_GroupID, SV_GroupIndex | 一对一 |
| Shared memory | __shared__ float sh[256] | groupshared float sh[256] | 一对一 |
| Wave size | warp = 32 threads | wavefront = 32 (RDNA) / 64 (Vega) | **RDNA 是 32**。[numthreads(64,1,1)] 在 RDNA 占 2 wavefront |
| Sync | __syncthreads() | GroupMemoryBarrierWithGroupSync() | 注意名字 |
| Barriers | __syncwarp() | DeviceMemoryBarrier(), GroupMemoryBarrier() | 用 GroupMemoryBarrierWithGroupSync 更安全 |
| bf16 | __nv_bfloat16 | uint16_t reinterpret-cast to float16_t via f16tof32 / f32tof16 HLSL intrinsic | **DXIL SM 6.2+ 原生支持 min16float / float16_t**。iGPU Strix Halo / Phoenix 都是 SM 6.6 |
| Half precision matmul | mma.sync (Tensor Core) | wave intrinsics + dot4 (RDNA matrix) | **D3D12 6.4 暴露 wave intrinsics 但很新**，DXC support 在 6.6+ 完整。RDNA3 (Phoenix) 有 WMMA 但 **DXIL 暴露度有限**，port 时用标量 bf16 GEMM 更稳妥 |
| Uniform memory | __constant__ | cbuffer (root CBV) | 一对一 |
| Atomic | atomicAdd, etc. | InterlockedAdd, etc. | 一对一 |
| Texture (SRV) | texture<float4, 2D> | Texture2D<float4> | 一对一 |
| Buffer (SRV) | cudaBuffer<T> | StructuredBuffer<T> / ByteAddressBuffer | 一对一 |

**最关键的 port 风险点**:
1. **没有 Tensor Core**。CUDA 在 SM 75+ 直接调 mma.sync，HLSL DXIL 在 RDNA3 上要用 wave matrix intrinsics（DXIL_SM 6.6），**实现复杂且编译器支持不全**。**Port 初期用标量 bf16 GEMM + register tiling**（每个 thread 处理 4×4 bf16 sub-tile，用 f16tof32 转换后做 fp32 FMA），放弃 WMMA。
2. **RDNA wavefront = 32**（不是 NVIDIA 64）。[numthreads(32,1,1)] 是 RDNA 的基本单位; NVIDIA 的 [numthreads(64,1,1)] 占 2 wavefront 在 RDNA 上变成 2 个 32-thread wavefront，**寄存器分配和 LDS bank conflict 行为都不同**。
3. **LDS（groupshared）默认 64KB/CU**（RDNA3）。RDNA 上 LDS bank = 32 dwords wide，groupshared float[256] 在每个 wavefront 内冲突 1-way（每个 thread 不同 bank）。**128 element 的 LDS 在 RDNA 上 stride 要避开 32-dword 边界**。

### 4.2 iGPU 是 AMD（已确认 t_mxfp4_gemv_v3_server.cpp:88）

- VendorId == 0x1002 选中 AMD adapter
- 大概率是 **RDNA3 (Phoenix/Strix)** 或 **RDNA3.5 (Strix Halo)**，SM 6.6 完整，bf16 native
- **没有 FP4** 在 DXIL 6.6。MXFP4 / NVFP4 用 uint4 模拟，跟现有 FC shader 一样。AMD 的 FP4 support 在 RDNA4 (SM 7.0+)。
- **bf16 GEMV 用 scalar fma 是确定的**，因为 vector ALU 是 fp16 native 但 LDS 读写是 fp32-only 在 RDNA3（不能直接 LDS bf16）。**LDS 里存 fp32，VGPR 算 bf16，每次 load/store 一次 conversion**。

### 4.3 FP4 / bf16 在 AMD iGPU 上支持

- **FP4 (E2M1 / NVFP4 / MXFP4)**: RDNA3 没有 native FP4。已有 FC kernel (t_mxfp4_gemv_sk.dxil) 用 **byte unpacking** 模拟（pk.x & 0xFF 拆 8 个 nibble，lookup table kE2M1x2[16] 转成 int，乘以 act 的 int8，累加到 fp32 acc）。
- **bf16**: RDNA3 有 bf16 native ALU。SM 6.6 DXIL 支持 uint16_t reinterpret + f16tof32/f32tof16 intrinsic，**和 NVIDIA CUDA 的 __nv_bfloat16 等价**。
- **fp32**: RDNA3 全速支持。
- 结论: **FC 沿用现有 MXFP4 unpacking 路径，attn/MoE 全部用 bf16 weights + bf16 LDS + fp32 accumulate**。这是最稳的 port 策略。

---

## 5. 推荐 port 顺序

### 推荐顺序: **FC+MoE -> FC+MoE+Attn -> 完整 head**

理由（数据来自 A3B_MTP_ANALYSIS.md）:

| 场景 | MTP head 时间 | 加速 | 工作量 | 风险 |
|-----|-------------|------|-------|------|
| **基线**: FC iGPU, attn/MoE dGPU | 70 ms/draft | 1.04x | 已完成 | - |
| **推荐 step 1**: FC+MoE iGPU | 12-15 ms/draft | 1.4x | 3-5 天 | 中 |
| **推荐 step 2**: + Attn iGPU | 5-7 ms/draft | 1.6x | +2 天 | 中 |
| **Step 3（可选）**: + LM_head iGPU | 3-5 ms/draft | 1.6x (上限) | +3 天 | 高 |

**为什么先 MoE 不是先 Attn**:
- 当前 MoE 在 CPU torch 是 50-60ms / draft（绝对瓶颈）。把 MoE 干掉直接 5x 改善 head 时间。
- Attn 在 dGPU 上已经 ~0.5ms（自由 token 解码），port 到 iGPU 改善 < 1ms，但 risk 大（flash attn kernel 的 HLSL 实现比 MoE GEMV 复杂一个数量级）。
- MoE 复用现有 v3_server FC 基础设施（MXFP4 packing、weight load、BATCH_ALL dispatch 模式都现成）。Attn 是新 CS、没有现成模板。

### 推荐 step 1 实施细节（FC + MoE on iGPU）

```
1. 服务端协议扩展（v3_server.cpp）
   - 新增 MOE_LOAD: 上传 256 expert weights (switch_gate, switch_up, switch_down)
                    + routing gate weight + shared expert weights
                    + shared_expert_gate weight
   - 新增 MOE_CALL: 吃 1 个 [1, 2048] bf16 activation token
                    输出 1 个 [1, 2048] bf16 output
                    内部: routing CS + 8 routed GEMV (BATCH_ALL) + shared GEMV + combine
   - 沿用现有 MXFP4 packed 格式（FC）for routing gate 的 bf16 (改 1 个 weight 类型)

2. 客户端 (mtp.py)
   - 修改 igpu_fc.expert dispatch 路径: 把 self.mlp.forward(x) 替换成 igpu_moe(...)
   - self.mlp.igpu_moe_call(x) -> 把 x H2D -> 等 D2H output

3. 验证
   - 用 t_mtp_head_driver.py: 输入 [1, 2048] bf16 + 1 个 token_id, 对比 PyTorch ref
   - 容差 < 1e-2 (bf16 accumulate 噪声)
```

### 推荐 step 2 实施细节（+ Attn iGPU）

```
1. 服务端协议扩展
   - 新增 ATTN_LOAD: 上传 QKV proj + o_proj + q_norm + k_norm
   - 新增 ATTN_INIT_KVCACHE: 分配 KV cache buffer on device (max_seq_len × 2 × 256 × 2bytes)
   - 新增 ATTN_APPEND_KV: 吃 [1, 2048] x + position -> 写新 KV rows
   - 新增 ATTN_FORWARD: 吃 [1, 2048] x + position -> 写 [1, 2048] output
                       内部: CS#1 (QKV proj + Q/K norm + RoPE)
                             CS#2 (Flash attn GQA + sigmoid(gate)*attn + O proj)

2. 客户端
   - self.attn.igpu_attn_call(x, position) 替代 _project + forward
   - KV cache 完全 on device, truncate_kv 通过 ResetEvent + buffer resize

3. llama.cpp 的 partial-seq-rm rollback
   - 暂不实现。FreeToken 的 KV cache 是 contiguous, 验证失败直接用 checkpoint 覆盖。
   - 对应 llama.cpp 的 PR #22558 思想: "checkpoint on device 不 D2H copy"
```

### 推荐 step 3（可选，+ LM head iGPU）

- LM_head = embed_table.t() (tied)，V=248320
- 跟现有 FC shader 一样处理，**但 dtype 改成 bf16**，因为 LM_head 权重是 bf16 not MXFP4
- 需要新写一个 bf16 GEMV shader（256 thread per row, group reduction）
- 工作量估 2-3 天

---

## 6. D3D12 HLSL Compute Shader 实现要点

### 6.1 复用模板

直接照搬 t_mxfp4_gemv_v3_server.cpp + d3d12_gemv_sk.hlsl:
- Server 模板: t_mxfp4_gemv_v3_server.cpp:75-560（D3D12 device 创建 + command queue + PSO + dispatch loop）
- Shader 模板: d3d12_gemv_sk.hlsl:1-38（groupshared reduction, numthreads(256,1,1), root CBV + SRV + UAV）
- Root signature: 沿用现有 8 个 root param（6 SRV + 1 UAV + 1 CBV）。**新 MoE shader 需要更多 SRV**（256 expert weights），可以:
  - **方案 A**: 把 expert weights 拼成一个 mega buffer（[256*512*2048] bf16），CS 用 byteAddressBuffer[k*stride + ...] 寻址。**最简单，强烈推荐**。
  - 方案 B: 每 expert 一个 SRV descriptor，root signature 扩展到 256+ SRV。D3D12 root signature 上限 64，D3D12 root param 用 descriptor heap 替代（用 t-register space 绑定），但 **DXIL SM 6.0 不支持 unbounded descriptor tables easily**，需要 SM 6.6 + 大量 descriptor 拷贝。

### 6.2 推荐 MoE Shader（CS#moe_routed）伪代码

```hlsl
// Root CBV: M, K, N, ns (ns=8 experts)
// SRV t0: x[1, 2048] bf16 (activation, broadcast)
// SRV t1: switch_gate [256*512, 2048] bf16 (mega-buffer, [expert_id * 512 * 2048 + r * 2048 + k])
// SRV t2: switch_up [256*512, 2048] bf16
// SRV t3: switch_down [256*2048, 512] bf16
// SRV t4: top_idx[1, 8] uint32, top_w[1, 8] float
// UAV u0: routed_out[1, 2048] bf16
// UAV u1: scratch[1, 512] fp32 (intermediate silu(gate)*up)
// groupshared float sh[256] (reduction)

[numthreads(256, 1, 1)]
void main(uint3 g : SV_DispatchThreadID, uint t : SV_GroupIndex) {
    // Layout: dim.x = 8 (experts), dim.y = 4 (M=2048 / 512 split) -> 8*4 = 32 thread groups
    uint expert_id = g.x;          // [0..7]
    uint tile_row  = g.y;          // [0..3], each tile = 512 rows of the output

    uint row_base = tile_row * 512;

    // Read top_idx and top_w for this expert
    uint eidx = top_idx[expert_id];
    float w_e = top_w[expert_id];

    // Compute (silu(gate) * up) for this tile of 512 rows
    //   sg_tile = x[1, 2048] @ switch_gate[eidx, row_base..row_base+512, :]
    //   su_tile = x[1, 2048] @ switch_up[eidx, row_base..row_base+512, :]
    float acc[16];   // 256 thread * 16 = 4096 / 4 fp32 per thread (1 row per thread)
    // ... vectorized dot product with bf16 weights via f16tof32
    // ... write to scratch[1, 512] as fp32 (so CS#down can read)

    GroupMemoryBarrierWithGroupSync();

    // Compute down: sh[512] @ switch_down[eidx, row_base..row_base+512, :]
    //   output = sum_k sh[k] * switch_down[eidx, m, k]
    // Each thread does 2048/256 = 8 output elements
    float rdn[8] = {0};
    for (uint k = 0; k < 512; ++k) {
        float sh_v = sh[expert_id*512 + k];   // OR scratch[expert_id, k]
        for (uint m = 0; m < 8; ++m) {
            uint weight_idx = (eidx * 2048 + row_base + t*8 + m) * 512 + k;
            rdn[m] += sh_v * f16tof32(switch_down[weight_idx]);
        }
    }

    // Atomic add to routed_out
    for (uint m = 0; m < 8; ++m) {
        uint out_idx = tile_row * 512 + t*8 + m;
        InterlockedAdd(routed_out_u32[out_idx], float_asuint(rdn[m] * w_e));
        // (using fp32 atomics on reinterpreted uint)
    }
}
```

注意: 实际实现需要 **intermediate scratch buffer** 存 silu(gate)*up 的 fp32，因为 silu+mul 不能 in-place over bf16（精度），且要被 down GEMV 读取。**两次 Dispatch** (CS#moe_gateup + CS#moe_down)。

### 6.3 推荐 Attn Shader（CS#qkv_proj_rope）

```hlsl
// Root CBV: H=2048, QH=16, KH=2, HEAD=256, ROTARY_DIM=64
// SRV t0: x[1, 2048] bf16 (input)
// SRV t1: wq[2*16*256, 2048] bf16 (Q+gate projection)
// SRV t2: wk[2*256, 2048] bf16 (K)
// SRV t3: wv[2*256, 2048] bf16 (V)
// SRV t4: q_norm[256] fp32, k_norm[256] fp32 (Gemma form = 1+weight)
// SRV t5: rope_freq[64] fp32 (precomputed inv_freq)
// UAV u0: Q[16, 256] bf16, K[2, 256] bf16, V[2, 256] bf16, gate[16, 256] bf16
// UAV u1: scratch (Q-norm intermediate fp32 [16, 256], K-norm intermediate fp32 [2, 256])
// groupshared float sh_x[2048]   (load x once, reuse 4 times)

[numthreads(256, 1, 1)]
void main(uint3 g : SV_DispatchThreadID, uint t : SV_GroupIndex) {
    // Layout: dim.x = 1 (single dispatch since M=1, N=5120 wide)
    //   - Compute Q/K/V/gate in 4 phases, each tile of 1280 columns per pass (5120/4)
    //   - Each thread processes 4 output elements

    // Phase 1-4: For each output column group, compute:
    //   out = sum_k x[k] * w[col, k]   (bf16 * bf16 -> fp32 acc)
    //   Apply Q-norm if output is Q part (col in [0..16*256))
    //   Apply K-norm if output is K part
    //   Apply RoPE if output is Q or K and col in [0..rotary_dim/2]

    // After compute, write to UAV (Q[16, 256], K[2, 256], V[2, 256], gate[16, 256])
}
```

**注意 RoPE 在 fused proj 内部的处理**:
- Rotary dim = 64 (25% of 256)，只对前 64 dim 算 cos/sin
- Pair (i, i+32) for i < 32: rotate by pos * inv_freq[i]
- pos 通过 CBV 传入（per-call scalar）
- inv_freq[32] 预计算存 CBV
- 在 thread 写 Q/K 前 inline 做 rotate

### 6.4 推荐 Attn Shader（CS#flash_attn_gqa_o_proj）

```hlsl
// Root CBV: cache_len, head_dim=256, n_q=16, n_kv=2
// SRV t0: Q[16, 256] bf16
// SRV t1: K_cache[cache_len, 2, 256] bf16
// SRV t2: V_cache[cache_len, 2, 256] bf16
// SRV t3: gate[16, 256] bf16
// SRV t4: wo[2048, 4096] bf16  (o_proj)
// UAV u0: out[1, 2048] bf16
// groupshared float sh_q[16][256]    // Q across all 16 heads
// groupshared float sh_kv[256]      // one cache row at a time

[numthreads(256, 1, 1)]
void main(uint3 g : SV_DispatchThreadID, uint t : SV_GroupIndex) {
    // Phase 1: Load Q + gate into LDS (all 16 heads, 16 threads per head)
    // Phase 2: For each cache position m = 0..cache_len:
    //          - Load K[m], V[m] into LDS (2 heads, 256 dim each = 512 elements)
    //          - For each query head h in 0..15:
    //              kv_head = h / 8 (GQA repeat 8)
    //              score = sum_d Q[h, d] * K[m, kv_head, d] * scale
    //              online softmax (running max + running sum)
    //          - After loop: out[h, d] = sum_m (exp(s_m - max) * V[m, kv_head, d]) / Z
    //          - out *= sigmoid(gate[h, d])
    //          - wo GEMV: out_proj[h, d] = sum_h sh_attn[h, d] * wo[d, h*256 + d]
    // Phase 3: Atomic write final bf16 to out buffer
}
```

**简化路径**: cache_len 通常 < 1024，单 dispatch (256 thread × 1 group) 足以 hold 整个 cache 在 LDS（每 row 512 elements × 1024 row = 512K elements = 2MB，超出 LDS）。**所以 cache 必须 spill 出 LDS，每次迭代 load 一行**（[2*256] = 512 fp32 = 2KB fits）。

---

## 7. 风险点 + 替代方案

| 风险 | 等级 | 替代方案 |
|------|------|---------|
| **MXFP4 -> bf16 conversion 路径走不通**: 现有 FC shader 用 uint4-affine (_UAFF16 table, not kE2M1) 与 e2m1 FC 不同。**MoE 是 bf16 不需要 MXFP4**，但 FC 还是 MXFP4 affine，shader 必须用 _UAFF16 lookup | 低 | 完全照搬现有 FC shader pattern，仅切换 dtype |
| **RDNA wavefront 32 影响 register 压力**: 标量 bf16 GEMV 256 thread/group × 16 register/thread = 4096 VGPR, 在 RDNA 上每 wavefront 用 32 thread × 128 reg = 4096 / 32 = 128 reg/thread, **超出 RDNA3 上限 256 reg/thread 但仍在范围内**，DSO 不会自动 spill | 低 | 把 thread count 降到 128, 每个 thread 处理 16 outputs, VGPR 占用减半 |
| **MoE BATCH_ALL 8 dispatch + H2D overhead 不划算**: v3_server BATCH_ALL 一次 dispatch 内多个 weight，但每个 weight 仍要 H2D 上传。MTP draft 一次只算 8 expert，H2D 8 个 bf16 (8 * 512 * 2048 * 2 bytes = 16MB) per step 太慢 | **中** | 把 256 个 expert weights **预 stick 在 device 上**（sticky MOE_LOAD 一次，per-step 只 H2D 1 个 2048-element activation）。v3_server 已经有 sticky FC_LOAD 路径，照搬 |
| **bf16 accumulate 精度不够**: mtp.py:97 用 fp32 mean-of-squares + fp32 RMSNorm。HLSL 里 accum 用 fp32 (float), LDS 用 bf16, GEMV 内部 acc += f16tof32(weight) * f16tof32(act) | 低 | 一致: fp32 accumulate 全程, bf16 只在 weight load 和 act 复用 |
| **KV cache 太大超出 VRAM**: 单 sequence cache_len × 2 × 256 × 2bytes = cache_len × 1KB. 8192 context = 8MB per sequence, 16 concurrent sequence = 128MB. iGPU VRAM 通常 4-16GB | 中 | **环形 buffer + truncated attention**。当前 mtp.py:153 truncate_kv 已经支持 truncate 策略，cache 长度 capped 在 recent N tokens |
| **Flash attention GQA repeat 在 fused kernel 里复杂**: 16 query head 共享 2 kv head, 每次 query head h 对应 kv_head = h/8 | 低 | 完全在 LDS 内部 replicate K/V（每个 wavefront 处理 8 query heads, load K/V 一次, 复用 8 次） |
| **D3D12 6.6 wave intrinsics 不成熟**: 如果用 WMMA 路径, DXC 编译器支持度参差 | 高 | **完全不用 WMMA**，用 scalar bf16 FMA（已知稳定，参考 d3d12_gemv_sk.hlsl 模式） |
| **与现有 _p0_backup 老 shader 路径冲突**: history 里有 _p0_backup/ 目录的旧 shader, 代码已迁出但留下 dxil 残骸 | 低 | 完全新建 shader 文件 t_mtp_attn_*.hlsl, t_mtp_moe_*.hlsl, 不动 _p0_backup |
| **校验容差**: mtp.py 已有 t_mtp_head_driver.py 验证，但仅对 FC 精度。Attn/MoE 加 iGPU 后需要新增 attn/MoE 比对测试 | 中 | 复用 tests/kernels/ 下的 avx2 GEMV 比对模式 |

---

## 8. 工作量估算（细化）

### Step 1: FC + MoE iGPU（推荐先做）

| 子任务 | 工作量 | 风险 | 备注 |
|-------|-------|------|------|
| 服务端 MOE_LOAD 协议扩展（256 expert weight stick） | 1 天 | 低 | 复用 v3_server upload heap + resource state machine |
| 新 CS#moe_gate (gate_logits + softmax + topk + norm_topk_prob) | 1 天 | 中 | top-k 用 bitonic sort on LDS, 8-选-1 简单 |
| 新 CS#moe_gateup (8 routed gemv + silu(g)*u per expert, BATCH_ALL) | 2 天 | 中 | 沿用 FC shader pattern |
| 新 CS#moe_down (8 routed down + weighted combine) | 1 天 | 中 | scratch buffer 复用 |
| 新 CS#moe_shared (shared gate/up/down + sigmoid scalar gate) | 0.5 天 | 低 | 单 GEMV, 已知模板 |
| 客户端 (mtp.py) MtpHeadMoe.igpu_call wrapper + bf16 H2D/D2H | 0.5 天 | 低 | 改 < 50 行 |
| 验证测试: MoE iGPU vs PyTorch ref | 1 天 | 中 | 复用 t_mtp_head_driver.py |
| **小计** | **5-7 天** | 中 | 与 A3B_MTP_ANALYSIS.md 估的 3-5 天偏高, 因为要新写 4 个 CS |

### Step 2: + Attn iGPU

| 子任务 | 工作量 | 风险 | 备注 |
|-------|-------|------|------|
| 服务端 ATTN_LOAD 协议（QKV proj + o_proj + q/k norm） | 0.5 天 | 低 | 跟 MOE_LOAD 同模板 |
| KV cache buffer 分配 + ATTN_APPEND_KV | 0.5 天 | 中 | ring buffer on device |
| 新 CS#qkv_proj_rope (4-pass GEMV + q/k norm + RoPE) | 2 天 | 高 | 跨子步骤 fuse, RoPE 在 GEMV 内 inline |
| 新 CS#flash_attn_gqa_o_proj (1-token decode flash + o_proj) | 2 天 | 高 | cache in LDS 行循环, online softmax |
| 客户端 MtpHeadAttention.igpu_call | 0.5 天 | 中 | _project + forward 合并 |
| 验证: attn iGPU vs PyTorch ref | 1 天 | 中 | 沿用 attn 比对测试 |
| **小计** | **5-7 天** | 高 | 比 A3B_MTP_ANALYSIS.md 估的 +2 天多, 因为 RoPE inline 和 flash attn 都复杂 |

### Step 3 (可选): + LM head iGPU

| 子任务 | 工作量 | 风险 |
|-------|-------|------|
| 新 CS#lm_head_bf16_gemv (256 thread × 248320 outputs, 2-pass reduction) | 2 天 | 中 |
| 协议 LMHEAD_LOAD + LMHEAD_CALL | 0.5 天 | 低 |
| 客户端 wrapper | 0.5 天 | 低 |
| 验证 | 0.5 天 | 低 |
| **小计** | **3-4 天** | 中 |

### 总工作量

| 阶段 | 累计 | 加速目标 |
|------|------|---------|
| Step 1: FC + MoE | 5-7 天 | 1.4x |
| Step 2: + Attn | 10-14 天 | 1.6x |
| Step 3 (可选): + LM head | 13-18 天 | 1.6x (上限) |

**注意**: A3B_MTP_ANALYSIS.md 总估「+Attn」只 2 天, 本文细化到 5-7 天。差异在于:
1. Attn 跨子步骤 fuse 比 FC 的单 GEMV 复杂
2. KV cache 管理是新增
3. RoPE 跟 QKV proj 同 CS 的融合需要重新设计 LDS 布局

---

## 9. 关键参考文件 / 链接

### llama.cpp 源码位置
- src/models/qwen35.cpp::graph_mtp -- MTP head GGML graph（**核心参考**）
- src/models/qwen35.cpp::build_layer_attn -- 标准 Qwen3.5 attention（MTP head 直接复用结构）
- src/models/qwen35.cpp::build_layer_ffn -- Qwen3.5 dense SwiGLU（MTP head 用，跟 MoE 不同）
- include/llama.h::llama_context_params::n_rs_seq -- partial seq_rm 配置（暂不实现）
- src/llama-memory-recurrent.h -- GDN rollback 实现（**仅 GDN 用，不影响 MTP head**）
- common/speculative.cpp::speculative_add_inline -- speculative decoding 主流程

### PR 链接
- [PR #22673](https://github.com/ggml-org/llama.cpp/pull/22673) -- MTP support（主参考）
- [PR #22400](https://github.com/ggml-org/llama.cpp/pull/22400) -- GDN partial seq_rm（**已 ship 进 #22673**）
- [PR #22558](https://github.com/ggml-org/llama.cpp/pull/22558) -- server avoid checkpoint host copies（**关键 commit**）
- [PR #22587](https://github.com/ggml-org/llama.cpp/pull/22587) -- CUDA row-per-warp GDN kernel（GDN 优化参考，与 MTP head 无关）

### FreeToken 现有代码
- python/freetoken/models/qwen3_5_moe/mtp.py -- 当前 MTP head (FC iGPU, attn/MoE dGPU)
- benchmarks/cpu_moe_microbench/t_mxfp4_gemv_v3_server.cpp -- FC server 模板
- benchmarks/cpu_moe_microbench/t_mxfp4_gemv_v3_server_full.cpp -- 简化版 server
- benchmarks/cpu_moe_microbench/d3d12_gemv_sk.hlsl -- HLSL compute shader 模板
- benchmarks/cpu_moe_microbench/A3B_MTP_ANALYSIS.md -- 已有的高层场景估算
- benchmarks/cpu_moe_microbench/MtpParallelDriver -- 主模型 + MTP 并行架构（方向 1）

### 第三方
- [AMD RDNA3 ISA reference](https://www.amd.com/system/files/TechDocs/rdna3-shader-instruction-set-architecture.pdf) -- wave size, LDS bank, VGPR limits
- [Microsoft D3D12 SM 6.6 documentation](https://learn.microsoft.com/en-us/windows/win32/direct3d12/) -- DXIL wave intrinsics, root signature 1.2

---

## 10. 待决策 / 留给后续

1. **是否需要 partial-seq-rm rollback?** 当前 FreeToken MTP head 是 dense attention (no GDN), KV cache 直接 contiguous, 验证失败直接 checkpoint 覆盖即可。**结论: 暂不需要**。llama.cpp 的 partial-seq-rm 是 GDN 特有（PR #22400 标题 "for GDN models"）。
2. **MoE weights 留在 GPU 还是 per-call H2D?** v3_server BATCH_ALL 当前是 per-call H2D, 256 expert × 512 × 2048 × 2 bytes = 512MB, 16GB/s H2D = 32ms/step. **绝对不能 per-call H2D**. 必须 sticky MOE_LOAD.
3. **KV cache 长度上限?** 当前 MtpHeadAttention.kv_len() 无 cap, 但 iGPU VRAM 限制需要 cap. 推荐 cap = 4096 (per sequence), ring buffer.
4. **精度容差?** FC MXFP4 已有 t_mxfp4_gemv_v3_server.cpp 验证 < 1e-2 误差. Attn/MoE bf16 比对需要 fp32 reference, 用 PyTorch CPU bf16 ref 作基准.
5. **是否需要适配多 sequence 并发?** 当前 mtp.py 是单 sequence. FreeToken 整体是否多 seq 待查. 如果多 seq, KV cache 需要 per-seq partition + batch attn (bigger fusion opportunity).

---

## 11. 一句话总结

**Port llama.cpp MTP 算法 = 复用 llama.cpp graph_mtp 的 GGML op 序列到 D3D12 HLSL，每个子步骤一个 CS（不强求跨步骤 fuse），分两步走 FC+MoE -> +Attn，估 10-14 天达到 1.6x 加速。** 核心 port 模板是 d3d12_gemv_sk.hlsl，核心风险是 RoPE/flash-attn 跨子步骤 fused CS 的 LDS 布局。
