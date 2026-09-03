# FreeToken 底层原理 + 优化方案 (架构总览)

> **目标读者**：项目维护者，需要对系统每个环节的"做什么、为什么慢、可以怎么优化"形成完整心智模型
> **适用模型**：Qwen3.6-35B-A3B-MXFP4-MTP-NVFP4（混合 GDN 线性层 + 完整自注意力，256 选 8 的 MoE，MTP K=3，MXFP4-Affine 量化）
> **运行环境**：NVIDIA RTX 4070 Laptop（8GB VRAM，CUDA 13），Python 3.12，PyTorch 2.x

---

## 第 0 章：从一句话到一次计算的完整链路

**核心类比**：想象模型是一家**超精密的水管工厂**，水管里流的是"数字"。要让它把"你好"变成"世界"，需要这些工人接力：

1. **装订员**（Embedding）：把"你好"两个字查表 → 变成两个 2048 维的浮点向量（每个数代表这个字的一种"语义浓度"）
2. **40 层精加工车间**（Decoder Layers）：每层把这两个向量过一道，叠加上"上下文含义"
3. **质检员**（LM Head）：最后把 2048 维的向量查一张 152064 行的表，输出每个字可能是下一个字的概率
4. **拣货员**（Sampler）：从概率里挑一个 → "世界"

每生成 1 个新字，**整条流水线就要再跑一次**，但带的东西越来越重（前面所有生成过的字都得带上作参考）。这就是 LLM 推理。

---

## 第 1 章：硬件层 — 你的 RTX 4070 Laptop 是什么样的"工厂"

### 1.1 物理现实

```
RTX 4070 Laptop GPU 规格：
  计算单元 (CUDA cores): 4608 个
  显存 (VRAM):       8 GB GDDR6
  显存带宽:          256 GB/s
  算力 (FP16):       15.5 TFLOPS
  算力 (FP4):        ~120 TFLOPS (理论)
  
CPU 内存 (RAM):     16~32 GB DDR5 (共享给 iGPU)
硬盘:               NVMe SSD (3~5 GB/s 顺序读)
```

### 1.2 内存金字塔

```
        寄存器 (per SM):    ~256 KB   延迟 1 cycle
        ↓
        L1/SRAM (per SM):   128 KB    延迟 ~30 cycles
        ↓
        L2 Cache (全局):    4 MB      延迟 ~150 cycles
        ↓
        VRAM:               8 GB      延迟 ~400 cycles, 带宽 256 GB/s
        ↓
        共享内存 (iGPU):     8 GB      延迟 ~400 cycles, 带宽 50~80 GB/s (PCIe + WDDM 桥)
        ↓
        NVMe SSD:           1 TB      延迟 100 µs, 带宽 3 GB/s
```

**关键推论**：在算力 (15 TFLOPS) 充足时，**瓶颈永远是"数据搬到算力旁边有多快"**。模型权重占 17.5 GB（35B × 4bit），**它不可能全部塞进 8 GB VRAM**。这就决定了 FreeToken 必须做"分层搬运"。

### 1.3 WDDM 共享内存的代价

Windows 用的 **WDDM (Windows Display Driver Model)** 把 GPU 当成"独占显存用完才能用系统内存"的设备。这跟 Linux 的 UVM（统一内存）不一样——WDDM 走的是：

```
VRAM 满了 → 把"溢出页"移到系统内存 → 访问触发 page fault → driver 把页换回 VRAM
```

这个 page fault 一次 ≈ **50~200 µs**。如果模型频繁缺页（thrash），**整个推理就卡在翻页上**。FreeToken 启动日志里能看到这行：

```
KV device=shared: allocating 16384 pages (16384 tokens) via the WDDM shared pool; VRAM overflow is driver-managed
```

意思：**KV cache 用 WDDM 共享池**，访问它走的是 driver-managed 换页。如果 KV 访问模式不连续，速度会被翻页拖慢。

---

## 第 2 章：模型架构 — "35B" 这个数字到底什么意思

### 2.1 三个尺寸数字

| 数字 | 含义 |
|---|---|
| **35B** | 模型**总参数量** 35 billion (175 亿)。每个权重用 4 bit 存 → 17.5 GB |
| **A3B** | 每生成 1 个 token，**实际激活**的参数 ~3 billion。原因是 MoE 只用 8/256 个专家 |
| **2048** | hidden size。每个 token 内部用 2048 个浮点数表示 |

**核心洞察**：35B 模型**大部分参数是"字典"**——256 个专家里每个 token 只查 8 个。所以：
- **存**：要 17.5 GB（要查的字典都在那）
- **算**：只要 3B × 2 bytes = 6 GB 的算力（每个 token 只翻 8 本字典）

### 2.2 模型文件长什么样

```
E:\models\Qwen3.6-35B-A3B-MXFP4-MTP-NVFP4\model.safetensors.index.json
├── layers.0.self_attn.q_proj.weight       [MXFP4, 2048×2048 packed]
├── layers.0.self_attn.k_proj.weight       [MXFP4]
├── layers.0.self_attn.v_proj.weight       [MXFP4]
├── layers.0.self_attn.o_proj.weight       [MXFP4]
├── layers.0.linear_attn.in_proj.weight    [MXFP4, GDN 层混合]
├── layers.0.linear_attn.conv1d.weight    [FP32, depthwise conv]
├── layers.0.mlp.gate.weight              [MXFP4, 256 专家 × 2048×1536]
├── layers.0.mlp.up_proj                  [MXFP4, 256 专家]
├── layers.0.mlp.down_proj                [MXFP4, 256 专家]
├── ... × 40 层 ...
├── mtp.fc.weight                         [MXFP4, MTP head 的 fc 投影]
└── lm_head.weight                         [BF16, 152064 × 2048]
```

### 2.3 MXFP4-Affine 量化详解

**为什么不直接用 FP16**：175 亿 × 2 bytes = 35 GB，放不下。

**为什么不用 INT4**：INT4 是整数 0~15，但权重是连续浮点数，**直接量化精度损失大**。

**MXFP4-Affine 是怎么做的**：

```
把 32 个连续权重分一个 block：
  每个 block 有 1 个 scale (FP32) 和 1 个 bias (FP32)
  32 个权重各自是 4-bit 整数 (0~15)
  
反量化公式：weight[i] = nibble[i] * scale + bias
```

**和 e2m1 的区别**：传统 MXFP4 用 e2m1 编码（0,1,2,3,4,6,8,12 这些"魔法数字"），但 Qwen3.6 导出用的是 **uint4-affine**（直接查表）。这个区别在 `benchmarks/cpu_moe_microbench/t_mxfp4_gemv_sk.hip` 里修了——用 e2m1 表会让 FC 输出 cos=-0.33 反相关，把 MTP 接收率打到 0%。

---

## 第 3 章：每次生成 1 个 token 的完整流程

```
用户输入: "你好" → [token_id=8160]
   ↓
[1] Embedding 查表 (8192 → 2048 维)
   ↓
[2] Prefill: 把 prompt 一次性过 40 层
     每层:
       a. 完整自注意力 (full attn) — 算所有 token 两两关系, O(n²) 内存
       b. GDN 线性层 — 用递推式压缩历史, O(n) 内存
       c. MoE FFN — Router 选 8 专家, 各自跑 SwiGLU
   ↓
[3] LM Head: 最后 1 个 hidden 查 152064 词表 → 概率
   ↓
[4] Sample: 取 argmax → token_id
   ↓
[5] Decode: 拿新 token + 之前的 KV cache, 再过 40 层 → 下一个 token
   ↓
[6] 重复 [5] 直到 EOS 或 max_tokens
```

**Prefill vs Decode 的本质区别**：
- Prefill：批处理 prompt 的所有 token。**算力密集**（一次处理 100 个 token）
- Decode：每次只生成 1 个 token。**内存带宽密集**（每次要从 17.5 GB 权重里读用到的部分）

**Decode 阶段的速度公式**：
```
吞吐量 = 1 / (权重搬运时间 + 计算时间)
       ≈ 1 / (读权重带宽 / 算力能消化的速度)
       ≈ 算力 / 每次访问的权重字节
```

**例子**：decode 1 个 token 需要读 ~6 GB 权重（MoE 8 个专家 + attention），在 256 GB/s 显存带宽下：
- 理论下界 = 6 GB / 256 GB/s = **23 ms/token = 43 t/s**

这接近 50 t/s 的目标。但实际只有 6 t/s —— **差距来自哪里？**

---

## 第 4 章：实际瓶颈在哪 — 6 t/s vs 50 t/s 差了什么

### 4.1 用工具看真实数据

```
启动: python -m freetoken.cli daemon --host 127.0.0.1 --port 1900
启动引擎: POST /engine/start
```

启动日志：

```
Free memory before loading model: 6.89 GiB       # VRAM 总量
Free memory after initialization: 1.21 GiB        # 装完权重还剩 1.2 GB
Free GPU memory before capturing CUDA graphs: 1.20 GiB
Free GPU memory after capturing CUDA graphs: 0.98 GiB
Start capturing CUDA graphs with sizes: [1, 2, 4]
MTP enabled: K=3, igpu_fc=True
[MTP] draft warmup done in 195.7 ms
```

**含义**：35B 模型装载完成后，**VRAM 只剩 0.98 GB**。1.21 GB 减去 0.23 GB 用于 CUDA graph capture。这意味着：

- **KV cache 不能完全放在 VRAM**
- **MoE 256 个专家里只有少数能放 VRAM**
- **每次 decode 要从系统内存搬权重进 VRAM**

### 4.2 当前架构的算账

| 步骤 | 当前路径 | 占用 |
|---|---|---|
| 加载模型 | 17.5 GB 权重 → 6.5 GB VRAM + 11 GB RAM | ~30 s |
| KV cache | 16384 tokens × K/V heads 量化到 Q4_0 | 走 WDDM 共享池 |
| MoE FFN | **offload 到 CPU** (--moe-backend=offload) | 每次 decode 触发 ~80 ms |
| MTP | 启用但 K=3 时全 reject | 浪费 |
| 调度 | C++ 已优化，但单 req 没并发 | 无 batching |

**单 decode 步 ~167 ms**（6 t/s）拆解（粗估）：
- MoE offload (CPU 计算): ~80 ms
- 完整自注意力 + GDN: ~50 ms
- CUDA graph launch overhead: ~10 ms
- Python + CPU 调度: ~15 ms
- 其他 (sample, KV 写回): ~12 ms

**目标 50 t/s = 20 ms/step**，要在每个环节都砍。

---

## 第 5 章：项目代码地图 — 在哪里改什么

### 5.1 三层架构

```
┌─────────────────────────────────────────────────────────┐
│ HTTP Server (FastAPI)                                    │
│ python/freetoken/server/                                 │
│   - app.py: FastAPI 路由                                 │
│   - function_call_parser.py                              │
└─────────────────────────────────────────────────────────┘
           ↓ ZMQ
┌─────────────────────────────────────────────────────────┐
│ Daemon + Frontend                                         │
│ python/freetoken/daemon/                                │
│   - daemon.py: 主进程, 管 engine 生命周期                │
│   - panel.{html,js,css}: 前端 (Tao 排除)                 │
└─────────────────────────────────────────────────────────┘
           ↓ in-process Engine (subprocess)
┌─────────────────────────────────────────────────────────┐
│ Backend Worker                                           │
│ python/freetoken/cli.py (daemon 模式)                   │
│   ↓                                                     │
│ Scheduler (编排)  ──→  Engine (执行)  ──→  Model.forward │
│                                                              │
│ scheduler/scheduler.py     engine/engine.py    models/qwen3_5_moe/model.py
│   _prepare_batch              forward_batch       forward
│   _forward                    sample              attention + MoE
│   _process_forward_output
│   overlap_loop
└─────────────────────────────────────────────────────────┘
```

### 5.2 关键文件清单（按调用频度排序）

| 频度 | 文件 | 关键函数 |
|---|---|---|
| **每 decode 步** | `scheduler/scheduler.py` | `_prepare_batch`, `_forward`, `_process_forward_output`, `overlap_loop` |
| **每 decode 步** | `engine/engine.py` | `forward_batch`, `sample` |
| **每 decode 步** | `models/qwen3_5_moe/model.py` | `forward` (40 层循环) |
| **每 decode 步** | `models/qwen3_5_moe/mtp.py` | `forward_with_state` (MTP head) |
| **每 decode 步** | `attention/linear.py` | `prepare_metadata`, `forward` |
| **每 prefill** | `attention/{triton,trtllm,...}.py` | 各 backend 的 `prepare_metadata` |
| **每 N 步** | `cache.py`, `decode.py`, `prefill.py` | 调度策略 |
| **每次启动** | `engine/engine.py` `_load_weights` | 权重装载 |
| **不频繁** | `server/app.py` | HTTP |

### 5.3 C++ 已优化清单（已完成）

```
python/freetoken/scheduler/csrc/
├── sched_index.h          # 13 函数声明
├── sched_index.cpp        # 实现 (~320 行)
└── pybind_module.cpp      # pybind11 绑定
```

已替换的 Python 热路径：
- `make_input` / `make_write` → `_make_input_tuple` / `_make_write_tuple`
- `make_positions` → `_make_positions` (torch.arange loop)
- `build_decode_fla_meta` / `build_prefill_fla_meta` → `build_fla_metadata`
- `restore_linear_states` → `_restore_linear_states`
- `accept_count` → `MtpDriver.accept_count`
- `gpu_int_to_cpu_list` → `.to(torch.int32).tolist()`
- `build_linear_table_idx_decode_hybrid` → 调度器的 hybrid 分支
- `build_mtp_verify_meta` → 4-tensor 分配
- `write_tokens` → token_pool scatter

---

## 第 6 章：优化杠杆 — 从下到上分 5 层

### 6.1 杠杆 A：硬件层（最少能动的）

| 优化项 | 当前状态 | 收益预期 | 难度 |
|---|---|---|---|
| 切 Linux (UVM 替代 WDDM) | Windows WDDM | 2x 翻页延迟消失 | 高（环境换平台） |
| 升级 VRAM (买 16GB 卡) | 8 GB | 权重全留 VRAM, 省 PCIe | 高（硬件） |
| 用 SDD 替代 HDD | 已 NVMe | 无 | - |
| 关闭其他 GPU 进程 | - | 显存碎片回收 | 低 |

**结论**：硬件层已到顶。主要优化空间在软件。

### 6.2 杠杆 B：模型加载 + 内存布局

**当前问题**：35B 权重 17.5 GB，VRAM 只有 8 GB。要么分页换入换出，要么 CPU 计算。

| 优化项 | 当前 | 备选 | 预期收益 |
|---|---|---|---|
| 权重分页策略 | naive (装不下就报) | **predictive prefetch** — 根据 layer 顺序预读下一层权重到 VRAM | 30~50% |
| 权重 layout | 通用 (NX, K) for GEMM | **block-quant 适配 kernel** (MXFP4 直接 matmul) | 1.5~2x |
| LM Head 量化 | BF16 (152064×2048×2B = 600 MB) | NVFP4 (~150 MB) | 省 VRAM，量化速度 |
| MoE expert 分组 cache | --moe-backend=offload 全 CPU | **iGPU HIP server** (commit 2ec0a85 已实现) | 3~5x MoE 速度 |
| KV cache 量化 | q4_0 已用 | 试试 q2_K 减半 | VRAM 但影响质量 |

**重点押注：MoE iGPU offload** — commit 55af654 显示"Form-2 GTT 全层驻留 35 t/s standalone"但"engine in-process HIP writes dead after CUDA init"。这是重点。

### 6.3 杠杆 C：推理算法 — 单 token 计算

**Decode 单步 167 ms 拆解**：

```
完整自注意力 (full attn)     ~30 ms   ← 可以用 GDN 替换但有质量风险
GDN 线性层 (linear)         ~15 ms   ← 已 O(n)，基本到顶
MoE FFN (256 选 8)            ~80 ms   ← 最大瓶颈
LM Head + Sample              ~5 ms
Router (gate)                 ~2 ms
Embedding                     ~1 ms
CUDA Graph dispatch          ~10 ms
KV cache 写回                  ~5 ms
调度 + Python                 ~15 ms
其他 (mtp logits, draft)        ~4 ms
```

| 优化项 | 当前 | 备选 | 预期收益 |
|---|---|---|---|
| **MoE offload → iGPU** | CPU 计算 | HIP shared-memory pool (commit 2ec0a85) | **3x** MoE 速度 |
| **Form-2 GTT 全层驻留** | VRAM/RAM 分层 | 把整个模型塞进 iGPU 共享内存 (commit 55af654) | **5x** standalone，但需解决 in-process HIP 冲突 |
| **Block-quant GEMM kernel** | 通用 MXFP4 反量化 | Tritron kernel 直接读 4-bit 做 matmul | 1.3~1.5x |
| **Persistent kernel** | 每个 decode 起新 kernel | 单个 kernel 持续推理 (类似 vLLM 早期方案) | 省 launch overhead |
| **Continuous batching** | 1 req 1 batch | 4 req 并发 → 权重摊销 | **4x** 理论上限 (受 VRAM 限制) |
| **Speculative decode** | MTP K=1 工作 (K=3 死) | 修复 K=2+ 让 1 verify 接受多个 draft | **2x** (K=2 成功) 或 **3x** (K=3 成功) |
| **Attention kernel** | triton flash-attn | trtllm / flash-attn-3 | 1.2~1.5x |

### 6.4 杠杆 D：调度层 — Python 开销

**当前**：单 req decode ~15 ms Python 开销。

| 优化项 | 当前 | 备选 | 预期收益 |
|---|---|---|---|
| **更多 C++ 化** | 已 P0-P10 (13 函数) | 继续 P11+ (`_build_track_metadata`, `_process_forward_output` reply loop, `_schedule_next_batch`) | 5~10 ms/step |
| **CUDA Graph capture** | 已捕获 [1,2,4] | 捕获更大 batch size | launch overhead 减半 |
| **Overlapping** | overlap_loop 已有 | 让 MoE (CPU) 和 attention (GPU) 真并行 | 节省 MoE 时长 |
| **Skip synchronization** | 每步 `.item()` 同步 | 用 pinned host + non-blocking | 已有 P5 |
| **NumPy/zero-copy** | torch tensor | 直接 numpy 或 C array | 边际 |

### 6.5 杠杆 E：MTP / 多 token 生成

**当前**：K=1 accept 30%, K=3 全 reject。

| 优化项 | 当前 | 备选 | 预期收益 |
|---|---|---|---|
| **修复 K=2+ accept** | K=1 only | 调查 hybrid GDN final-state-only snap 的 all-or-nothing 问题 | K=3 成功 → 3x |
| **MTP head KV cache** | 用主模型的 KV | 给 MTP 独立 KV cache (commit 3108df6 验证 <3% delta) | 边际 |
| **Tree draft** | linear chain | tree-structured draft + verify | 5x+ |
| **Multi-token prediction training** | 单 token 训练 | 让模型原生多 token | 训练阶段 |

---

## 第 7 章：从 6 t/s 到 50 t/s 的可行路径（推荐顺序）

### 路线 1：MTP 路径（最便宜，可能拿到 2~3x）
1. 修 K=2+ accept — 调查 `_mtp_process_verify` 为什么不接受 d2/d3
2. 验证：commit msg 说 "d2 self-fed never hits" — 可能要改 draft 循环结构

### 路线 2：iGPU MoE 路径（架构改动，需要排查）
1. 解决 in-process HIP writes deadlock（commit 55af654 已知问题）
2. 把 CPU offload 切到 iGPU HIP server
3. 验证：35 t/s standalone，但 in-process 还有 3.7 t/s 的问题

### 路线 3：Batching 路径（内存吃紧，理论上限 4x）
1. 实现真正的 continuous batching
2. 多 req 共享权重，权重搬运摊销
3. 挑战：8 GB VRAM 装不下 2 req 的 KV + 中间激活

### 路线 4：模型加载优化（边际但稳定）
1. Predict prefetch — 下一层权重提前 50 ms 预热
2. Block-quant GEMM kernel 适配 MXFP4
3. LM Head NVFP4 量化

---

## 第 8 章：优化设计的具体检查清单

下次决定改什么时，按这个顺序判断：

1. **这个改动在 hot path 吗？**（每 decode 步都跑？还是每 prefill？还是偶尔？）
   - 每 decode 步 → 优先做（影响 6 t/s → 50 t/s 的核心）
   - 每 prefill → 次之
   - 偶尔 → 推迟

2. **这个改动省什么？**
   - **省带宽**（减少 VRAM 读写）→ 通常最有效
   - **省算力**（减少 FLOPS）→ 看 kernel 选型
   - **省 Python**（减少调度开销）→ 当前 15 ms / 167 ms ≈ 9%，到顶了
   - **省同步**（减少 .item()、.cpu()）→ 已有 P5 优化

3. **这个改动吃多少 VRAM？**
   - 加 CUDA graph buffer → 测可用
   - 加 persistent kernel workspace → 测可用
   - VRAM 满 → 失败（必须等权重释放）

4. **这个改动有正确性验证吗？**
   - 单 req decode + greedy + 短 prompt → 看输出文本是否合理
   - MTP accept rate (0~1) → 看是否改善
   - 数值对比 vs baseline commit → 看是否回归

5. **commit 链里有没有相关讨论？**
   - `git log --grep="mtp"` — MTP 相关
   - `git log --grep="moe"` — MoE 相关
   - `git log --grep="hip"` — iGPU 相关
   - `git log --grep="kv"` — KV cache 相关

---

## 第 9 章：参考资料

- 项目内部：`README.md`, `CORE_DESIGN.md`, `PHASES.md`, `MEMORY_BUDGET.md`, `IGPU_MOE_ARCH.md`, `FORM2_GTT_RESIDENCY_REPORT.md`, `IGPU_ZEROCOPY_VERDICT.md`
- MTP 修复：commit `93c431c` (re-applied as `4bb2b4e`), revert `9f6e52f`
- 量化修复：commit `3108df6` (FC weight uint4-affine)
- iGPU 路径：commits `2ec0a85` (engine integration), `55af654` (35 t/s standalone)

---

**最后一句话**：50 t/s 在 8 GB VRAM 上是**接近硬件极限**的目标。当前 6 t/s 的 8x 空间，主要来自：
1. **MoE offload 的 CPU 计算**（最大头，iGPU 化能拿 3x）
2. **MTP 没工作**（K=3 全 reject 是 bug，修了能拿 1.5~2x）
3. **单 req 没 batching**（要重新设计调度 + 内存）

任何一个单独解决都不够。**至少 3 个同时做才到 50 t/s**。