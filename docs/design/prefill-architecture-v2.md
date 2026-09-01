# Prefill Architecture v2 — 主流大模型引擎 Prefill 优化方案

> 设计文档 · 目标：解决 FreeToken（及类似大模型引擎）prefill 时间过长问题。
> 现状：35B-A3B 模型上一条 8k prompt prefill 可达 200ms-2s，期间所有 decode 被阻塞（_schedule_next_batch 行 841-844 是 prefill-OR-decode 互斥），这是单请求下 TTFT 长、混合负载下 decode 卡顿的主因。

---

## 0. 现状摘要（已读代码确认）

| 阶段 | 现状 | 行号 |
|---|---|---|
| 调度 | _schedule_next_batch：prefill-OR-decode 互斥（prefill 先，否则 decode），无 mixed batch | scheduler.py 839-849 |
| Chunking | max_extend_tokens=8192 chunk，超出则 ChunkedReq 续跑 | prefill.py 126-189 |
| Prefix cache | radix match + cached_len 复用 handle.page_table 行 | prefill.py 55-103 |
| Attention kernel | extend_paged_attention（短）/ paged_attention（长，单请求）；triton 块瓦片 | attention.py 761+883 |
| Overlap | _overlap_loop：下一批调度与上一批结果 drain 重叠（CPU-side）。MoE 专家 D2D prefetch 在 prefill 期间进行 | engine.py 678+; moe/offload_cache.py 316+ |
| 流 | self.stream (host) 与 engine.stream (device) 互 wait；GDN 状态 COW 在 engine stream | scheduler.py 231-251 |
| Hybrid GDN | full_attention_interval=4 → 16 full + 24 GDN；GDN chunked prefill 复杂（chunked carry state）| prefill.py 144-158 swa；fla_metadata build |
| 存储 | MoE expert offload CPU/iGPU；KV bf16→q8_0（本次新加）| engine.py 530+ |

关键瓶颈（按影响降序）：
1. 阻塞式调度——一条 prefill 期间所有 decode 不能前进 → 高并发时 decode jitter 极大
2. 单 forward per step——没有混合 batch（prefill + decode 同 forward），无法利用 decode 显存空闲
3. MoE prefill 不是真正异步——moe_prefill_overlap 只是把专家从 PCIe 重叠到 GPU 计算，CPU 端 schedule 仍阻塞
4. Prefix cache radix 复用粒度——一旦 token 不命中，整条 prompt 重做（虽然 handle.cached_len 正确传递）
5. Chunked prefill GDN carry 复杂——chunk_size 受 SWA / token_budget 影响，每次切 chunk 都有 carry 开销
6. Linear attention scan——GDN 层 chunked prefill 用 sequential scan，不能并行（vs FlashAttn 全并行）
7. Linear state pool 锁——linear_slot_idx + ping_pong 串行化处理，hybrid 模型多个 chunk 限制并发

---

## 1. 设计目标与权衡

目标（数字可验证）：
- TTFT(8k prompt) P99 ≤ 200ms（现状 ~1s）
- decode 抖动（P99-P50 of decode interval）在混合负载下 ≤ 2x（现状 ~10x+）
- prefill throughput (tokens/s) 单 batch ≥ 8000 tok/s（35B 在 dGPU + iGPU MoE 下）
- hybrid GDN 模型的 prefill chunk 切分开销 < 5%

约束（不能破）：
- TTFT 不退化（这是 prefill 优化的核心 KPI）
- 显存占用不退化
- 与 KV q8_0 / iGPU MoE / MTP 推测协同（MTP 受益于 prefill 越快越好——首字延迟直接受益）
- 不引入跨主机/跨进程开销（保持单机引擎）

---

## 2. 业界方案对比（精炼）

| 方案 | SGLang | vLLM v0.6+ | TRT-LLM | TGI | FlashAttn |
|---|---|---|---|---|---|
| Mixed batch (chunked) | ✅ RadixAttention + mixed_chunk | ✅ enable_chunked_prefill | ✅ enable_chunked_context | ✗ | — |
| Disaggregated prefill | ✅ DistServe | ✅ (实验性) | ✗ | ✗ | — |
| Spec prefill | ✗ | ✗ | ✅ draft-prompt | ✗ | — |
| Prefix cache radix | ✅ tree radix | ✅ block table | ✗ (PromptCache) | ✅ | — |
| Pre-allocated KV pool | ✅ | ✅ | ✅ | ✅ | — |
| CUDA Graph hybrid | ✅ chunked-prefill graph | ✅ | ✅ | ✓ | — |
| Async prefill stream | ✅ DistServe | (实验性) | ✗ | ✗ | — |

FreeToken 当前 ≈ SGLang 2023 年的能力，但 SGLang 2024 后已上 DistServe。

---

## 3. FreeToken-Prefill-v2 架构（4 模块改造）

### 3.1 模块总览

```
┌────────────────────────────────────────────────────────────────────┐
│                    Scheduler (overlap_loop)                        │
│  ┌──────────────┐  ┌──────────────────┐  ┌────────────────────┐   │
│  │ PrefillMgr   │  │ DecodeMgr        │  │ PrefillStreamer    │   │
│  │ (现有 chunk) │  │ (现有 running)   │  │ (新) 异步 prefill   │   │
│  └──────┬───────┘  └────────┬─────────┘  └──────────┬─────────┘   │
│         │                   │                       │             │
│         └───────────────────┴───────────────────────┘             │
│                             │                                     │
│                   _schedule_next_batch_v2                         │
│                             │                                     │
│           ┌─────────────────┼──────────────────────┐              │
│           │                 │                      │              │
│   ┌───────▼────────┐ ┌──────▼─────────┐ ┌─────────▼──────────┐   │
│   │ Prefill-only   │ │ Decode-only    │ │ Mixed chunked-prefill│   │
│   │ batch          │ │ batch          │ │ batch (P+D 同 forward)│  │
│   └────────────────┘ └────────────────┘ └─────────────────────┘   │
│                             │                                     │
│                       engine.forward_batch                         │
└────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌────────────────────────────────────────────────────────────────────┐
│                    Engine.forward_batch_v2                         │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │ 1. Build mixed cu_seqlens:                                  │    │
│  │    prefill reqs: seqlens_q=extend_len, seqlens_k=device_len │    │
│  │    decode  reqs: seqlens_q=1, seqlens_k=device_len         │    │
│  │ 2. Run extend_paged_attention (covers both prefill+decode  │    │
│  │    via block-tile causal attention; decode becomes Q=1     │    │
│  │    row at the tail, prefill gets full extend)               │    │
│  │ 3. lm_head: gather all decode reqs + prefill reqs'         │    │
│  │    last-token position                                      │    │
│  │ 4. Sampler: only decode reqs get sampled tokens;            │    │
│  │    prefill reqs get cached (no sample) for this pass        │    │
│  │ 5. Reuse free prefill schedule after consumed token budget │    │
│  └────────────────────────────────────────────────────────────┘    │
└────────────────────────────────────────────────────────────────────┘
```

### 3.2 模块 A：Mixed Chunked-Prefill Decode Batch

目标：一次 forward 同时处理 prefill chunks + decode reqs；只要 prefill chunk ≤ 某 token 上限（如 1024），decode 全员同 forward。

改动：
- Batch.phase 改为可选 "mixed"（prefill chunk + decode reqs 共存）
- _prepare_batch：
  - 对 mixed batch，构造 cu_seqlens_q = [prefill_extend_len, 1, 1, ...]，cu_seqlens_k = [prefill_device_len, d_device_len, ...]
  - prefill reqs 的 positions = arange(cached_len, device_len)，decode reqs 的 positions = [device_len-1]
  - lm_head：prefill reqs 取 last-position 行；decode reqs 取每 req 末行；统一一次 gather
- extend_paged_attention 已经在 Q block=128 上跑；混合时 decode Q=1 行退化为 128 块中 1 有效列，浪费约 128x 但每个 decode req 一行，开销 ≪ prefill 总开销
- Sampler：把 prefill reqs 标 skip_sample=True（sample_args 里加 flag）；只 decode reqs 出 token
- ChunkedReq 现在能进 mixed batch——这是关键：当前 ChunkedReq 在 prefill manager 内续跑，decode 期间不能推进；mixed batch 让 ChunkedReq 与 decode 并行推进，TTFT 更稳

约束：
- 显存：mixed batch 的 prefill chunk 多了 KV cache 占用（一次），通过 _kv_usage_pages 已经计入
- 调度公平：当前 decode reqs 已有 inflight_tokens 保护；混合批让 prefill 借走 decode 时槽，需新加 prefill_decode_overlap_budget（如 1024 tokens）
- 不破 GDN：混合批 GDN 状态更新时只更新 decode reqs（prefill chunk 走现有 carry 路径）

代码骨架（伪）：
```python
# scheduler.py
def _schedule_next_batch_v2(self):
    decode_batch = self.decode_manager.schedule_next_batch()
    prefill_chunk = self.prefill_manager.peek_next_chunk(self.prefill_budget)
    if decode_batch is None and prefill_chunk is None:
        return None
    if decode_batch is None:
        return self._prepare_batch(prefill_chunk)
    if prefill_chunk is None or prefill_chunk.extend_total > self._mixed_chunk_cap:
        return self._prepare_batch(decode_batch)
    # mixed batch
    return self._prepare_mixed_batch(decode_batch, prefill_chunk)
```

### 3.3 模块 B：PrefillStreamer（异步预填流）

目标：长 prefill（≥ 8k）进一步切成多个小 chunk，每分与后续 decode 并行推进，TTFT 由首块决定。

当前现状：ChunkedReq 已经能切 chunk，但每个 chunk 之间要等下一轮调度——延迟是「chunk 数 × 调度间隔」。在 overlap_loop 里是「下一批调度的 forward time + 上一批 drain 的 host time」。

改造：
- _mixed_chunk_cap = 1024 tokens（一次 forward 最多 prefill 这么多）
- 一条 8k prompt 拆成 8 个 1k chunks，每 chunk ~50ms（35B 在 dGPU + iGPU MoE），8 chunks 串行 = 400ms
- 但每 chunk 与 decode 同步推进（mixed batch），总 TTFT ≈ 首 chunk 完成 + 第一 token 解码时间（~50ms decode）→ TTFT ≈ 100ms（vs 现状 800ms+）
- 控制：_pending_prefill_chunks 记录一个 req 的剩余 chunks；scheduler 在 chunked req 续跑时复用 mixed batch

额外收益：
- 短 prompt（< 1k）单 chunk，行为不变
- 中 prompt（1k–8k）2-8 chunks，自然走 mixed
- 长 prompt（> 8k）8+ chunks，TTFT 极快，但总 prefill 时间线性（符合预期）

### 3.4 模块 C：Async Prefill Stream（轻 DistServe）

目标：长 prompt（> 32k）第一次解码响应延迟主导；引入独立 prefill stream，让 prefill 和 decode 在 GPU 上真正物理并行（不是混合 batch 那种 time-sharing）。

为什么需要这个：mixed chunked 已经够 8k prompt，但 > 32k prompt 仍要 ~400ms，TTFT P99 不达标。DistServe (SOSP'24) 证明 prefill 内存带宽 vs decode 计算敏感，预填独立引擎 GPU 利用率更高。

FreeToken-specific 实现（轻量）：
- 不引入第二个进程，而是复用现有 engine_stream_ctx——为 prefill 分配一个独立 CUDA stream
- 调度：长 prompt 一旦进入 prefill manager，立即提交到 prefill stream，与 decode 在不同 stream 并行
- 显存：prefill 占独立 KV 槽 + 共享 MoE expert pool（已有 overlap 机制）；decode 用自己的 KV 槽
- 同步点：prefill KV 写完 → 切换 req 模式（prefill → decode）；这个同步走 event-wait 而不是 stream 全 wait——DecodeReq 仍在跑

约束：
- 显存：两条流共享一个 CUDA context，KV 池需要为 prefill 预留槽；新加 prefill_kv_reserve_tokens 配置
- CPU MoE executor：仍然串行；expert offload 是 PCIe-bound，多 stream 不增加吞吐
- 复杂度：实现 + 集成测试成本高，先做模块 A 和 B，C 是 1.0+ 路线

### 3.5 模块 D：Prefix Cache 升级（双层 + 前缀压缩）

当前：radix 树 prefix cache，按 token 块匹配；命中后整段 reuse KV handle。

痛点：
1. 系统提示（system prompt）经常 200–500 tokens 重复——每个请求都重做 KV 浪费
2. 多轮对话的 n-1 轮前缀——经常命中但中间插入 tool call 后整段作废

升级方案：
- 双层 radix：
  - L1：系统提示 cache（持久，跨用户共享）—— boot 时填，LRU 保留
  - L2：会话级 cache（按用户/session）
  - 命中规则：拼接 L1 + L2，radix match 时优先 L1
- Prefix 压缩：对长 system prompt（> 1000 tokens）做 prompt embedding 预计算（用 MTP 头 hidden state 第一个；或单独一个 2-layer prefix encoder）；KV handle 存储压缩 latent；decode 时按需解压缩
- 失效机制：tool_call 之后把 system prompt 提到 L1（永久），chat 部分按 L2 处理

实现成本：
- L1 cache：~200 LOC（boot-time fill + LRU + handle.tenant）
- Prefix 压缩：~500 LOC（encoder + 适配 attention 路径）——风险高，建议 1.1+ 路线

### 3.6 模块 E：GDN Chunked Prefill 加速（hybrid 专项）

当前痛点：hybrid GDN 模型 prefill 必须 sequential scan（vs FlashAttn 全并行）；chunked 时每个 chunk 都要 carry hidden state 来回 fetch，开销线性。

加速方案：
- Chunk size 与 L2 cache 对齐：GDN scan 块大小 = 32（业内标准），与 extend_paged_attention block_m=128 解耦；混合批里 GDN 走 32-token 子块
- Ping-pong 优化：当前 ping_pong 是 (slot_a, slot_b) 双缓冲——扩展为 4-slot 双工（per-chunk carry slot ×2 + output slot ×2），消除 chunk 边界 wait
- Skip chunk on cache hit：GDN 层 prefix 命中时跳过该 chunk 的 scan（前提是 state 已经 COW-restored）
- State quantization：GDN state fp32 → bf16 减半（不损失精度，state 在迭代中已经被 RMSNorm 归一化）

实现：~300 LOC，hybrid 模型收益 >30% prefill 加速。

### 3.7 模块 F：调度公平 + Head-of-line 缓解

当前问题：长 prompt 占 prefill manager，prefill budget 被吃光，短 prompt 也要排长队——典型的 head-of-line blocking。

改造：
- Preemption：长 prompt prefill 中途若被新短 prompt 抢，让长 prompt 退到 prefill_budget 末尾（保留 ChunkedReq.cached_len）
- Priority queue：prefill manager 内按 (prompt_len, arrival_time) 排序——SJF + 老化，避免饥饿
- Burst 模式：当 decode backlog > N 时，自动把 prefill chunk_cap 临时减半（如 1024 → 512），让 decode 优先
- Admission control：当 active reqs ≥ max_running_req - headroom，新 prompt 立刻 fail-fast 返回 503（让客户端重试到其他实例）

实现：~200 LOC + 新配置 --prefill-scheduling-policy {fifo|sjf|burst}。

---

## 4. 实施路线图

| 阶段 | 模块 | 期望收益 | 实现成本 | 风险 |
|---|---|---|---|---|
| P0（本次 PR 后续） | F 调度公平 (部分) | 短 prompt P99 ↓ 50% | 1–2 天 | 低 |
| P1（1 周） | A Mixed Batch + B PrefillStreamer | TTFT P99 ↓ 70%；decode jitter ↓ 80% | 1–2 周 | 中（关键路径） |
| P2（2 周） | E GDN 加速 | hybrid 模型 prefill ↑ 30% | 1 周 | 中（hybrid 专项） |
| P3（1 月） | D Prefix Cache 双层 | TTFT(有 system prompt) ↓ 60% | 2 周 | 低 |
| P4（远期） | C Async Prefill Stream | 长 prompt > 32k TTFT ↓ 50% | 3 周 | 高 |
| P5（远期） | D Prefix 压缩 | 极大共享 prompt 场景 | 1 月+ | 高（模型相关） |

---

## 5. 与现有特性的协同

已有特性 | 与本设计的协同
---|---
MTP 推测解码（本次新加） | MTP 的 TTFT 直接受益于 P1（混合批 prefill 首块快）；MTP draft 不影响 prefill 路径
KV q8_0 量化（本次新加） | 量化 KV 让更多 prefill chunk 同 forward（KV 显存省一半 → _mixed_chunk_cap 可放宽）；验证时一起 bench
iGPU MoE offload | prefill 期间 CPU offload 已是 PCIe-bound；P1 mixed batch 不增加 PCIe 争用（decode reqs 也走 CPU MoE）
CUDA Graph 禁用（graph.py:75） | 简化了 mixed batch 实现——CUDA graph 不能捕 mixed batch 的 shape 不固定，省了一个 capture path
hybrid GDN | E 模块专项；其余模块尽量避免 GDN 路径开销
MoE prefill overlap | 已经存在；P1 不与之冲突，反而 mixed batch 时 prefill req 的 MoE 与 decode req 的 MoE 可以并发预取

---

## 6. 度量与验证

Baseline（本设计之前）：
- TTFT P50/P90/P99: 多 prompt length 曲线
- decode 抖动：负载测试（混合 50% prefill + 50% decode）

Target（设计之后）：
- TTFT P99(8k prompt) ≤ 200ms
- decode interval P99-P50 ≤ 2x
- prefill throughput ≥ 8000 tok/s/req（35B 模型）
- 0 回归：CUDA graph 仍然禁用、MTP 仍然工作、KV 量化仍然工作

Bench 脚本建议（benchmarks/cpu_moe_microbench/t_prefill_v2_bench.py）：
- 用真实模型（35B MXFP4-MTP）
- 三种负载：(a) 单 prefill，(b) prefill + decode 混合，(c) 多 prefill 并发
- 比较 v0（现状）、v1（仅 F）、v2（F + A+B）、v3（+ E）

---

## 7. 风险登记

风险 | 触发 | 缓解
---|---|---
Mixed batch 显存 OOM | decode backlog 加上 prefill chunk 超过 KV | 调 _mixed_chunk_cap；KV 量化（q8_0）
Chunk 切太细 overhead > 收益 | 短 prompt 不必要切 | _min_chunk_tokens=256，短 prompt 单 chunk
GDN carry 丢失 | chunked prefill 跨 chunk 状态错 | 必须严格沿用现有 fla_metadata build；新加单元测试
Prefix cache 误命中 | token 级别误匹配 | token-level MD5 checksum
异步 prefill stream 与 decode 死锁 | event wait 顺序 | 现有 stream.wait_stream 模式扩展，明确记录
调度优先级导致饥饿 | SJF 让长 prompt 永远后排 | 老化因子（wait_time / prompt_len）

---

## 8. 备选：保持现状的 micro-optimizations

如果决定不引入架构改造，还有一些微优化可立即做（无架构风险）：
1. CUDA Graph 重新启用（graph.py:75 之前是为了避免 prefill shape 不固定；P1 引入 mixed batch 反而稳定了 shape，可能能再次启用——但要重新做 graph capture）
2. extend_paged_attention 块大小调优：当前 _select_extend_tile 按 head_dim 选；可加 prefill vs decode 区分（prefill 用更大 block_m）
3. Prefix cache 启用 L1 提示系统 cache（仅模块 D 的简化版）
4. Attention kernel 的 fused rope + quant（与本次 KV q8_0 集成）
5. CPU MoE executor 的 prefill 批预填——目前 decode 才有 batch fetch；prefill 阶段也走 batch fetch（PCIe 复用）

---

## 9. 结论

FreeToken prefill 当前的瓶颈是单 forward / 阻塞调度 / 单一前缀 cache——这是 2023 年级别的问题，不是 GPU kernel 慢（kernel 已经是 triton 优化版）。

改造 ROI 排序：
1. P1: Mixed Batch + PrefillStreamer——核心收益 70% TTFT ↓，1-2 周可做
2. P2: GDN 加速——hybrid 模型专项，hybrid 是 FreeToken 主战场（35B 是 hybrid）
3. P0: 调度公平——低风险立即收益
4. P3-P5: 长期——Prefix 双层、异步 stream、压缩——按需推进

先做 P0 + P1，预计 2 周内 TTFT P99 从 1s 降到 200ms 内。
