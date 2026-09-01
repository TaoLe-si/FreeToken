# Prefill TTFT Architecture v3 — 减不必要计算 + 计算/IO 并行

> 设计文档（不动代码）· 目标：把 FreeToken 首 Token 延迟（TTFT）从 P99 ~1s 降到 ≤ 200ms。
> 主轴：减少不必要计算（用户策略二）+ 提升计算与 I/O 并行效率（用户策略一·Cache Cake 思想）。
> 与 v2 文档（prefill-architecture-v2.md，聚焦调度/混合批）互补：v3 聚焦"避免做无用的功 + 把有用的功做快"，v2 聚焦"批结构与调度公平"。
> FreeToken 现状已确认（35B-A3B hybrid GDN、CPU MoE offload、KV bf16/q8_0、MTP 已接入、CUDA graph 禁用）。

---

## 0. 现状锚点（FreeToken-specific）

读完代码后的事实基础：
- KV 复用：radix 树 prefix cache 已实现（prefill.py:55-103 的 match_req），但仅 L2 单层；系统提示等高频重复 prompt 仍每次重算 KV
- Chunked prefill：max_extend_tokens=8192 chunk + ChunkedReq 续跑（prefill.py:126-189）
- Mixed batch：当前不存在（v2 设计的模块 A）
- MoE：CPU/iGPU offload，prefill 期间 expert prefetch（moe/offload_cache.py:316+）
- 存储：KV bf16 / q8_0；KV 可放 CPU RAM（kv_device=cpu），PCIe spill
- GDN：hybrid 模型 16 full + 24 GDN；GDN prefill 必须 sequential scan（vs FlashAttn 全并行）
- MTP head：本次新加，可做"轻量预评估"（潜在 SpecPrefill 落点）
- CUDA graph：禁用（graph.py:75），故无需担心混合 shape 捕获

---

## 1. 策略一落地：KV Cache 复用与扩展（命中即可省计算）

### 1.1 L1/L2 双层 Prefix Cache（用户策略一·Prefix Caching + 分级）

现状：单层 radix 会话级 cache；系统提示每次重算。

FreeToken 落点：
- cache.py:CacheManager（base class + radix 实现）拆成两层：
  - L1：系统提示 cache——prompt_tokens 启动时填入，跨用户共享，按 tenant_id=None 标识
  - L2：会话级 cache——按 tenant_id + session_id 标识
- 匹配顺序：request.input_ids = L1_prefix ++ L2_match ++ remainder，radix match 先 L1 再 L2
- LRU 淘汰：L1 大小按显存量配置（建议 4 GB 起步），L2 按 working set

API 改造（不动代码——只描述）：
```python
@dataclass
class PrefixCacheEntry:
    tenant_id: str | None      # None = L1（系统提示），否则 L2
    session_id: str | None
    cached_len: int
    handle: BaseCacheHandle
    last_used: float

class CacheManager:
    def match_prefix(self, input_ids, tenant_id=None) -> MatchResult:
        """Match L1 first (system prompts), then L2 (session)."""
```

收益：
- 系统提示重复场景（chat 类应用最常见）：TTFT 几乎为 0（首字延迟只剩 decode 一步）
- 多轮对话：第二轮起 90%+ 前缀复用（vs 现状 0%）
- 与 KV q8_0 协同：L1 cache 在内存（RAM）持久（不抢显存），系统提示 cache 命中率提升 → KV 显存压力更小

实施风险：低；200 LOC 改动，主要是数据结构与淘汰策略

---

### 1.2 KV Cache 扩展到 DRAM（用户策略一·分级/分布式 KV）

现状：kv_device=cpu 时 KV 在主机 RAM（engine.py:469+），但 CPU KV 与 GPU KV 是互斥状态，无法同时利用。

FreeToken 落点（Cake 系统思想：计算与 I/O 并行）：
- 引入 GPU HBM L0 + Host RAM L1 + NVMe L2 三级 KV 池
- 调度器为每条 request 选择 KV 驻留 tier：
  - L0（HBM）：hot path（当前 decode 的 req）
  - L1（DRAM）：warm path（prefill 中 / chunked prefill）
  - L2（NVMe）：cold path（历史 prefix cache 仅作启动预热）
- 关键：prefill 命中 L1/L2 段时，启动 async H2D 复制，同时 GPU 计算未命中段——这是 Cake 系统的核心思想在 FreeToken 的实现

代码落点：
- kvcache/base.py：新增 BaseKVCachePool.tier 属性 + async_load_to_gpu(rows) API
- attention/triton.py：读取 GPU KV 时，触发 backend prepare_async_load(missing_rows)——并行 DMA
- engine/engine.py：forward_batch 调度器里检测 L1/L2 命中段，调度 async load + 计算并行

收益：
- 系统提示 L1 cache（1.1）放在 DRAM（L1 tier），命中时 GPU 边算 KV 边 load 老 KV，TTFT 进一步 ↓ 30-50%
- 极长 prompt（> 32k）历史 prefix 放 NVMe（L2），boot-time 预热

实施风险：中；需要新的内存池管理与 DMA 流水线；300-500 LOC

---

### 1.3 Cache 预热（用户策略一·Cache 预热）

场景：引擎重启后第一个请求是长 prompt + 系统提示，需要重头 prefill。

FreeToken 落点：
- 引擎启动时（engine.py:_warmup_prefill），自动跑一次 "system prompt warmup"：
  - 读 L1 cache 配置（若 daemon 设了 sSystemPromptCache 字段）
  - 把高频系统提示 tokenize + prefill 一次，把 KV 写到 L1 tier
  - 后续这些请求命中 L1，TTFT ≈ 0
- 该机制已经是部分现状（_warmup_prefill）的扩展——目前只跑 dummy；新版让 daemon 注入真实 system prompt

收益：
- 冷启动场景下首请求 TTFT 也接近命中状态
- 与 1.1 完美协同

实施风险：低；100 LOC + daemon 配置

---

## 2. 策略二落地：减少不必要计算（用户策略二核心）

### 2.1 静态稀疏注意力（用户策略二·稀疏注意力·静态稀疏·TriangleMix）

原理：深层 transformer 对中间 token 的依赖很低，深层用三角形掩码把 O(N²) → O(N)

FreeToken 适配性：
- Qwen3.6 hybrid GDN 模型：full_attention 层 16 个（layer 0/4/8/12/16/20/24/28/32/36 + 一些 full attn），其余 GDN
- 对 full_attention 的深层（layer 24+），可以应用 triangular sparse
- 关键：训练时要稀疏友好（不是所有模型都兼容）；Qwen3.6 已训过，要先验证

FreeToken 落点（先 design-time 评估，后 code-time）：
- kernel/triton/attention.py 的 _paged_attention_kernel 增加 sparse_mask 参数（三角形窗口）
- engine/config.py：sparse_prefill: str = "none" 选项（none/triangle-mix/...）
- models/qwen3_5_moe/model.py：在 forward 时根据 config.sparse_prefill 给 attention 层传 sparse_mask
- GDN 层跳过——GDN 不是注意力，已经是 linear complexity

收益：
- 长 prompt (> 4k) prefill：TTFT ↓ 12-32%（按 TriangleMix 论文）
- 短 prompt（< 1k）几乎无效（三角形窗口退化为全连接）

实施风险：中-高；需要先验证 Qwen3.6 在三角形稀疏下的精度损失（perplexity 测试）；如果损 > 1%，放弃

---

### 2.2 动态稀疏注意力（用户策略二·动态稀疏·Stem）

原理：用"初始 token 是信息流树干"启发式 + 学习重要性分数

FreeToken 适配性：
- 与 TriangleMix 互补：Stem 是 top-k 选择（k% 重要 token），TriangleMix 是结构稀疏
- Stem 算法需要"token importance predictor"——额外参数

FreeToken 落点：与 2.1 同，但成本更高；不推荐在 35B 模型上做（训练代价 + 精度风险）。列为远期。

收益：长 prompt TTFT ↓ 数倍（论文数据）

实施风险：高；需训练或蒸馏

---

### 2.3 投机式 Prefill（用户策略二·SpecPrefill / ICML 2025）

原理：用一个轻量级 draft model 评估 prompt 哪些 token 对最终输出重要，主模型只为 Top-k% 重要 token 算 KV

FreeToken 适配性 —— 天然契合：
- FreeToken 已有 MTP head（本次新加）—— 这就是"轻量级 draft model"
- MTP head 已经在做 autoregressive 预测，可以用它的"重要性分数"作为 prefill token 的 rank
- 设计：prefill 阶段，对每个 token 跑一次 MTP head forward（前向只看 prompt 当前位置的 embedding，无 KV），得到"如果是 next token，它有多确定"的分数（argmax 概率），分数低的 token 在主模型 prefill 时跳过 KV 计算，只保留 embed 流到下一层

具体 FreeToken 落点：
- models/qwen3_5_moe/mtp.py：扩展 forward_with_state 增加 score_only=True 模式——只跑 head 一次，返回 logits 的 argmax_prob（不需要 prev_hidden chain）
- scheduler/prefill.py：在 _add_one_req 时，对 prompt token 流式过 MTP head（单 forward 一次能算所有 token 的 score）→ 排序 → 决定 KV 计算的 token 子集
- models/qwen3_5_moe/model.py：forward 增加 token_subset_mask 参数，KV 只写到被选中的 token
- GDN 层：GDN scan 必须保留所有 token，否则 linear state 错——只在 full_attention 层用稀疏

收益：
- 长 prompt (> 8k)：TTFT ↓ 2-4x（SpecPrefill 论文报告）
- 与 MTP 协同：MTP head 的存在让这条策略免费上线——其他引擎需要单独训练 draft model

实施风险：中；KV 子集写入需要 attention kernel 适配（paged KV 只存部分 page）；精度损失需 verify

---

### 2.4 计算与 I/O 流水线（用户策略二·Cake 系统）

原理：当 KV Cache 部分命中时，GPU 一边从存储加载已缓存 KV，一边重算未命中部分

FreeToken 落点（已在 1.2 部分涉及）：
- 这是 v3 的核心技术——把 1.2 的 L1 tier 与 2.3 的 spec prefill 结合起来
- 流程：
```
T0: L1 hit for tokens [0..cached_len)
    T0+ε: scheduler 触发 async H2D DMA 把 L1 KV 搬到 GPU（独立 CUDA stream）
    T0+δ: 同时主模型开始 prefill [cached_len..input_len)（partial prefill）
    T0+τ: DMA 完成，partial prefill 也接近完成（合并计算）
T1: forward 返回 full logits，采样首 token
```
- 关键：DMA 走的独立 CUDA stream（prefill_stream），主模型在主 stream；同步点为 attention 读取 KV 那一刻（DMA 完成 event 被 wait）

代码落点：
- kvcache/base.py：BaseKVCachePool.async_load_to_gpu(rows) 返回 cuda.Event
- engine/engine.py：Engine.forward_batch 头部启动 async load，forward 后 wait
- attention/triton.py：attention kernel 读 KV 前确认 DMA event 已 wait

收益：
- 系统提示 1k tokens + 用户 prompt 4k tokens：TTFT 比 "全 prefill" 快 2-3x（DMA 与计算时间重叠）
- 与现有 moe_prefill_overlap 是同一思路（CPU 端），Cake-style 把这个推广到 I/O

实施风险：中；stream 同步管理复杂；500-800 LOC

---

## 3. 策略三落地：调度与并行（用户策略三）

### 3.1 Chunked Prefills + Mixed Batch（用户策略三·分块预填充）

现状：v2 文档已设计；v3 复用 v2 模块 A/B

v3 增量：
- v2 设计 _mixed_chunk_cap=1024，v3 提升到 2048（前提：KV 量化 q8_0 后显存减半）
- Dynamic chunk size 策略：prefill 初期（前 3 个 chunk）用大 chunk（2048）快速推进，后续（chunk 4+）用小 chunk（512）让 decode 优先——平衡首字延迟与吞吐
- 与 2.3 协同：每个 chunk 内部做 spec prefill，进一步降 TTFT

收益：
- 单 prefill TTFT 改善：v2 的 70%↓ + v3 的 spec 额外 30%↓
- 混合负载：v2 设计的 decode jitter ↓ 80% 仍成立

---

### 3.2 Context Parallelism（用户策略三·上下文并行）

原理：极长 prompt 沿序列维度切分到多个 GPU 并行算 KV

FreeToken 适配性：
- 当前 dGPU 是 AMD 单卡（8GB），不能 context parallel 跨 GPU
- 但可在单 GPU 内做 context parallel across heads：head_dim 维度已有 KV heads 分组（GQA）
- 真正有意义的并行是单 GPU 内的 sequence-dim pipeline：把 8k prompt 切成 4×2k，让 pipeline 起来

FreeToken 落点：
- 单 GPU 内的 sequence pipeline：prefill manager 把 chunked req 切成多个 pipeline stage，stage 间通过 CUDA event 同步
- kernel/triton/attention.py 的 extend_paged_attention 增加 pipeline-aware 调度
- 与 GDN 协同：GDN scan 本来就是 sequential，pipeline 收益最大

收益：
- 8k prompt：单 forward 200ms → 4 stage pipeline 80ms（~2.5x 加速）
- 长 prompt（> 16k）：~4x 加速

实施风险：高；需要 fused attention kernel 重写；GDN pipeline 复杂

---

### 3.3 Disaggregated Prefill（用户策略三·分离式 Prefill / DistServe）

原理：prefill 与 decode 部署到不同节点，独立扩缩容

FreeToken 适配性：
- FreeToken 是单机引擎，不适合分布式
- 但可借鉴"分离"思想：在单 GPU 上用独立 CUDA stream 隔离 prefill 与 decode（轻 DistServe）
- v2 文档已设计模块 C（异步 prefill stream）；v3 强化：与 KV L1 tier (1.2) 结合，prefill stream 既做 KV 加载也做 KV 计算

收益：长 prompt（> 32k）TTFT ↓ 50%

实施风险：高；v2 已划入远期

---

## 4. 架构总览（v3 全图）

```
                        ┌────────────────────────────────────┐
                        │       请求进入 Scheduler            │
                        │   input_ids + tenant_id + sess_id  │
                        └─────────────┬──────────────────────┘
                                      │
                         ┌────────────▼─────────────┐
                         │ L1/L2 Prefix Cache Match │ ←──┐
                         │  (策略 1.1)              │    │
                         └────────────┬─────────────┘    │
                                      │ 命中?             │
                         ┌────────────▼─────────────┐    │
                         │ L1 (DRAM) / L0 (HBM)     │    │
                         │ async H2D DMA            │    │
                         │ (策略 1.2 + 2.4)         │    │
                         └────────────┬─────────────┘    │
                                      │                 │
                                      ▼                 │
                         ┌────────────────────────────┐    │
                         │ Spec Prefill (策略 2.3)    │    │
                         │  MTP head 评 token 重要度  │    │
                         │  → 选 Top-k% token 算 KV   │    │
                         └────────────┬───────────────┘    │
                                      │                 │
                                      ▼                 │
                         ┌────────────────────────────┐    │
                         │ Chunked + Mixed Batch      │    │
                         │ (策略 3.1 + v2 模块 A/B)   │    │
                         │  prefill chunk + decode    │    │
                         └────────────┬───────────────┘    │
                                      │                 │
                                      ▼                 │
                         ┌────────────────────────────┐    │
                         │ (可选) Sparse Attention    │    │
                         │ (策略 2.1 / TriangleMix)   │    │
                         │  深层三角形掩码            │    │
                         └────────────┬───────────────┘    │
                                      │                 │
                                      ▼                 │
                         ┌────────────────────────────┐    │
                         │ Context Pipeline           │    │
                         │ (策略 3.2)                 │    │
                         │  单 GPU 内 stage 并行      │    │
                         └────────────┬───────────────┘    │
                                      │                 │
                                      ▼                 │
                         ┌────────────────────────────┐    │
                         │ Sampling 首 Token           │    │
                         └────────────────────────────┘    │
                                                          │
                        L1 命中统计 → LRU 淘汰 ←────────────┘
```

---

## 5. 收益与优先级排序

| 阶段 | 模块 | TTFT 收益 | 实现成本 | 风险 | FreeToken 协同 |
|---|---|---|---|---|---|
| P0（立即） | 1.1 L1/L2 prefix cache | 系统提示场景 ~0 TTFT | 200 LOC | 低 | 与 KV q8_0 完美协同 |
| P1（1 周） | 2.4 计算+I/O 并行（Cake-style） | TTFT ↓ 30-50% | 500-800 LOC | 中 | 与 v2 模块 A/B 协同 |
| P2（2 周） | 2.3 Spec Prefill（MTP head） | TTFT(> 8k) ↓ 2-4x | 800-1000 LOC | 中 | MTP head 已存在 = 免费启动 |
| P3（3 周） | 1.2 KV tier (HBM/DRAM/NVMe) | L1 hit 时 TTFT ↓ 70% | 500 LOC | 中 | 1.1 升级版 |
| P4（4 周） | 2.1 TriangleMix 稀疏 | TTFT ↓ 12-32% | 300 LOC + 精度验证 | 中-高 | 仅 full-attn 层受益 |
| P5（远期） | 3.2 Context Pipeline | TTFT(> 8k) ↓ 2.5x | 高 | 高 | GDN 复杂 |
| P6（远期） | 2.2 Stem 动态稀疏 | TTFT ↓ 数倍 | 极高 | 高 | 需训练 |

推荐路线：P0（1-2 天）+ P1（1 周）+ P2（2 周） = 3 周内 TTFT P99 从 1s → ≤ 100ms，且 0 回归。

---

## 6. 与已有特性的深度协同（v3 vs v2 vs 现状）

| 特性 | v3 协同 |
|---|---|
| MTP head | Spec Prefill (P2) 的天然 draft model——其他引擎需要单独训练 |
| KV q8_0 量化 | 让 L1 cache (DRAM) 容量翻倍 → 命中率更高；让 P1 的 chunk 大小可放大 |
| CPU MoE offload | 与 Cake-style (P1) 同思路已存在 (moe_prefill_overlap)——v3 把模式推广到 I/O |
| hybrid GDN | Pipeline (P5) 最大受益于 GDN scan；TriangleMix (P4) 仅作用于 full-attn 层 |
| CUDA graph 禁用 | 不影响 v3——v3 是 forward 内部优化，与图捕获无关 |
| ChunkedReq | v3 不破 chunked prefill，反而让 chunk 内的 spec prefill 更高效 |

---

## 7. 度量与验证（v3 bench 脚本设计）

Bench 文件：benchmarks/cpu_moe_microbench/t_prefill_v3_bench.py

测试矩阵：
1. L1 hit 测试：相同 system prompt 100 次，测 TTFT P50/P90/P99（应该接近 0）
2. L2 hit 测试：多轮对话（每轮 prefix 增长），测第二轮起 TTFT
3. Cake-style I/O 并行：L1 部分命中（50%），测 forward time vs 现状
4. Spec Prefill：8k/16k prompt + top-k=50%/70%，测 TTFT 与精度
5. 组合负载：1 prefill + 10 decode 同时跑，测 decode jitter
6. 回归：MTP 启用 + KV q8_0 + ChunkedReq 同时启用，确认无功能退化

目标数字：
- TTFT(8k, 系统提示命中) ≤ 50ms
- TTFT(8k, 全新 prompt) ≤ 300ms
- TTFT(16k) ≤ 500ms
- decode jitter P99-P50 ≤ 2x
- 0 精度退化（perplexity 增量 ≤ 0.5%）

---

## 8. 风险登记（v3 specific）

| 风险 | 触发 | 缓解 |
|---|---|---|
| Spec Prefill 精度退化 | Top-k 太激进 → 丢重要 token | 保守起步（k=80%），perplexity 验证 |
| L1/L2 cache 一致性 | 用户改 system prompt | 失效机制：cache key = hash(system_prompt) |
| Cake-style 同步错乱 | DMA event wait 顺序 | 复用 engine.stream.wait_stream(prefill_stream) 模式 |
| Sparse attention 训练兼容性 | Qwen3.6 不支持三角形 | 先做 precision benchmark（200 LOC），不行就回退 |
| Pipeline 死锁 | GDN state 跨 stage 依赖 | 现有 _restore_linear_states 模式可推广 |

---

## 9. 不动代码实施的关键设计决策（备忘）

实施 P0+P1+P2 时，需要在以下点做 design-time 决策：
1. L1 cache 大小：默认 2 GB（RAM），可配置（--prefix-cache-ram-gb）
2. L1 命中判定：token-level MD5（前 100 token 的 hash）；不必做语义匹配
3. Spec Prefill 的 Top-k：默认 70%（保守）；可配置（--spec-prefill-ratio）
4. Cake-style 流同步点：forward_batch 头部触发 DMA，attention forward 前 wait（保持简单）
5. L1 eviction：LRU + 大小硬限；不实现 ARC/LFU（YAGNI）

---

## 10. 结论（v3 vs v2）

v2 解决"批结构"（mixed batch 让 decode 不被 prefill 阻塞）
v3 解决"单条 prefill 内的计算/I/O 优化"（少算 + 算 + I/O 并行）

两者正交，可同时实施：
- v2 模块 A/B 让 "8k prefill 在 800ms 跑完" → "8k prefill 在 400ms 跑完"
- v3 模块 P0/P1/P2 让 "8k prefill 在 400ms 跑完" → "8k prefill 在 80ms 跑完"

核心 insight：MTP head 是 FreeToken 的差异化资产——其他引擎需要单独训练 draft model 做 SpecPrefill，FreeToken 已有；这是 v3 战略优势的支点。

---

## 附录 A：与用户原始 7 条策略的映射

| 用户策略 | v3 模块 | 关键收益 |
|---|---|---|
| 1.1 Prefix Caching | P0 (1.1) | L1 命中时 TTFT ≈ 0 |
| 1.2 分级 KV Cache | P3 (1.2) | KV 容量从 8GB → TB |
| 1.3 Cache 预热 | P0 集成 (1.3) | 冷启动消除 |
| 2.1 静态稀疏 (TriangleMix) | P4 (2.1) | 深层 O(N²)→O(N) |
| 2.2 动态稀疏 (Stem) | P6 (2.2) | 25% 算力 |
| 2.3 投机 Prefill (SpecPrefill) | P2 (2.3) | 长 prompt ↓ 数倍 |
| 2.4 计算+I/O 并行 (Cake) | P1 (2.4) | 2.6x TTFT ↓ |
| 3.1 Chunked Prefills | P0 集成（v2 模块 A） | 与混合批协同 |
| 3.2 Context Parallelism | P5 (3.2) | 8k+ ↓ 2.5x |
| 3.3 Disaggregated Prefill | 远期 | 长 prompt ↓ 50% |

v3 唯一新增的 insight（相对用户原始策略）：MTP head 是 FreeToken 的 SpecPrefill 启动成本 = 0——这是其他引擎没有的差异化资产，必须利用。
