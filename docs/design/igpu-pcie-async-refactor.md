# iGPU PCIe Async Refactor — Design + Execution Plan

> **状态**：设计阶段
> **作者**：FreeToken 项目
> **日期**：2026-09-03
> **目标**：把 `--moe-backend=igpu` 路径从 7 t/s 提升到 25~35 t/s

---

## 一、问题陈述

### 1.1 当前性能

| 后端 | throughput | 备注 |
|---|---|---|
| `--moe-backend=offload` (CPU) | ~6 t/s | kernel 算力瓶颈 |
| `--moe-backend=igpu` (iGPU/HIP) | ~7 t/s | PCIe IPC 瓶颈 |
| 理论上限 (PCIe 16 GB/s) | ~43 t/s | 17 GB 权重 1.28 MB activations |
| 目标 | 50 t/s | 留 20% 余量 |

**核心矛盾**：iGPU 解锁了死锁，但解码仍卡在 7 t/s。瓶颈从 CPU kernel 算力（80 ms）变成 PCIe 同步开销（140 ms）。

### 1.2 根本原因

`igpu_shared_executor.decode()` 当前实现（P0 Form-2 设计，但只追求正确性）：

```
per MoE layer per token (40 层 × 6 sync 调用 = 240 次/步):
  1. hidden_states.cpu()                  -- D2H, 全 GPU sync
  2. topk_ids.cpu()                       -- D2H
  3. topk_weights.cpu()                   -- D2H
  4. stream.synchronize()                  -- 强制 GPU 等所有算完
  5. hipMemcpy H2D hidden staging          -- PCIe round-trip
  6. hipMemcpy H2D ids staging             -- PCIe round-trip
  7. hipMemcpy H2D weights staging         -- PCIe round-trip
  8. igpu_moe_decode_dev()                 -- HIP kernel + internal sync
  9. hipMemcpy D2H output                  -- PCIe round-trip
 10. out.copy_(out_host)                   -- H2D 回 GPU
```

**240 次同步，每次 ~50-100 µs 固定开销 → 12-24 ms 仅 IPC 同步**
**HIP kernel 计算时间 ~1 ms（被同步开销掩盖）**

### 1.3 物理极限测算

```
总 PCIe 数据 (per decode step):
  hidden 上下文:  1 token × 2048 × 4B = 8 KB (D2H)
  ids:           8 × 4B = 32 B
  weights:       8 × 4B = 32 B
  output:        1 × 2048 × 4B = 8 KB (H2D)
  per layer:    ~16 KB
  × 40 层:      ~640 KB

@ 16 GB/s PCIe 3.0: 640 KB / 16 GB/s = 40 µs (理论极限)
@ 实际 + 同步开销:  ~10-20 ms (现实极限)
```

---

## 二、目标设计

### 2.1 设计原则

1. **批量化 IPC**：所有 40 层共享同一组 staging buffer，1 次 D2H 入口 + 1 次 D2H 出口
2. **HIP Stream Async**：用 `hipMemcpyAsync` + `hipStreamSynchronize` 替换每层同步
3. **CPU pipeline**：HIP kernel 排队后 CPU 立刻返回，让 GPU attention / GDN 重叠 HIP 计算
4. **可观测性**：每步埋点，打印每阶段耗时，方便定位

### 2.2 架构变化

**Before** (current):
```
Python:
  for layer in range(40):
    D2H hidden  # sync
    D2H ids     # sync
    D2H wts     # sync
    sync()      # 全 GPU 等
    H2D staging # PCIe
    H2D staging # PCIe
    H2D staging # PCIe
    igpu_moe_decode_dev()  # HIP kernel + sync
    D2H out     # PCIe
    H2D out     # PCIe
```

**After** (target):
```
Python:
  # 入口: 一次 D2H 所有 hidden+ids+weights (async on cuda stream)
  hipMemcpyAsync(stage_in, hidden_gpu, ...);
  hipMemcpyAsync(stage_in, ids_gpu, ...);
  hipMemcpyAsync(stage_in, weights_gpu, ...);
  # GPU 工作: hip stream 排队 40 层 kernel
  for layer in range(40):
    igpu_moe_decode_dev_async(layer, stage_in[layer], ...);  # queue, 不 sync
  # 出口: 一次 D2H output + 一次 H2D 回 GPU (async)
  hipMemcpyAsync(out_gpu, stage_out, ...);
  hipStreamSynchronize(hip_stream);  # 等 HIP 全算完
  return out_gpu
```

### 2.3 关键 API 改动

| 旧 API | 新 API | 原因 |
|---|---|---|
| `hipMemcpy` (sync) | `hipMemcpyAsync` | 不阻塞 CPU |
| `hipMemset` (sync) | `hipMemsetAsync` | 不阻塞 CPU |
| `igpu_moe_decode_dev` (内部 `hipStreamSynchronize`) | `igpu_moe_decode_dev_async` (无 sync) | 让 HIP 自己排队 |
| `cudaStreamSynchronize` | 不需要（HIP stream sync 即可） | 跨 stream 同步由 CUDA event 处理 |
| `hidden_states.cpu()` (full sync) | pinned buffer + `cudaMemcpyAsync` | 跨 stream 不需要 sync |

### 2.4 数据流详细

```
staging buffer 布局 (host pinned, hipHostRegister'd):

┌──────────────────┐
│ in_hidden  [40 × 2048 × 4B] = 320 KB    # per-layer hidden state
│ in_ids     [40 × 8 × 4B]      = 1.3 KB
│ in_weights [40 × 8 × 4B]      = 1.3 KB
│ out        [40 × 2048 × 4B] = 320 KB   # per-layer output
└──────────────────┘

GPU side:
┌──────────────────┐
│ stage_in_gpu  (1 alloc)
│ stage_out_gpu (1 alloc)
└──────────────────┘

数据流:
  step 0: layer 0 forward → GPU hidden
  step 1: cudaMemcpyAsync(stage_in, layer_0_out)  # 1 次
  step 2: hipMemcpyAsync(hip_staging, stage_in)    # 跨 GPU, 1 次 PCIe
  step 3: hip stream queue 40 kernels
  step 4: hipStreamSynchronize()
  step 5: hipMemcpyAsync(stage_out, hip_staging)   # PCIe, 1 次
  step 6: cudaMemcpyAsync(layer_1_in, stage_out)  # 1 次
```

注意：层间依赖（layer N+1 需要 layer N 的输出）让 40 层不能完全并行。但可以分两批：
- batch A: layer 0~19 (20 层 kernel queue)
- batch B: layer 20~39 (20 层 kernel queue)
- 每批内部 HIP stream async, 批间同步

这能让 CPU 在 HIP 算 layer 1-19 时并行准备 layer 20-39 的输入。

---

## 三、执行计划（按阶段，每阶段独立验证）

### Phase 0：基线测量（30 分钟）

**目标**：用真实模型跑当前 `--moe-backend=igpu`，测量 7 t/s 时各阶段耗时分布

**做什么**：
- 在 `igpu_shared_executor.decode()` 加详细计时（cudaEvent + perf_counter）
- 跑 1 个 decode step，记录每层 D2H / H2D / kernel 时间
- 输出 `_igpu_baseline.json`

**验证标准**：拿到数据，确认 PCIe IPC 是主要瓶颈（应该 >50% 总时间）

**风险**：无，纯加日志

### Phase 1：HIP Stream Async 基础（半天）

**目标**：把 `igpu_moe_decode_dev` 改成 async 版（不内部 sync）

**做什么**：
- `hip_moe_dll.hip` 增加 `igpu_moe_decode_dev_async`（kernel 后不调用 `hipStreamSynchronize`）
- 重编译 DLL
- Python 端切换到 async 版调用，外部统一 sync

**验证标准**：
- `_igputest.py` 仍然输出正确数值（vs reference，rel err < 1e-3）
- 当前 7 t/s 不应退化（<10% 偏差可接受）

**风险**：低。功能等价，只改同步语义

### Phase 2：单层 IPC 批量化（半天）

**目标**：每层只做 1 次 staging H2D（合并 hidden+ids+weights 为一个连续 buffer）

**做什么**：
- 预分配 per-layer staging buffer (hidden + ids + weights + output 连续内存)
- Python 端合并 D2H：1 次 `cudaMemcpyAsync` 拷所有 hidden，1 次所有 ids，1 次所有 weights
- `hipMemcpyAsync` 同样合并

**验证标准**：
- `_igputest.py` 输出正确
- `Phase 0` 的计时 log 显示：D2H 次数从 40 降到 3
- throughput 应有 1.5~2x 提升（同步次数减半）

**风险**：中。staging buffer layout 要跟 DLL 严格匹配

### Phase 3：HIP Stream 串行 40 层（1 天）

**目标**：40 层 kernel 在 HIP stream 上排队，只在末尾 sync 一次

**做什么**：
- 改 `decode()`：所有 40 层用同一个 HIP stream 串行 queue
- 入口：1 次 D2H 整个 step
- 40 层用同一个 staging buffer，HIP stream 自动 serial
- 出口：1 次 H2D + 1 次 sync

**验证标准**：
- `_igputest.py` 输出正确
- 引擎端 e2e 测试 throughput ≥15 t/s (vs 7 baseline)
- log 显示 sync 次数从 240 降到 ~3

**风险**：高。需要 DLL 修改支持层间 serial 模式

### Phase 4：CUDA Graph 包装（可选，1 天）

**目标**：把整个 decode step CUDA-graph-capturable

**做什么**：
- 用 `torch.cuda.graph` capture `decode()` 全过程
- 重复执行时 `graph.replay()`

**验证标准**：
- Python overhead < 2 ms / step
- throughput ≥25 t/s

**风险**：高。需要 replay-safe 代码（无 Python 副作用）

### Phase 5（如果以上都成功）：MTP + Batching

MTP K=2+ + 连续 batching 叠加，看能否到 40~50 t/s

---

## 四、风险与备选

### 4.1 技术风险

| 风险 | 概率 | 影响 | 备选 |
|---|---|---|---|
| HIP async 内部有隐式 sync | 中 | Phase 1 退化为 7 t/s | 用 cuda event + `cudaStreamWaitEvent` 跨 stream 同步 |
| PCIe staging buffer 大小不可控 | 低 | 内存不足 | 缩小到 32 层 × batch=2 |
| CUDA Graph capture 期间 HIP 调用失败 | 高 | Phase 4 退回到 Phase 3 | Phase 4 跳过，直接验收 Phase 3 |
| GPU 调度顺序导致 deadlock | 极低 | 服务挂死 | 已有 deadlock 检测，进程退出可恢复 |

### 4.2 数据正确性

每阶段必须满足：
- `_igputest.py` 输出与 baseline rel err < 1e-3
- e2e chat 输出文本合理（不重复，不退化）
- MTP accept rate 不退化（应保持 0.30 tok/verify）

### 4.3 兜底方案

如果 Phase 3 拿不到 15 t/s：
- 退回 Phase 2（仍然比 baseline 快 2x）
- 重新评估：是否还有别的优化空间
- 考虑切换到 **CPU offload 优化** 路径（不在本文档范围）

---

## 五、验证脚本清单

| 文件 | 用途 |
|---|---|
| `_igputest.py` | 单 layer weight × activation 数值正确性 |
| `_igpu_baseline.json` | Phase 0 输出，记录 7 t/s 时的耗时分布 |
| `_bench_decode.py` | e2e 单 req decode 跑 N step，记录平均 throughput |
| `_bench_mtp.py` | MTP accept rate 不退化验证 |

---

## 六、预期收益曲线

| 阶段 | 累计 sync 数/步 | 预期 throughput | 累计收益 |
|---|---|---|---|
| Baseline | 240 | 7 t/s | 1.0x |
| Phase 0 | 240 | 7 t/s | 1.0x (measure) |
| Phase 1 | 240 | 7 t/s | 1.0x (refactor only) |
| Phase 2 | 80 | 12~15 t/s | 2.0x |
| Phase 3 | 3 | 25~30 t/s | 4.0x |
| Phase 4 | 0 | 30~35 t/s | 5.0x |
| Phase 5 (MTP) | - | 40~50 t/s | 7.0x |

---

## 七、立即执行

**Phase 0 先跑**，确认基线数据。这是整个重构的"锚点"——没有它所有后续优化都是猜测。

Phase 0 完成后输出：
- `_igpu_baseline.json` (结构化耗时分布)
- 文字总结（瓶颈在 D2H / H2D / HIP kernel / Python overhead 哪一项）


---

## 八、Phase 0 测量结果（基线数据）

**测量方法**：独立 Python 脚本 `_igpu_phase0.py`，复刻 `igpu_shared_executor.decode()` 的每一步调用（不调真实模型），测时。bs=1, steps=2。

### 8.1 实测耗时分布（avg over 2 steps, ms）

| 步骤 | 旧预期 | 实测 avg | 实测占比 | 备注 |
|---|---|---|---|---|
| d2h_hidden (1×) | 同步开销主导 | 1.19 ms | 2% | 第一次 2.3ms（page fault 冷启） |
| d2h_ids/weights (各1×) | - | 0.07 ms | 0.1% | |
| stream.synchronize() | - | 0.06 ms | 0.1% | |
| **h2d_staging_hidden (1×)** | **同步主导** | **9.10 ms** | **15%** | ⚠️ **第一次冷启 18ms**（PCIe + 首次 H2D 缺页）；后续 0.03ms |
| h2d_staging_ids/weights (各1×) | - | 0.03 ms | 0.1% | |
| **hip_kernel (40层)** | **被掩盖** | **49.17 ms** | **81%** | ⚠️ **真正瓶颈**：每层 1.23ms 780M kernel |
| d2h_out | - | 0.21 ms | 0.3% | |
| h2d_out | - | 0.15 ms | 0.3% | |
| **总步长（仅 decode 数据流）** | - | **60.65 ms** | 100% | **≈ 16.5 tok/s（独立测量）** |

**对比 e2e**：实际 `--moe-backend=igpu` e2e 是 7 tok/s。差异在 attention + GDN + sampling + KV cache + Python overhead ≈ 80 ms（这部分 dGPU 算）。

### 8.2 关键发现

**PCIe IPC 不是主要瓶颈**（旧假设错）：
- 1 次冷启 H2D 隐藏 18ms（缺页），后续单次只 0.03ms
- 240 次 sync 总耗时 ~6ms（不是预估的 12-24ms）
- 总 IPC 成本 < 10ms，占 decode 总时间 < 15%

**真正瓶颈 = HIP kernel 计算**：
- 40 层 × 1.23 ms/layer = 49 ms
- 每层 8 专家 × 1 token 算 = 8 个小 kernel 串行
- 780M 算力（~1 TFLOPS）下 8 专家 matmul 确实要 ~1ms

### 8.3 修正后的优化方向

| 旧优化方向 | 状态 | 新方向 |
|---|---|---|
| Phase 2 批量化 IPC | ❌ 不需要（已几乎为零） | 改为优化 kernel 本身 |
| Phase 3 HIP Stream 串行 40 层 | ⚠️ 边际收益 | 改为：HIP kernel 内并行化 |
| Phase 4 CUDA Graph 包装 | 仍值得做（省 Python overhead ~10ms） | 保留 |

### 8.4 三个新优化目标

1. **kernel 加速（49ms → 20-30ms）**：
   - 整层 8 专家融合成一个 kernel（一次 launch 处理 8 个 expert）
   - gate_up + act + down 融合（少 2 次 launch）
   - 用 780M 上 FP16/BF16 中间格式代替 FP32

2. **dGPU 与 iGPU 重叠（节省 49ms）**：
   - 当前 decode 串行：dGPU attention → dGPU GDN → iGPU MoE → dGPU attention → ...
   - 如果 iGPU 算 layer 0 时 dGPU 同时算 layer 1 的 attention，可隐藏 49ms 中的大部分
   - 实现：双 stream，dGPU 走 attention+GDN，iGPU 走 MoE，CUDA event 同步

3. **MTP K=2+**：
   - 当前 K=1 accept 30%, K=3 全 reject
   - 修 K=2+ 让一次 verify 接受 2-3 个 draft → 实际 1.5-2x throughput
   - 这部分独立于 iGPU path

### 8.5 决策

**继续做 Phase 1（HIP Stream Async）作为基础设施**，因为：
- 重叠（方向 2）必须基于 async 流水线
- CUDA Graph（Phase 4）也必须基于 async 接口

**Phase 2 (IPC 批量化) 暂时不做**——边际收益 < 1ms。

**Phase 3 (HIP Stream 串行) 改为 Phase 3'：HIP + CUDA 双 stream 重叠**：
- 这是 49ms kernel 隐藏到 attention 后面的关键
- 预期：e2e 从 7 t/s → 20-25 t/s（隐藏 ~40ms iGPU 计算）

### 8.6 Phase 0 验证脚本可重用

脚本 `_igpu_phase0.py` + 输出 `_igpu_baseline.json` 可作为 Phase 1/2/3 的回归基准：
- 改动后跑同一脚本，看 `h2d_staging_hidden` 减少（说明 IPC 优化生效）
- 看 `hip_kernel` 时间（说明 kernel 优化生效）
- 整体 `step_total` 应该减少

### 8.7 修正后的预期收益

| 阶段 | 主要优化 | 预期 step 总耗时 | e2e 预期 |
|---|---|---|---|
| Baseline (Phase 0) | - | 60 ms (decode) + 80 ms (其他) = 140 ms | 7 t/s |
| Phase 1 | HIP async + 重叠 | 60 ms 但被重叠 | 12-15 t/s |
| Phase 2 (kernel 优化) | 8 专家融合 + FP16 | 30 ms | 18-22 t/s |
| Phase 3 (dGPU+iGPU 重叠) | 双 stream pipeline | hidden 30ms | 25-30 t/s |
| Phase 4 (CUDA Graph) | 省 Python overhead | hidden 10ms | 30-35 t/s |

