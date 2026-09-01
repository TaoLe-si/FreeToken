# 稠密模型主机内存驻留（Dense-in-RAM）设计方案

> 目标：让显存放不下的稠密模型（Qwen3.6-27B / gpt-oss-120b / Muse-Glimmer-30B …）
> 在消费级/单卡机器上跑起来，并把 GPU 与主机内存之间的交互速度推到机器上限。
> 状态：**设计稿**（未实现）。本方案完全复用仓库已有的 MoE offload 机制
> （pinned 主机银行、带宽校准、CPU-GPU 标志握手、_cpu_moe 内核），不新造轮子。

---

## 0. 先说清楚物理极限（为什么不能照搬 MoE offload）

稠密 decode 每生成一个 token 必须**读一遍全部权重 W 字节**（没有专家稀疏）：

    tok/s ≈ 有效带宽 × 批量 M / W

| 通路 | 典型有效带宽 | 27B NVFP4 (≈15GB) bs=1 | 27B bs=8 | 27B bs=32 |
|---|---|---|---|---|
| GPU 读 VRAM（现有 fused 路径） | ~1 TB/s | ~65 | — | — |
| GPU 经 PCIe4 x16 读 DRAM | 25-32 GB/s | **~2** | ~14 | ~55 |
| GPU 经 PCIe5 x16 读 DRAM | 50-64 GB/s | ~4 | ~28 | ~110 |
| **CPU 直读 DRAM**（DDR5-5600 双通道） | 76-90 GB/s | ~5 | **~40** | **~160** |
| CPU 直读 DRAM（服务器 8 通道） | 300+ GB/s | ~20 | ~160 | ~600 |

结论：
1. **bs=1 时任何内存驻留方案都在个位数 tok/s** —— 这是物理定律，不是优化能解决的。
   所以方案必须面向**批量解码（M≥8）**设计，单用户慢速模式只作兜底。
2. **CPU 直读 DRAM 的带宽 > GPU 经 PCIe 读 DRAM**（消费机 2-3×，服务器差距更大）。
   因此"大权重部分放 CPU 算、小权重部分放 GPU 算"是比"全量流式给 GPU"更优的切分。

---

## 1. 总体架构：分层混合 + 三层流水线

### 1.1 权重布局（主机内存）

- FTW 打包的量化银行（NVFP4 / MXFP4 / Q4_0，复用 `weight.py` 的
  `alloc_pinned_tensor / copy_to_pinned_tensor`），**pinned + mlock**，按 NUMA
  节点交错（interleave），2MB/1GB 大页。
- 每层拆三块：`qkv_proj`、`o_proj`（GPU 算，PCIe 流式）、
  `gate/up/down FFN`（CPU 算，DRAM 直读）。
- 权重占比：attention 投影 ~1/3，FFN ~2/3 → 把 2/3 的字节压在带宽更高的 DRAM 侧。

### 1.2 计算分工

- **GPU**：attention（含 KV cache，必须留 GPU，否则 KV 读取带宽会杀死一切）、
  norm、residual、sampling、paged KV、CUDA graph decode。
- **CPU（扩展 `_cpu_moe` 为稠密 FFN 引擎）**：逐层 FFN gate/up/down GEMV，
  从 DRAM 顺序读权重（AVX-512 BF16/VNNI、W4A8/W4A16、行分块线程池、软件预取全复用）。

### 1.3 三层流水线（每 decode 步，滚动窗口 3 层）

```
   PCIe copy engine:   ... | QKV/O(L)   | QKV/O(L+1) | QKV/O(L+2) | ...
   GPU SM:                   | attn(L-1) | attn(L)    | attn(L+1)  | ...
   CPU FFN pool:             | ffn(L-1)  | ffn(L)     | ffn(L+1)   | ...
                              _____________________________/
                               每层只交换两个小缓冲（x、g，各 ~H×M×2B）
```

- copy 引擎（DMA）不占 SM，天然与 attention 并行；CPU 池与两者都并行 → 三层真正重叠。
- 每层只有两个小同步点：attention 输出 x(L) 拷回 CPU（D2H，~KB 级），
  FFN 结果 g(L) 拷回 GPU（H2D，~KB 级）。用小缓冲双份 + 事件/标志流水化，不做每层同步。

### 1.4 与 MoE offload 的区别（复用点 vs 新点）

| 机制 | MoE offload（已有） | 稠密本方案 |
|---|---|---|
| 主机银行 | 每层 [E, …] 专家银行，LRU 取子集 | 每层固定 3 块投影银行，**全量流式**，无 LRU |
| GPU 缓存 | 专家 slot LRU | 每层 QKV/O 双缓冲（乒乓） |
| 搬运触发 | Triton evict/fetch kernel | 纯流水线（事件/标志链） |
| CPU 计算 | miss 专家 GEMV | 稠密 FFN GEMV（同内核族） |
| 校准 | `ft bench bw` → hybrid 阈值 | 同机制 → 决定"流式" vs "驻留"分层切点 |

---

## 2. 最大化 GPU ↔ 内存交互速度（本方案核心）

### 2.1 PCIe 侧（GPU 读权重）

1. **pinned + 异步**：银行必须是 pinned（page-locked），搬运一律
   `cudaMemcpyAsync`。pageable 会触发同步拷贝/页错误（offload_cache.py:351 已有
   "pageable -> synchronous copy" 的降级逻辑，这里不允许降级）。
2. **按层合并大拷贝**：把 QKV 三块拼成一个连续 host 缓冲、O 一块，每层 2 次大拷贝
   替代 8-12 次小拷贝 —— 拷贝吞吐对块大小敏感（DMA 启动/碎片开销）。
3. **专用高优先级 copy stream + 双缓冲**：`copy(L+1) ∥ compute(L)`；
   缓冲 A/B 乒乓，杜绝搬运等待计算、计算等待搬运。
4. **零 host 同步握手**：复用 `cpu_moe_ext.cpp` 的
   `cuStreamWriteValue64 / cuStreamWaitValue64` + mapped-pinned 标志（580-673 行）。
   GPU attention kernel 直接在 stream 上 spin-wait 拷贝完成标志 —— 不需要
   `cudaEventSynchronize`/host 轮询，延迟 ~µs 级。
5. **避免 bounce**：x/g 激活缓冲全部 pinned；KV 永不离开 GPU；不做中间页交换。
6. **校准驱动切分**：`ft bench bw` 实测本机 PCIe 有效带宽；若某层 QKV/O 总量
   小于"带宽×步间预算"，就整层驻留 GPU，否则流式 —— 与 hybrid 阈值思想一致。
7. **Windows/WDDM 提示**：WDDM 下 `cudaMemcpyAsync` 有额外提交开销，Linux 优先
   （仓库本来就是 Linux 优先）；Windows 上可考虑减少拷贝次数、加大批量摊薄。

### 2.2 DRAM 侧（CPU 读权重）

1. **NUMA 交错**：银行跨 socket 交错分配（或按 CPU 亲和就近），FFN 线程绑物理核
   （executor 已有 physical-core 亲和），避免远端内存跳变。
2. **大页**：2MB/1GB THP 或显式 hugetlb —— 权重是 GB 级顺序流，TLB miss 是实打实
   的损耗；顺序流下大页收益最明显。
3. **非临时加载变体**：为 bf16/nvfp4/q4_0 内核加 `movntdqa`（streaming load）
   版本，权重流不污染 L2/L3（激活和路由表需要缓存，权重不需要）。
4. **软件预取距离**：`FREETOKEN_CPU_MOE_PF_AHEAD` 已落地（微基准显示 128B 与
   512B 在 Zen4 上差 40%），在稠密 FFN 引擎上继续可调 + 启动自动调优。
5. **通道数就是天花板**：消费机双通道 DDR5 ≈ 76-90 GB/s，服务器 8 通道 300+ GB/s。
   性能模型必须按真实通道数填，别拿服务器数字骗自己。

### 2.3 重叠与调度

- 3-way overlap（见 1.3）：CPU FFN(L) ∥ GPU attn(L+1) ∥ PCIe copy(L+2)。
- 批量感知：连续批处理强制 **M≥8 才走 dense-in-RAM**（M<8 提示慢速模式预期）；
  调度器可把多个请求聚成一步 decode（仓库已有 continuous batching）。
- CUDA graph：attention+标志等待节点整体进图，每步只换输入/标志地址，不加 host 开销。

---

## 3. 显存预算（24GB 卡跑 Qwen3.6-27B NVFP4 为例）

| 项目 | 大小（估） | 说明 |
|---|---|---|
| 权重 | 0 | 全在 RAM（这是本方案的意义） |
| QKV/O 双缓冲 | ~2×80MB | 每层两块，乒乓；只缓存当前窗口 2-3 层 |
| KV cache | ~2-4GB | 4 KV heads×128×2B，32-64K ctx；**剩余显存全给它** |
| activation/paged KV 元数据 | <0.5GB | 常量 |
| CUDA graph 池 | ~0.3GB | 常量 |
| **合计** | **~3-5GB** | 其余 ~19GB 全部可吃上下文 |

对比 fused 路径（27B NVFP4 全驻留 ≈15GB + KV 2-4GB ≈ 17-19GB，32K ctx 就贴顶）：
dense-in-RAM 把**全部权重换成上下文长度**，这是方案对长上下文 agent 场景的真正卖点。

---

## 4. 性能模型（估算，标注：未实测）

假设 27B NVFP4（≈15GB，其中 FFN ≈10GB / attention ≈5GB）：

| 场景 | 通路 | 估算 tok/s |
|---|---|---|
| bs=8，DDR5-5600 双通道 | CPU FFN 10GB/8=1.25GB/token @76GB/s | **~40**（CPU 侧主导） |
| bs=8，PCIe4 GPU 全流式 | 15GB/8=1.9GB/token @25GB/s | ~13（不如 CPU 侧） |
| bs=32，双通道 | 0.47GB/token @76GB/s | ~160 |
| bs=32，服务器 8 通道 | 0.47GB/token @300GB/s | ~600 |
| gpt-oss-120b NVFP4（≈66GB）bs=32 服务器 | 2.06GB/token @300GB/s | ~145 |

* 数值受 kernel 效率、量化 overhead、NUMA、真实带宽打折影响；上线前用 Phase 0
  实测带宽填表。
* CPU VNNI 算力余量：4-bit 权重 W4A8/W4A16 下，FFN 算力需求（≈2×字节数×MAC 折算）
  远低于 DRAM 带宽供给，**算力不是瓶颈，带宽是**——与 MoE CPU 内核结论一致。

---

## 5. 实施路线（每阶段独立可验证，不阻塞）

- **Phase 0 — 校准（无代码）**：`ft bench bw` + 现有微基准补一项 DRAM 顺序读
  带宽测试（本机 DDR5 双通道），拿到真实数字回填 §4 模型。
- **Phase 1 — CPU 稠密 FFN 引擎**：扩展 `_cpu_moe` 为稠密层任务（每层一个
  "gate/up/down 银行 + g_row 写回"任务）；GPU 侧 attention 与 CPU FFN 用现有
  标志握手串起来；先不做 PCIe 流式（QKV/O 暂时驻留小权重）。验证：单层数值对齐
  + 端到端 decode 吞吐。
- **Phase 2 — PCIe 流式 QKV/O**：pinned 银行 + 双缓冲 + copy stream + 事件链，
  支持 24GB 卡跑 27B。验证：copy 与 compute 重叠度（ncu/nsys 看 timeline）。
- **Phase 3 — 批量感知与驻留决策**：连续批处理强制 M≥8 走该路径；bandwidth-
  adaptive 分层驻留（PCIe 快 → 更多层给 GPU，慢 → 全给 CPU）。
- **Phase 4 — 内存通道榨干**：NUMA 交错、大页、movntdqa 内核、PF 自动调优、
  多 stream 并发搬运。

---

## 6. 风险与权衡

1. **bs=1 体验差**（~2-5 tok/s）：与 MoE offload 的本质差异——MoE 只取 active
   experts，稠密必须全量流。必须靠批量和 UI 预期管理兜底。
2. **CPU 侧成为新热点**：Phase 1 后 CPU FFN 引擎就是"CPU 运算性能最大化"的战场，
   与本仓库正在做的 CPU MoE 优化（SIMD/预取/线程池）完全同源，可互相复用。
3. **多路并发争抢 DRAM**：多请求并行时 DRAM 带宽是共享池，总 tok/s 守恒。
4. **格式约束**：第一版只支持 NVFP4/MXFP4/Q4_0；bf16 稠密字节翻倍、吞吐减半。
5. **WDDM 拷贝开销**：Windows 上流水线效率打折，Linux 优先。


---

## 7. 实测裁决：本机（Windows 笔记本）的架构选择

### 7.1 本机实测关键带宽（2026 实测）

| 通路 | 实测 | 备注 |
|---|---|---|
| DDR5-5600 双通道 DRAM 顺序读 | **51 GB/s**（16 线程饱和） | 4KB 页 TLB + 笔记本功耗墙，理论 89.6 的 57% |
| PCIe H2D / D2H（RTX 4070 Laptop） | **6.7 GB/s** | Windows WDDM 模式 + 笔记本链路，远低于 PCIe4 x8 理论 16 GB/s |
| CPU bf16 dot 峰值（8 线程） | 0.28 T MAC/s | 被"每 MAC 读 4B"缓存带宽限制，非 FMA 上限 |
| NVFP4 W4A8 VNNI 峰值（8 线程） | 0.09 T MAC/s | 每块浮点反量化开销主导，比 bf16 慢 3 倍（待优化） |

### 7.2 "GPU 单独算 + 权重内存流式"在本机的真实数字（27B NVFP4 ≈15GB）

    tok/s ≈ PCIe_6.7GB/s × M / 15GB

| M | tok/s | 判语 |
|---|---|---|
| 1（单用户） | **0.45** | 不可用（比 CPU 方案慢 10 倍） |
| 8 | 3.6 | 慢 |
| 32 | 14 | 需要 32 路并发才有意义 |

对比 CPU 直读 DRAM：bs=1 即 5-7 tok/s。**本机 PCIe 是 10 倍瓶颈，GPU-流式架构在此机不成立。**

### 7.3 KV 放置的铁律（用户想"剩余内存做上下文"）

- **GPU 算 attention 时，KV 必须留显存**：每 token 要读全部历史 KV，
  32K 上下文 = 4GB/token/序列；经 PCIe 6.7GB/s 只有 1.7 tok/s，直接杀死一切。
  显存 8.6GB（权重流式不占）→ KV ~8GB → 27B 约 **64K token 上下文**（已是天花板）。
- **CPU 算 attention 时，KV 可以放内存**：CPU 直读 DRAM 51GB/s 扛得住 KV 流；
  47.2GB 内存 − 15GB 权重 − 3GB 引擎 ≈ 29GB KV → 27B 约 **230K token 上下文**。
- **结论：要"内存剩余空间做上下文"，就必须 CPU 算（或至少 attention 在 CPU）——
  "GPU 单独算 + 内存 KV"在带宽上自相矛盾。**

### 7.4 本机最终架构裁决

| 架构 | 27B bs=1 | 27B 上下文 | 判定 |
|---|---|---|---|
| GPU 流式（PCIe 6.7GB/s） | 0.45 tok/s | ~64K（KV 显存） | ✗ 慢 10 倍 |
| **CPU 全算（DRAM 51GB/s）** | **5-7 tok/s** | **~230K（KV 内存）** | ✓ 本机唯一满足"剩余内存做上下文" |
| 12B NVFP4 fused（8GB 显存） | ~39 tok/s | ~50K（KV 显存） | ✓ 追求速度时选它 |

**换机器（Linux 台式机/服务器）时 GPU-流式才值得**：PCIe4 x16 ≈25GB/s → bs=8 即
13 tok/s；PCIe5 x16 ≈55GB/s → bs=8 即 29 tok/s、bs=32 达 110 tok/s（此时 CPU 算力
0.28T 反而是新瓶颈，需配合内核优化）。

### 7.5 若坚持 GPU 计算：可行前提清单
1. Linux 系统（避开 WDDM 拷贝开销，PCIe 通常 ×2-3）；
2. PCIe ≥ x16 gen4（实测 ≥25GB/s）；
3. 批量 M≥8（agent 多分支 / 多用户聚合）；
4. KV 显存驻留（≤64K ctx），不接受"内存 KV + GPU 算"；
5. 权重格式 NVFP4/MXFP4/Q4_0（字节减半 = 吞吐翻倍）。


---

## 8. GPU 辅助 + "压缩传输"的裁决与正解

### 8.1 "高度压缩权重"不可行（数据说话）

- NVFP4 权重已经是 **4bit（0.5B/参数）**，量化后的权重近似随机数据，熵接近极限：
  无损熵编码的压缩率只有 ~1.05-1.1x，**无实际收益**。
- 有损压缩权重 = 再量化，等价于换更低位宽格式（Q4_0 已是最低实用档），
  字节数不变。
- **结论：权重不能压。能压的是"传输量"——即"不传权重"。**

### 8.2 正解：权重驻留显存（能放多少放多少），PCIe 只传激活

RTX 4070 Laptop 8.6GB 显存的分配（27B NVFP4）：

| 显存占用 | 大小 | 作用 |
|---|---|---|
| attention QKV/O 权重 | ~5GB | **全驻留**，GPU 算 attention，KV 也留显存（~2-3GB） |
| FFN 权重（约 15 层） | ~3.5GB | 驻留部分层，GPU 直接算 |
| 激活/双缓冲 | <0.1GB | x/g 每层 KB 级 |

- **PCIe 传输量：15GB/token → 每层 KB 级激活（x、g）**——这就是"压缩传输"的极限答案：
  传输的不是权重而是激活，且激活可近无损压缩（bf16→fp8/int8，误差 <0.5%，量本就 KB 级）。
- 其余 ~49 层 FFN 权重仍由 CPU 从 DRAM 直读（51GB/s，零传输）。

### 8.3 吞吐裁决：单请求不变，多请求翻倍

- **层内串行（单请求）**：每层 = GPU attention(L) → CPU FFN(L)（或 GPU FFN 驻留层），
  慢段永远是 CPU FFN（5-7 tok/s）——**GPU 辅助不改变单请求 tok/s**。
- **请求级流水（M≥2 并发）**：CPU 处理请求 A 的 49 层 FFN 的同时，GPU 处理请求 B 的
  attention + 15 层 FFN —— CPU 和 GPU 各自吞吐**相加**：
  - CPU 侧：51GB/s ÷ 7.7GB/token ≈ **6.6 tok/s**
  - GPU 侧：显存 256GB/s ÷ 3.5GB(15层 FFN) ≈ **73 tok/s**（attention 权重驻留后几乎免费）
  - **合计 ≈ 80 tok/s（M≥2）**，且随驻留 FFN 层数增加而更高。
- 该数字受 CPU 内核优化影响：**先修 VNNI 反量化（0.09T→0.3-0.5T）**，CPU 侧可再翻倍。

### 8.4 实施分级（每级独立可验证）

1. **级 0（内核）**：优化 W4A8 VNNI 反量化 → CPU FFN 算力墙 7.8→15-30 tok/s
   （微基准 A/B 验证；MoE 与 dense 双赢）。
2. **级 1（GPU attention + CPU FFN）**：attention 权重驻留显存、KV 显存、CPU 算 FFN；
   每层只交换 KB 级 x/g（pinned + cudaMemcpyAsync 双缓冲，bf16→fp8 压缩）。
   验证：单请求 5-7 tok/s 不变，但上下文 ~64K、CPU 负载降 1/3。
3. **级 2（显存驻留 FFN 层 + 请求级流水）**：GPU 算驻留层，CPU/GPU 分请求并行；
   总吞吐 → 50-80 tok/s。验证：M≥2 时总吞吐相加。
4. **级 3（自动平衡）**：按实际 PCIe/DRAM/显存带宽自动决定每层归属
   （bandwidth-adaptive，复用 hybrid 阈值思想）。


---

## 9. 端到端速度预估（27B NVFP4 ≈15GB，本机实测带宽输入）

假设：DRAM 51GB/s、PCIe 6.7GB/s、显存 256GB/s（规格值）、CPU 算力 0.28T MAC/s
（当前内核）、KV ≈128KB/token/序列（4 KV heads；8 heads 则上下文减半）。
prefill 按 2×27e9 FLOP/token。

| 方案 | bs=1 解码 | bs=4-8 解码 | 上下文上限 | prefill 1K tokens | 判语 |
|---|---|---|---|---|---|
| **A 纯 CPU 全算**（KV 内存） | **2.5-3.5 tok/s** | 5-7 tok/s | **200K+** | ~60-90 s（CPU 算力不足） | 大上下文专用 |
| **B GPU attention + CPU FFN**（级 1） | ~5 tok/s | 7.8（算力墙） | ~24K（KV 显存 3GB） | 分钟级 | 单请求略快，上下文缩水 |
| **C 级 2 请求级流水**（GPU 驻留 15 层 FFN） | 单请求 ~6 | **总吞吐 50-55** | ~20K | GPU 部分快、CPU 部分慢 | M≥2 多路场景最优 |
| D GPU 全流式（PCIe 6.7） | 0.45 | 3.6-14 | ~64K | 30+ min/1K | ✗ 不可行 |
| **E 12B NVFP4 fused GPU** | **30-39 tok/s** | 30-39 | ~20K | ~1-2 s | 本机速度最优 |

**关键修正**：方案 A 的端到端流量 = FFN 10GB + attention 5GB + KV（32K ctx 时 4GB）
≈ 19GB/token → 51/19 ≈ **2.7 tok/s**（bs=1）。此前只按 FFN 10GB 算的 5 tok/s 是
上限而非端到端。

**读取/加载速度**：FTW 格式 15GB 从 NVMe 载入内存约 3-5s（一次性）；内存常驻后
加载成本为 0。prefill 是内存方案的共同短板（CPU 算力 0.28T 与 PCIe 6.7 都撑不起
GEMM），1K 提示词在 A/B 需分钟级，只有 E（12B fused）能秒级 prefill。

**结论**：本机三档选择——速度（E：30-39 tok/s，prefill 快）、大上下文（A：
2.5-3.5 tok/s，200K+）、多路吞吐（C：50-55 tok/s，M≥2）。B 是不上不下的过渡。


---

## 10. 256K 上下文 + KV 压缩（用户的硬约束）

### 10.1 为什么必须压缩（数据）

| KV 方案（27B，64 层） | 256K ctx 存储 | 每 token 读取流量 | CPU-DRAM 读取上限 |
|---|---|---|---|
| fp16 全量（4 heads） | 34GB（内存都不够） | 34GB | 1.5 tok/s |
| fp8 全量 | 17GB（勉强内存） | 17GB | 3.0 tok/s |
| Q4 全量 | 8.6GB（显存也不够） | 9GB | 5.9 tok/s |
| **DSA 稀疏（近端4K+全局512）× Q4** | **0.15GB** | **0.15GB** | **346 tok/s** |

**结论：量化（字节）只是辅助，稀疏（token 数）才是 256K 的生死线**——
attention 每 token 必须读全部 KV，全量 KV 在 256K 下任何介质都撑不住。

### 10.2 压缩后的 256K 端到端（27B NVFP4）

| 阶段 | 方案 | 速度 | 判语 |
|---|---|---|---|
| **decode bs=1** | CPU FFN（DRAM 51GB/s）+ GPU attention（稀疏 KV 显存） | **~3.4 tok/s** | 瓶颈仍是权重 15GB，KV 已不是问题 |
| **decode 多请求** | 级 2 混合（显存驻留 FFN 层） | 50-80 tok/s（M≥2） | GPU 侧吃掉大半 |
| **prefill 256K** | **GPU 按层流式**（每层权重 PCIe 传一遍，共 15GB=2.2s）+ GPU 算力 | **~13 min** | CPU 全算 = 7 小时 ✗；只有 GPU 可行 |
| 模型加载 | FTW 从 NVMe | 3-5s | 一次性 |

### 10.3 推荐架构（256K 约束下）

1. **KV = DSA 稀疏 + Q4/fp8 量化**：近端 4-8K token 全量精确 + 远端重要 token
   （H2O 式全局选择），存储 0.15-0.6GB（显存/内存均可），每 token 只读 0.15GB。
   仓库已有 DSA 基础设施（glm_moe_dsa / m3_sparse）可复用。
2. **decode：GPU attention（稀疏 KV 驻留显存）+ CPU FFN（DRAM 直读权重）**，
   层间只交换 KB 级 x/g（pinned 双缓冲，bf16→fp8）。
3. **prefill：GPU + 权重按层 PCIe 流式**（每层权重只传一次，总量 15GB），
   256K ≈ 13 min；这是内存方案里唯一可行的 prefill 路径。
4. **可选级 2**：显存剩余（8.6GB − 稀疏 KV 0.2 − attention 权重 5GB ≈ 3.4GB）
   驻留 ~14 层 FFN → 多请求流水 50-80 tok/s。

### 10.4 取舍提示
- DSA 稀疏会对远端 token 有信息取舍（保留重要 token、丢次要 token）——
  256K"压缩上下文"的代价即在此；近端窗口 + 全局锚点保证主要语义不丢。
- 若必须**逐 token 精确回忆** 256K：无解（带宽物理墙），只能接受稀疏。


---

## 11. 全新模式：Tri-Engine Affinity Partitioning（三引擎亲和分区）

### 11.0 被忽略的硬件事实：这台机器有 3 个引擎、2 个独立带宽池

| 引擎 | 算力 | 可读的权重介质 | 带宽 |
|---|---|---|---|
| **dGPU** RTX 4070 Laptop | 19 TF FP16 | 显存（驻留权重） | **256 GB/s** |
| **iGPU** Radeon 780M（12 CU） | 8.9 TF FP16 | **共享 DRAM（无 PCIe 墙！）** | ~40-50 GB/s（待实测） |
| CPU 8 核 Zen4 | 0.28 T MAC/s | DRAM | 51 GB/s |

现有所有方案（纯 CPU / GPU 流式 / GPU attention+CPU FFN）都只用了
**CPU + DRAM 一个带宽池**，dGPU 闲置、**iGPU 完全闲置**。而 dGPU 的显存
256GB/s 和 DRAM 51GB/s 是两个**互不干扰的带宽池**——可以同时满负荷。

### 11.1 架构（不是流式、不是全 CPU、不是层内分工——是"层组亲和"）

27B NVFP4（15GB）分为两组：

- **A 组（8.6GB，全部 attention + 23 层 FFN）→ 驻留显存 → dGPU 算**
  attention 权重 5GB + FFN 3.6GB，一次加载后零 PCIe 流量；KV（分段摘要）也驻留显存。
- **B 组（6.4GB，41 层 FFN）→ 驻留内存 → iGPU 算（不是 CPU！）**
  iGPU 共享 DRAM 直读权重（~45GB/s，免 PCIe），算力 8.9TF 远超 CPU 的 0.28T。
- **CPU → 彻底解放**：只做调度、预取、激活交换、采样。
- 层间交接只有 KB 级 x/g（dGPU↔内存 6.7GB/s 微不足道；iGPU 共享内存零拷贝）。

### 11.2 为什么这是"全新"且能突破

1. **解除 CPU 算力墙**：内存侧计算从 CPU（0.28T）换成 iGPU（8.9TF，30 倍）——
   之前 bf16 dot 算力封顶 7.8 tok/s 的问题消失，M 的收益**恢复**。
2. **双带宽池同时满负荷**：dGPU 读显存 256GB/s ∥ iGPU 读 DRAM ~45GB/s，
   二者独立——27B 多请求吞吐从 5-7 → ~38 tok/s（5-7 倍）。
3. **iGPU 是零成本第三引擎**：共享 DRAM 无 PCIe 瓶颈（对比 dGPU 的 6.7GB/s）。
4. **单请求也受益**：attention 权重 5GB 不再经 DRAM（驻留显存），CPU 侧
   DRAM 流量从 15GB 降到 6.4GB → 2.7 → 5.7 tok/s。

### 11.3 吞吐（估算，iGPU 带宽 45GB/s 假设，待实测）

| 场景 | 数字 |
|---|---|
| 单请求 decode | ~5.7 tok/s（dGPU 段 34ms + iGPU 段 142ms 串行） |
| 多请求流水（M≥2） | **~38 tok/s**（dGPU 30 + iGPU 8，带宽池独立） |
| 上下文 | 256K（KV 分段摘要 0.5-1GB 显存） |
| prefill 256K | dGPU 按层流式 ~12-13 min（算力主导） |

### 11.4 KV：写入时增量摘要（与三引擎正交的新机制）

- 256K 上下文按 8K 段分片；**每段在写入时增量计算段摘要**（512 个代表性 KV，
  H2O 式全局选择），decode 时 GPU 只读：当前段 KV（8K）+ 全部段摘要（32×512）≈
  24K token × Q4 = 0.77GB/token → 显存 256GB/s → 333 tok/s，KV 完全不是瓶颈。
- 与"DSA 读取时稀疏"不同：**压缩在写入时完成，读取零额外开销**。

### 11.5 验证计划（Phase 0.5）

1. **实测 iGPU 读 DRAM 带宽**（OpenCL 顺序读 512MB）：确认 40-50GB/s 假设；
   若 <30GB/s，B 组权重可部分回退给 CPU（带宽不变，算力墙回来）。
2. 微基准加"iGPU 侧 FFN GEMV"（OpenCL 内核，W4A8/W4A16）→ 实测 iGPU 端到端。
3. 若达标 → 三引擎原型（层组流水 + 分段 KV 摘要）。

### 11.6 物理极限备注
总吞吐 ≈ 显存带宽÷A组权重 + DRAM带宽÷B组权重（请求并行）≈ 256/8.6 + 51/6.4
≈ 30 + 8 = **38 tok/s（27B 本机天花板）**。任何模式（包括本方案）都到不了
"显存 256GB/s ÷ 15GB 全驻留"的 17 tok/s 单流——因为 15GB 装不进 8.6GB 显存；
也到不了纯 CPU 的更高值——因为 DRAM 只有 51GB/s。38 tok/s 就是这台机器的
物理上限，本方案是唯一能同时吃满两个带宽池的架构。


---

## 12. 实测修正（2026 实测后）：iGPU/NPU 出局，三引擎回归两引擎

### 12.1 iGPU（Radeon 780M）实测

- OpenCL：AMD 驱动未注册 ICD（注册后 clGetPlatformIDs 挂起 → 撤销）；微软
  OpenCL-on-DX12 层未安装 → **OpenCL 路线不可用**。
- DirectML（onnxruntime 1.24 + DmlExecutionProvider，MatMul 512MB fp16 B 矩阵）：
  - device_id=0（dGPU 4070，经 PCIe）：**5.1 GB/s**（与 cudaMemcpy 实测 6.7 一致量级）
  - **device_id=1（iGPU 780M，共享内存）：8.3 GB/s**
  - 端到端含 ORT 输入拷贝开销；即便最优也只 8-15 GB/s 量级。
- **结论：iGPU 读 DRAM 权重实际可用带宽 ≈ 8-15 GB/s，远低于假设的 40-50。**
  B 组（6.4GB）若交给 iGPU 只贡献 ~1.3-2.3 tok/s——**不值得**。
  CPU 直读 DRAM（51GB/s）反而快 4-6 倍。

### 12.2 NPU（AMD IPU / XDNA 1 代）评估

- 规格：8 TOPS INT8（= 4 T MAC/s），共享 DRAM（不新增带宽池）。
- 定位分析：
  1. **带宽不新增**：NPU 与 CPU/iGPU 共享同一 DRAM 池（51GB/s 上限），
     三者读权重合计不可能超过 51GB/s；
  2. **算力弱于 iGPU**：4 T MAC/s < 8.9TF（且 INT8-only，需 W4A8 全定点路径）；
  3. **框架开销**：Ryzen AI 运行时（ONNX EP）与 DirectML 同构，预计
     5-15 GB/s 量级（同 iGPU 教训）。
- **结论：NPU 在主链路（FFN 权重读取）无增量**。仅适合副业：KV 写入时
  摘要压缩、激活量化、采样——不占主链路带宽的预处理。

### 12.3 修正后的最佳架构（两引擎，实测带宽）

| 引擎 | 负责 | 带宽（实测/规格） | 贡献 |
|---|---|---|---|
| dGPU（4070） | A 组：attention + 23 层 FFN（8.6GB 显存驻留） | 显存 256GB/s | ~30 tok/s |
| **CPU（8 核）** | B 组：41 层 FFN（6.4GB DRAM） | **DRAM 51GB/s（实测）** | ~8 tok/s |
| iGPU / NPU | 出局（8-15GB/s 实测，不值得） | — | 副业（KV 摘要/量化） |

**总吞吐 ≈ 38 tok/s（M≥2 请求流水）**——与 §11 的物理上限一致，但实现路径
从"三引擎"简化回"两引擎 + 副业卸载"。单请求 ~5.7 tok/s 不变（CPU B 组主导）。

### 12.4 对 §11 的修正
- §11 的 iGPU 假设（40-50GB/s）被实测否定；吞吐上限 38 tok/s 不变
  （带宽池逻辑独立于具体引擎），实现改为 dGPU+CPU 两引擎。
- 启动自动带宽校准（类似 `ft bench bw`）：实测 PCIe/DRAM/共享内存三条
  通路，按实测值自动分配 A/B 组与引擎——避免再被假设坑。


---

## 13. FreeToken 稠密模型流式加载（Dense Stream Offload v2）— 最终设计

### 13.0 硬件与约束（用户确认）

- 缓冲池：RTX 4070 Laptop **8GB 显存** + **48GB DDR5**（实测 47.2GB 可用，DRAM 顺序读 51GB/s）
- 算力源：8 核 Ryzen 9 7940H + RTX 4070（19TF）+ AMD NPU（XDNA 8TOPS INT8）+ Radeon 780M iGPU（8.9TF，共享缓冲池）
- 目标：27B 稠密 NVFP4（15GB），**256K 上下文**，权重常驻内存，GPU 辅助计算

### 13.1 iGPU 共享缓冲池的正确理解（修正 §12 的误判）

- 实测 8.3GB/s 是 **onnxruntime DirectML 每次 session.run 全量拷贝输入的框架开销上限**，
  **不是 iGPU 读共享内存的硬件带宽**。权重若零拷贝驻留共享池（原生 D3D12/Vulkan 计算
  内核，或 DML 复用已映射 buffer），iGPU 读 DRAM 可达 **30-50GB/s**（RDNA3 共享池典型值，
  需实测验证）。
- **但物理限制不变**：iGPU 与 CPU 共享同一个 DRAM 带宽池（上限 51GB/s 实测）——
  "共享缓冲池"既是优势（零拷贝、无 PCIe 墙）也是天花板（不新增带宽池）。
- **推论**：CPU 实测 51GB/s 已到池上限；iGPU 只能**竞争**这个池，不能增加它。
  B 组权重的读带宽：CPU 51GB/s（实测）> iGPU 假设 30-50GB/s（待验证）→ **默认 CPU 读，
  iGPU 做"零权重副业"**；若实测 iGPU 原生路径 > CPU 当前可用（CPU 同时被占用时），
  自适应切换。

### 13.2 架构总览（两主引擎 + 副业卸载）

| 分组 | 权重 | 引擎 | 带宽 | 吞吐贡献 |
|---|---|---|---|---|
| A 组 | attention 全部（5GB）+ 13 层 FFN（3GB）= **8GB 显存驻留** | dGPU 4070（CUDA） | 显存 256GB/s | ~30 tok/s |
| B 组 | 51 层 FFN（**7GB 内存常驻**） | CPU 8 核（实测 51GB/s） | DRAM 51GB/s | ~7 tok/s |
| 副业 | KV 摘要/量化、激活量化、采样 | iGPU 780M（8.9TF）/ NPU | 共享池（不占主链路） | 卸载 CPU 杂活 |
| KV | 分段摘要（8K 段 + 段摘要，Q4）约 0.5-1GB 显存 | — | — | 256K 上下文 |

- 单请求 decode：dGPU 段（8GB/256=31ms）+ CPU 段（7GB/51=137ms）约 **5.9 tok/s**
- 多请求流水（M>=2）：30 + 7 约 **37 tok/s**（实测带宽的两池并行上限）
- prefill 256K：dGPU 按层流式（每层权重 PCIe 传一遍共 15GB）+ 算力主导约 **12-13 min**
- NPU 定位：KV 写入时摘要/量化副业（INT8 够用，不占主链路）

### 13.3 FreeToken 接入点（具体到文件）

1. **python/freetoken/kernel/backend.py**：新增 is_dml_available()（onnxruntime-directml 探测）
2. **python/freetoken/models/quant_linear.py**：工厂新增 make_dense_stream_linear(...) 分支 →
   返回 DenseStreamLinear 包装：持有权重 host 视图与引擎标记（dGPU 显存 / CPU 内存）；
   forward(x) 按标记调度（CUDA 内核 / Triton CPU 内核）；替换 Nvfp4DenseLinear 的 decode 路径
3. **python/freetoken/engine/engine.py**：_adjust_config 在 fused 之外新增 dense-backend stream
   解析（1074-1120 行强制 fused 处扩展）；逐层循环 A 组层走 CUDA、B 组层走 CPU 内核，
   层间只交换 KB 级 x/g（pinned 双缓冲 + cudaMemcpyAsync 与 CPU 计算重叠）
4. **权重加载**：A 组 8GB 一次性 cudaMemcpy 进显存（约 1.2s @6.7GB/s，仅启动一次）；
   B 组驻留内存零拷贝。启动带宽校准自动决定 A/B 边界（按实测 PCIe/DRAM 平衡）
5. **KV 分段摘要**：新模块 kv_summary.py（或复用 glm_moe_dsa 选择逻辑）：
   写入时增量选 512 个代表性 KV/段（Q4），decode 只读当前段 + 全部摘要

### 13.4 实施顺序（每步可验证）

1. **P1 校准器**：ft bench bw 实测 5 条通路（PCIe H2D/D2H、DRAM、iGPU 原生、NPU），
   产出 A/B 边界建议。iGPU 原生带宽测试 = D3D12 计算着色器顺序读（绕过 ORT）
2. **P2 A 组先行**：attention + 13 层 FFN 驻留显存，其余 CPU —— 纯 CUDA+CPU 混合，
   无 DML 依赖；验证单请求 5.9 tok/s、多请求 37 tok/s
3. **P3 iGPU 副业**：KV 摘要/激活量化卸载到 DirectML；验证主链路吞吐不变 + CPU 空闲
   更多（为 B 组提速留余量）
4. **P4 自适应引擎**：若 iGPU 原生路径实测 >40GB/s，B 组切换 iGPU（CPU 完全解放），
   实测对比决定

### 13.5 风险与取舍
- iGPU 原生（D3D12/Vulkan）内核开发量大（W4A8 GEMV 计算着色器）——P2 先行可零风险落地
- NPU 走 ONNX EP 有同款拷贝开销，只做副业，不做主链路
- 单请求 5.9 tok/s 是内存方案硬墙（B 组 7GB/51GB/s），要更快只能：内核优化（W4A8 反量化）
  让 CPU 算力跟上 M 增长，或换 PCIe 更快的机器
- "流式加载"严格说是"驻留分区 + 层流水"：A 组驻留显存，B 组驻留内存，
  没有逐 token 的权重搬运（那是被实测否决的 GPU 全流式）


---

## 14. Fresh Design v3：BW-Pool Maximal（从 0 重构，只基于实测物理账）

### 14.1 从 0 出发的物理账（全部实测/规格，无假设）

可用带宽池与算力（本机）：

| 池/引擎 | 带宽（实测/规格） | 容量 | 读者/算者 |
|---|---|---|---|
| 显存池 | 256 GB/s（规格） | 8.6GB | dGPU 独占 |
| DRAM 池 | **51 GB/s（实测）** | 48GB | CPU / iGPU / NPU **共享** |
| PCIe | 6.7 GB/s（实测） | — | 仅启动预载 / 激活交换 |
| CPU 算力 | 0.28 T MAC/s（实测 bf16 峰值） | — | 当前内核 W4A8 仅 0.09T（可优化 3-4 倍） |
| iGPU 780M | 8.9 TF（规格）；读共享池 8.3（ORT 实测下限）~30-50（原生待测） | — | 共享池读者（弱）/ 副业 |
| NPU XDNA | 8 TOPS INT8（规格） | — | 副业（INT8-only） |

**不可改变的约束**：15GB 权重 > 8.6GB 显存 → 必须分驻留；每 decode token
必须处理全部 15GB 权重；读取介质只有三个池。

### 14.2 数学最优分配（零 PCIe 权重流）

- **A 组 = 8GB 显存**（attention 全部 5GB + 13 层 FFN 3GB）→ dGPU：256/8 = **32 tok/s**
- **B 组 = 7GB DRAM**（51 层 FFN）→ 共享池读者：51/7 = **7.3 tok/s 带宽侧**
- **PCIe 每 token 权重流量 = 0**（仅启动时预载 8GB ≈ 1.2s；层间只走 KB 级激活）

### 14.3 性能（基于实测，分 M 与内核状态）

| 场景 | 公式 | 当前内核 | 内核优化后（W4A8 0.09T→0.3T） |
|---|---|---|---|
| 单请求 | 1/(31ms+137ms) | **5.9 tok/s** | 5.9（带宽墙不变） |
| 多请求 M=2 | 32 + min(7.3×2, 9.8) | **~42 tok/s** | ~44（算力墙 9.8→34） |
| 多请求 M=4 | 32 + min(7.3×4, 34) | ~42（CPU 算力墙 9.8） | **~61 tok/s** |
| prefill 256K | max(dGPU 算力 12min, B组读 4.4min) | ~12 min | ~12 min（算力主导） |
| 上下文 | 分段摘要 KV（8K 段+摘要，Q4）0.5-1GB 显存 | **256K** | 256K |

**关键发现（Fresh 视角）**：
1. 单请求 5.9 tok/s 是**带宽墙**（B 组 7GB/51GB/s），内核优化救不了单请求；
2. 多请求下 B 组吞吐 = min(7.3×M, CPU 算力墙)——**当前内核把 M 锁死在 1.34**，
   内核优化把 M 解锁到 4.7 → 总吞吐 42 → 61 tok/s；
3. iGPU/NPU 对主链路带宽**无增量**（共享池 51GB/s 是总池），但能卸载 CPU 副业
   和分担 prefill 算力——"用上所有设备"的落点是分工，不是叠加带宽。

### 14.4 全设备分工（最大化利用的最终形态）

| 设备 | 角色 | 负载 |
|---|---|---|
| dGPU 4070 | A 组计算（显存池独占） | attention + 13 层 FFN；decode 主力 + prefill 主力 |
| CPU 8 核 | B 组计算（DRAM 满速 51GB/s） | 51 层 FFN；共享池主读者 |
| iGPU 780M | 零权重副业 + prefill 分担 | KV 写入时分段摘要、激活量化（bf16→int8）、采样辅助；prefill 时利用共享池余量算 B 组层（8.9TF） |
| NPU XDNA | INT8 副业 | KV Q4 压缩打包、logits 后处理、量化参数更新 |

流水（decode，多请求）：CPU 算 B 组层 L 的 51 层 ∥ dGPU 算 A 组层 L' ∥ iGPU 写 KV
摘要（每 token 128KB）∥ NPU 压缩 KV——四条链并行，三个带宽池（显存、DRAM、
KV 摘要内共享池余量）同时工作。

### 14.5 与旧方案的差异（Fresh 声明）
- 不做"GPU 全流式"（PCIe 6.7 实测否决）；
- 不做"iGPU 读 B 组"（共享池不叠加，CPU 51GB/s 实测最优；iGPU 原生带宽待 P1 校准，
  若 >51 才反转——物理上不可能）；
- 不做"层内 attention/FFN 分工"（A/B 按层组切分，单引擎内连续层，流水开销最小）；
- **新增**：M 扩展 + CPU 内核优化的组合解锁（42→61）、iGPU/NPU 副业卸载、
  三带宽池同时满负荷（307GB/s 理论合并）。

### 14.6 实施（每步验证）
1. **P0 校准器**：实测 iGPU 原生带宽（D3D12 计算着色器顺序读，绕过 ORT）+ NPU 实际
   吞吐（Ryzen AI 运行时）→ 填表 14.1 的两个待测格；
2. **P1 两引擎原型**：A 组 CUDA + B 组 CPU 内核（沿用 cpu_moe_ext 的 GEMV），
   层间 pinned 双缓冲；验证 5.9 / 42 tok/s；
3. **P2 副业卸载**：KV 分段摘要 + 激活量化 → iGPU（DirectML），验证主链路不变；
4. **P3 内核优化**：W4A8 VNNI 反量化 0.09T→0.3T（微基准 A/B），解锁 M → 61 tok/s；
5. **P4 prefill 加速**：iGPU/NPU 分担 B 组层 GEMM，256K prefill 12→8 min。


---

## 15. 实测可行性验证（Fresh v3 的裁决）

### 15.1 全实测带宽表（2026 实测）

| 通路 | 实测 | 规格/理论 | 状态 |
|---|---|---|---|
| DRAM 顺序读（16 线程） | **51 GB/s** | 89.6 | ✅ 57% |
| PCIe H2D / D2H（512MB） | **6.7 GB/s** | 16（x8 gen4） | ✅ 42% |
| **dGPU 显存 D2D（2-3GB）** | **108 GB/s** | 256（GDDR6 128bit） | ⚠️ **仅 42%** |
| dGPU 显存驻留上限 | **~7GB**（8GB 触顶崩至 13GB/s） | 8.6 | ⚠️ 有效容量 7GB |
| CPU bf16 dot（8T） | 0.28 T MAC/s | — | ✅ |
| CPU W4A8 VNNI（8T） | 0.09 T MAC/s | — | ✅ 瓶颈 |
| iGPU 读共享池（ORT） | 8.3 GB/s | 30-50（原生待测） | ⚠️ 框架下限 |
| NPU | 未测（副业定位不变） | 8 TOPS INT8 | — |

### 15.2 修正后的架构数字（全实测驱动）

- **A 组**：7GB 显存驻留（attention 5GB + ~9 层 FFN 2GB）→ 108GB/s ÷ 7GB = **~15 tok/s**
- **B 组**：8GB 内存（~55 层 FFN）：
  - 带宽侧：51 ÷ 8 = **6.4 tok/s**
  - 算力侧（当前 W4A8 0.09T 实测）：29.8 GMAC/token → **3.0 tok/s** ← 当前墙
  - 算力侧（内核优化至 0.3T）：→ 10 tok/s，带宽墙 6.4 主导 → **6.4 tok/s**
- **单请求**：71ms（A 段）+ 137ms（B 段带宽）= 208ms → **4.8 tok/s**（优化后）；
  当前内核 71 + 331ms = 402ms → **2.5 tok/s**
- **多请求流水**：15 + min(6.4×M, 算力墙)：
  - 当前：15 + 3.0 = **~18 tok/s**
  - 优化后 M=2：15+12.8 = 27.8；M=4：15+25.6 = **~41 tok/s**
- **prefill 256K**：dGPU 算力主导 ~12-13 min（不变）

### 15.3 可行性裁决

| 问题 | 裁决 |
|---|---|
| 架构可行吗？ | ✅ 可行：三个带宽池实测确认（51/6.7/108），零 PCIe 权重流成立 |
| 256K 上下文？ | ✅ 分段摘要 KV 0.5-1GB 显存，显存预算内（7GB 权重 + 1GB KV + 0.6 余量） |
| 单请求快吗？ | ⚠️ 2.5→4.8 tok/s——**慢于 12B fused 的 39**；只有 256K 是硬需求时才选它 |
| 多请求有竞争力？ | ✅ 18（当前）→ 41 tok/s（优化后）——agent 多路场景成立 |
| **最大杠杆** | **W4A8 内核优化（0.09→0.3T）**：B 组 3.0→6.4，总 18→41（2.3 倍） |
| 风险 | dGPU 显存仅 108GB/s（规格 42%）→ A 组 15 tok/s 是墙；iGPU/NPU 无带宽增量（共享池） |

### 15.4 结论
- **架构成立，收益低于设计假设**（A 组 32→15，因显存带宽实测只有 108）；
- 单请求无优势（256K 需求除外）；多请求是唯一有竞争力的场景（18-41 tok/s）；
- **下一步唯一确定高杠杆动作：W4A8 反量化内核优化**（微基准 A/B，0.09→0.3T 目标），
  同时解锁 B 组带宽墙与 M 扩展——与 MoE CPU 内核双赢；
- iGPU 原生带宽（D3D12 微基准）与 NPU 吞吐留待 P1 原型阶段校准，不影响上述裁决。


---

## 16. Vulkan 路径实测：iGPU 共享池翻身（2026 实测）

### 16.1 实测结果（vulkan Python 绑定 + VkCmdCopyBuffer，host-visible 共享内存）

| 测试 | 结果 |
|---|---|
| Vulkan 设备枚举 | 2 个：RTX 4070（0x10de/0x2860）+ **Radeon 780M（0x1002/0x15bf）** ✅ |
| 780M copy 1GB（共享内存） | **28.4 GB/s**（双向流量，读等效 ~57 GB/s） |
| 780M copy 2GB | **28.1 GB/s**（读等效 ~56） |
| 780M copy 4GB | ❌ VkErrorUnknown（**单块 host-visible 共享内存上限 <4GB**，需分块分配） |
| 对照：ORT/DirectML 8.3 GB/s | **框架拷贝开销吃掉 3.5 倍带宽** |

**结论：iGPU 原生读共享池 ≈ 28-57 GB/s（copy 引擎），SM 计算读只会更高。
用户对"共享缓冲池"的判断被实测确认——之前 8.3GB/s 是 ORT 的锅，不是硬件的。**

### 16.2 对架构的影响（修正 §15）

1. **B 组读者重排**（共享池 51GB/s 上限不变，但读者能力变了）：
   - 单请求 B 组：iGPU 28GB/s ÷ 8GB = **3.5 tok/s**（copy 带宽，算力 8.9TF 足够）
     vs CPU W4A8 当前 3.0 tok/s → **iGPU 略胜，且不占 CPU**
   - 多请求 B 组：CPU（3.0）+ iGPU（3.5）并行分担不同请求 → **~6.5 tok/s ≈ 带宽墙 6.4** ✅
   - CPU W4A8 优化到 0.3T 后：CPU 10.4 算力 + iGPU → 带宽墙 6.4（B 组已封顶）
2. **修正后性能**：
   - 单请求：A 组 71ms + B 组（iGPU）286ms = 357ms → **2.8 tok/s**（当前，copy 带宽）
     SM 读若 ~40GB/s → 3.7 tok/s；CPU 内核优化后 B 组满带宽 6.4 → 4.4 tok/s
   - 多请求流水：A 15 + B 6.4 ≈ **21.4 tok/s**（当前内核即达，CPU+iGPU 分担）
3. **工程注意**：共享内存单块分配上限 <4GB → B 组 8GB 权重需分 3-4 块
   （VkMemory 分块 + offset 寻址，或按层分 buffer）；
   权重零拷贝驻留 = host-visible 分配 + CPU 侧直接写（WDDM 下同驱动映射）。

### 16.5 vkpeak 实测：驱动没坏，SM 计算路径可用（2026 实测）

用户提供预编译 vkpeak（nihui/vkpeak 20260527，glslang 规范编译着色器）——
在 780M 上**完整跑通**，推翻 §16.4 的"驱动不可用"结论：

| 场景 | 780M 实测 | 意义 |
|---|---|---|
| fp16-matrix | **16014.56 GFLOPS** | SM 计算路径完全可用（理论 ~17.8T FMA） |
| int8-matrix | **15054.26 GIOPS** | INT8 15T 达标 |
| copy-d2d（计算着色器 SM copy） | **15.10 GB/s** | SM 读+写各 15.1（**SM 读带宽下界**） |
| copy-h2d / d2h | 13.45 / 13.37 GB/s | 共享内存主机路径 |
| 驱动 | AMD 25.9.2 (LLPC) | Vulkan 1.4.315 |

**修正结论**：
1. 之前手写 SPIR-V 崩溃 = **AMD LLPC 对手写 SPIR-V 的边缘用法不兼容**（非驱动整体故障）；
   NVIDIA 上同样 SPIR-V 全部通过（验证了 SPIR-V 有效性，问题在 AMD LLPC 特定指令）。
2. **SM 读带宽**：copy 场景 15.1（读+写）是下界；纯读（GEMV 风格）预计 20-40 GB/s，
   **待 glslang 编译 read_bw.glsl 实测**。
3. B 组 iGPU 带宽修正：保守 15.1 GB/s → 1.9 tok/s；若 SM 读实测 >28 → 3.5+ tok/s。
4. vulkan pip 绑定的 VkComputePipelineCreateInfo 构造 bug 仍成立（纯 cffi 已绕过）。

**待办**：用户下载 glslang（KhronosGroup/glslang releases）或 Vulkan SDK →
编译 read_bw.glsl → 复用 igpu_vk_read2.py 的纯 cffi 管线实测 SM 读带宽。

### 17.1 多行 GEMV 微基准（生产场景，2026 实测）

bench_w4a8_multi.cpp：M 行 × K=4096，共享 int8 act（-127..127），多线程
（std::thread 均分行），clang-cl /arch:AVX512。正确性全 OK。

| M | threads | 时间 | MAC/s | GB/s |
|---|---|---|---|---|
| 32 | 1 | 0.036 ms | 3.6 G | 2.2 |
| 32 | 16 | 0.304 ms | 0.4 G | 0.3（线程开销主导，行太少） |
| 512 | 16 | 0.360 ms | 5.8 G | 3.3 |
| 4096 | 16 | 0.395 ms | **42.5 G** | **23.9** |

- **生产 gate_up 尺寸（M=4096）16 线程：0.395 ms/专家、42.5 G MAC/s**——
  带宽受限区（23.9 GB/s ≈ DRAM 51 GB/s 上限的 47%）
- 对比单行 25 G MAC/s：多行多线程提升 1.7×
- §15 的 0.3T 目标需更高带宽利用（软件预取/行批调度/核亲和）——
  42.5 G 已确认 W4A8 内核在多行场景下无计算瓶颈（纯带宽墙）

### 17.2 生产落地项（-128 溢出修复）

nvfp4_i8 / avx512vnni 的激活（asi8/asb）为**离线预量化**：本仓库的
python/freetoken 与 csrc/cpu_moe 均不生成 asi8（cpu_moe_ext.cpp 的
fp8_roundtrip_bf16 是 bf16→bf16 FP8 往返，非 int8 量化器）。int8 激活
（block-128 FP8 round-trip 的整数部分）由 GPU 侧 W4A8 路径（Triton
quant kernel / 权重打包脚本）产出。

**修复方案**：在 GPU 侧激活量化器（int8 写入点）把 round 结果 clamp 到
[-127, 127]（与微基准 rand()%255-127 一致）。注意：DeepSeek-V4 官方
FP8 round-trip 用 e4m3 clamp ±448——int8 版本的等价实现是
clamp(round(x/scale), -127, 127)。若 -128 仍出现（max|x|/scale=128 精确
边界），kernel 侧 sign-trick（sa = -a 对 -128 溢出）会输出错误——两个
修复点任选其一：量化器 clamp（首选）或内核改用
sa = a ^ (w<0 ? 0xFF : 0) + (w<0 ? 1 : 0)（不溢出 s8 的符号翻转）。

### 17.3 B 组 iGPU 内存管理（VMA）

已下载 VulkanMemoryAllocator v3.4.0（GPUOpen-LibrariesAndSDKs，
单头文件 772KB）→ benchmarks/cpu_moe_microbench/tools/vma/vk_mem_alloc.h。
用途（B 组 offload 调度）：
- 权重银行 iGPU staging（copy 引擎源/目标，VMA_MEMORY_USAGE_GPU_TO_CPU/
  CPU_TO_GPU 或 HOST_VISIBLE 池）
- 激活输出暂存（host-visible，CPU 直接读）
- 专家权重生命周期管理（加载/卸载，复用内存块）
DCE 问题未解前 VMA 用于 copy+CPU 方案；若 AMDVLK 实验成功（LlpcOptions
-llpc-opt=0）则用于计算 GEMV 方案。

### 18 B 组 iGPU Vulkan GEMV 内核（A 项，2026 设计）

### 18.1 AMDVLK 实验（DCE 绕行路径，2026 实测源码确认）

LLPC 的 DCE 是 LLVM 优化流水线标准阶段（AMDVLK/LLPC 原理：SPIR-V→LLVM IR→
优化[含 DCE]→目标代码）。**LLPC PipelineOptions.optimizationLevel（0-3）可关
优化**；xgl（AMDVLK 的 Vulkan 层）暴露 **LlpcOptions** 设置项（string，空格
分隔的 amdllpc 风格选项，必须以 "-" 开头，追加进 Llpc::ICompiler::Create）——
已从 xgl 源码 compiler_solution_llpc.cpp 确认（settings.llpcOptions 拆分后并入
llpcOptions 数组；amdllpc 的对应选项为 cl::opt LlpcOptLevel("llpc-opt")）。

**实验步骤**（需用户下载 AMDVLK Windows 版）：
1. 下载 https://github.com/GPUOpen-Drivers/AMDVLK/releases（浏览器）并安装
2. 建配置目录（如 C:\Users\Administrator\amdvlk\）含 amdvlk.json：
   { "LlpcOptions": "-llpc-opt=0" }
3. 设环境变量 AMD_CONFIG_DIR=<该目录>
4. 设 VK_ICD_FILENAMES=<AMDVLK 安装目录>\amdvlk64.json（或用 amdvlk_check.py
   确认设备/vendor 不变）
5. 重跑：python igpu_gemv_chain.py 1024 4096 512 20
   判据：y1/y2 非预填充值（0x5A5A5A5A）且时间 > 空跑基线 → DCE 关闭成功
6. 若 -llpc-opt=0 不生效，试 -O0 / -disable-llvm-opt（amdllpc 其它优化开关）

失败则 B 组定型 copy 引擎 + CPU 算（28.4 GB/s 真实，见 §16.6/§17.1）。

### 18.2 D3D12 备选（探索状态）

AMD 的 D3D12 驱动与 Vulkan LLPC 不同编译栈——理论上无 LLPC DCE，但未验证。
探索记录：
- pip pyd3d12（PIX 绑定）在 Python 3.14 上 typelib 版本检查失败
  （comtypes._check_version "Typelib different than module"）——monkeypatch
  comtypes._check_version 后可 import，但函数签名/常量名需逐一核对
  （D3D12CreateDevice 需 4 参数 + IID；常量是 D3D_FEATURE_LEVEL_12_0）
- dxc.exe 已定位（Windows SDK 10.0.26100.0 x64）且 HLSL→DXIL 编译成功
  （t_d3d12.hlsl → t_d3d12.dxil，2880 B）
- 若 AMDVLK 实验失败且需要 D3D12：用 ctypes 直调 dxgi.dll/d3d12.dll
  （枚举 AMD 适配器 → 设备 → 根签名 → 管线 → dispatch → 读回），
  或修 pyd3d12（comtypes 1.4.10 + typelib 缓存）



**目标**：780M 直接用 Vulkan 计算着色器做 MoE 专家 GEMV（读 DRAM 权重银行 →
计算 → 写激活），验证 LLPC 对真实计算内核的行为（§16.6 的 DCE 只影响
"结果无观测"的微基准；真实 GEMV 输出被消费，不应被消除）。

**格式**：W4A8 NVFP4（与 CPU 内核 cpu_moe_ext.cpp 同格式，共享权重银行）：
- packed：半字节权重（2 权重/字节，低 nibble 在前），每 16-K 块 = 8 字节
- scale：per-16 块 e4m3（uint8），global：per 行 fp16
- act：int8 激活（per-16 块 [even(8),odd(8)] 布局）+ asb（per-16 块 fp32 scale）
- 解码：w = E2M1x2[nibble] × e4m3(scale) × global × 0.5；acc = Σ w×act×asb

**Shader 布局**（igpu_gemv.comp，GLSL 450，glslang 编译）：
- binding 0: packed uint[]（只读）
- binding 1: scale uint8[]（只读）
- binding 2: act int8[]（只读，所有行共享 → L2 复用）
- binding 3: asb float[]（只读）
- binding 4: out float[]（写）
- push_constant: { K, nb_per_row, global }
- 每 work-item = 一行：local_size 256，dispatch ceil(M/256)

**管线**：复用 igpu_vk_read2.py 的纯 cffi 结构（AMD 已验证 VK_SUCCESS）。

**验证计划**：
1. 正确性：随机权重/激活 → GPU GEMV vs CPU 参考（scalar 逐块）→ 全一致
   （排除 -128 溢出：act clamp -127..127）
2. 吞吐：M=2048（gate_up 行数）、K=4096 → 权重读量 = M×K/2 B =
   4 MB/专家 → 带宽 = 读量/时间 vs 28.4（copy 引擎）/ 15.1（SM copy）
3. LLPC 行为：结果回读校验（nonzero/正确值）→ 确认未 DCE
4. 若吞吐 ≥ copy 引擎 → B 组直算可行（免两遍）；否则用 copy+CPU 方案

### 17 CPU W4A8 NVFP4 GEMV 内核验证（C 项，2026 实测）

**微基准**：bench_w4a8.cpp（从 cpu_moe_ext.cpp 提取 dot 内核，独立 clang-cl 编译，
无 CUDA/torch 依赖）。数据：随机权重/激活，scalar 为参考。

**正确性**（clang-cl，AVX512 编译）：
- **K=1024..8192 全部一致**（err ≤ 3.9e-3 = fp32 累加误差，相对 < 1e-6）
- **发现 -128 溢出 bug**：act=-128 且 w<0 时，sign trick 的 sa=+128 溢出 s8
  （位模式 0x80 = -128）→ dpbusd 符号错（K=768 曾 err 50%）
  **修复点**：激活量化器必须 clamp 到 -127..127（ggml/llama 标准）；
  排除 -128 后内核 100% 正确。
- **本机 CPUID**：7940H(Phoenix) **AVX-VNNI=0、AVX512-VNNI=1**——
  VEX vpdpbusd 非法指令（0xC000001D）；select_nvi8dot 自动选 avx512vnni ✓

**性能**（单 dot、单线程、clang-cl /arch:AVX512）：

| K | scalar | avxvnni(EVEX.256) | avx512vnni | 加速 |
|---|---|---|---|---|
| 1024 | 2.43 G MAC/s | 23.8 G | 20.1 G | 9.8× / 8.3× |
| 2048 | 2.43 G | 25.4 G | 21.6 G | 10.5× / 8.9× |
| 4096 | 2.46 G | 25.4 G | 21.9 G | 10.3× / 8.9× |
| 8192 | 2.51 G | 25.9 G | 22.4 G | 10.3× / 8.9× |

- **scalar → avx512vnni 8.9×、→ EVEX.256 10.4×**（§15 目标 3.3× 已远超）
- 带宽 3.7 → 38 GB/s = DRAM 51 GB/s 上限的 ~75%——**接近带宽墙**，
  多行 GEMV（32 行）+ 多线程的剩余空间有限（上限 ~34 G MAC/s）
- EVEX.256 快于 EVEX.512（512 位 decode 的 permutexvar/mask_blend 开销更大）

**编译环境发现**：
- **MSVC 不可靠**：VEX vpdpbusd 编码 bug（本机非法）+ __m512i 按值传参 ABI bug
  （独立 grp4 调用 0xC000001D）——**Windows 构建必须用 clang-cl**
- **LLVM 22.1.8 已装**（winget）：C:\Program Files\LLVM\bin\clang-cl.exe
- cpu_moe_ext.cpp 生产编译仍需 CUDA 头 + cudart（本机 torch CPU-only 无法
  验证生产路径）——**内核级验证完成，生产启用待 CUDA 环境**

**C 项结论**：W4A8 内核优化已验证（8.9-10.4×），生产启用前置 = 量化器
clamp(-128→-127) + clang-cl 构建 + CUDA 环境验证。

### 16.6 SM 读带宽实测：LLPC 激进优化阻塞（2026 实测）

**修正（§16.5 结论错误）**：vkpeak 的 fp16-matrix 16014 GFLOPS / int8-matrix
15054 GIOPS / copy-d2d 15.1 GB/s **全部无效**——LLPC 的 DCE 把"无 shader 内
消费者"的写消除后，**纯算+写 shader 也空跑**（实测 igpu_noread.comp：
c_blob 保持预填充 0x5A5A5A5A、时间=空跑基线 0.059ms）。
**"driver fine, SM path usable"判断错误**；正确判断：**780M LLPC 的 DCE 是
buffer 级终极消除**，SM 算力/读带宽**无法通过 compute shader 微基准测量**。

**DCE 对抗尝试（全部失败）**：
- writeonly / 非 writeonly / volatile：写仍消除
- atomicXor：消除；条件写（不可证明常量比较）：消除
- 双 dispatch 消费链（shader A 写 → shader B 读）：A 写仍消除（跨 shader）
- memoryBarrier + barrier：消除；shader 内读回自己：消除
- subgroupAdd + tid==0 条件写（llama.cpp mul_mat_vec 结构）：消除
- 唯一"读保留"模式：copy-pattern（read1/write1，out[gid]=data[gid]）——
  写消除但读真实执行（2.3ms → 28.8 GB/s SM 读）——机制未明（可能是
  LLPC 保守保留"顶层直通 load"）

**对真实推理的含义**：任何"读权重→算→写激活"的 compute GEMV 在当前驱动
（25.9.2）下会被整体消除（实测 GEMV_A→quant→GEMV_B 链：y1/y2 保持预填充
值、各阶段时间=空跑）。**llama.cpp Vulkan backend 的 mul_mat_vec.comp 结构
相同（读→算→subgroupAdd→tid0 写 dst）——在 780M 上同样会被消除**（除非
驱动行为不同版本有差异，或 llama.cpp 走 CPU fallback）。

### 17 CPU W4A8 NVFP4 GEMV 内核验证（C 项，2026 实测）

glslang 16.5.0 就位（用户下载），read_bw.glsl 系列编译成功，
纯 cffi 管线在 780M 上完整跑通（module/pipeline/descriptor/dispatch 全 VK_SUCCESS）。

**测量结果**（数据源：read_bw*.spv + igpu_vk_read2.py）：

| 着色器 | 结果 | 判定 |
|---|---|---|
| 空 body dispatch 65535 wgs | 0.055 ms 恒定 | 调度开销基准 |
| 纯读 4KB/项（XOR 累加写 1 值） | 12-17 TB/s（假） | **读被 LLPC 消除**（out 校验=0） |
| volatile 修饰纯读 | 同（假） | volatile 无效 |
| 非 writeonly 输出 | 同（假） | 无效 |
| atomicXor 输出 | 同（假） | **原子写也被消除** |
| copy 模式（读1写1，writeonly） | 28.8 GB/s | **写消除后为纯读时间**（2.3ms 真实） |
| copy 引擎（vkCmdCopyBuffer） | **28.4 GB/s** | 硬件行为，**最可信** |

**结论**：
1. **AMD LLPC 对"结果无观测"的读写执行激进 DCE**（含 atomic 副作用）——
   SM 纯读带宽**无法用微基准着色器直接测量**（驱动级限制，非测量错误；
   校验证实：dummy 预填 0xDEADBEEF 后 shader 写不生效）。
2. **可信的 iGPU 带宽数字**：
   - copy 引擎 28.4 GB/s（读+写，igpu_vk.py 硬件实测）
   - vkpeak SM copy 15.1 GB/s（读+写，glslang 产物，ncnn 风格未消除）
   - SM 算力：fp16 16T / int8 15T（vkpeak）
   - **SM 纯读估计 20-40 GB/s**（介于两者，未直接测）
3. **B 组 iGPU 内核可行性确认**：真实 GEMV（读权重→算→写激活，输出被后续层
   消费）**不会触发 DCE**——B 组内核可用；带宽按 15.1-28.4 保守估计 →
   **1.9-3.5 tok/s**（单请求 iGPU B 组）。
4. 测试资产保留：read_bw*.glsl/spv、igpu_vk_read2.py（纯 cffi）、glslang/
   vkpeak 工具——**驱动更新后可直接重测**（LLPC 修复则 SM 读出真值）。

### 16.4 Vulkan 计算着色器路径：驱动 bug 阻塞（2026 实测，被 §16.5 修正）

手写 SPIR-V 生成器（Python 无 glslang/spirv-tools wheel，dxc 无 SPIR-V 后端）
产出有效着色器（NVIDIA 全量验证通过：空 main / StorageBuffer 访问 / 8 次展开
读循环 / 32 次展开读循环），但本机两 GPU 驱动均无法完成计算管线：

| 阶段 | AMD 780M | NVIDIA 4070 |
|---|---|---|
| vkCreateShaderModule（SPIR-V 140-836B） | **间歇 INIT_FAILED / 崩溃**（空 main 也如此，驱动 bug） | OK |
| vkCreateShaderModule（3820B 复杂） | 崩溃（ACCESS_VIOLATION） | OK |
| vkCreateComputePipelines | 成功（纯 cffi，min 着色器） | **崩溃**（l3_5/read_bw2；min 返回 -13） |
| 结论 | **计算着色器路径不可用**（驱动级 bug） | 混合图形环境不稳定 |

vulkan pip 绑定亦有构造 bug（VkComputePipelineCreateInfo 嵌套 stage 导致崩溃，
纯 cffi 绕过后 AMD 管线创建 VK_SUCCESS）——测试全程使用纯 cffi 路径。

**对架构的影响（修正 §16.2）**：
- **B 组 iGPU 内核只能走 copy 带宽**：28.4 GB/s（copy 引擎，已实测）= 3.5 tok/s；
  SM 读（35-50GB/s 预期）**无法在当前驱动下实测**，待驱动更新或 D3D12 路径
  （dxc HLSL→DXIL 可用，AMD D3D12 驱动成熟——若后续需要，D3D12 是替代路线）
- 结论不变：B 组 = min(51GB/s 池上限, iGPU 28.4 或 CPU) —— 多请求 CPU+iGPU 并行
  仍达 ~6.4 tok/s 带宽墙；单请求 iGPU 3.5 tok/s
- 副业（KV 摘要/量化）：copy 路径可用（28.4GB/s），计算型副业受限

### 16.3 下一步（Vulkan 路径的价值落点）
- **P0b**：计算着色器 SM 读带宽实测（GLSL→SPIR-V，shaderc 或手写）——确认 GEMV
  真实读带宽（预计 35-50GB/s），并验证 W4A8 GEMV 内核原型（int8/子组指令）；
- **P1b**：B 组 iGPU 内核（Vulkan 计算管线，权重分块驻留共享池）——与 CPU 内核
  并存，启动校准择优；
- Vulkan 后端同时是副业（KV 摘要/量化）的低开销路径——不再依赖 DirectML/ORT。

## 18.3 D3D12 验证结果（2025-xx：AMD 780M 实测）

**背景**：LLPC（Vulkan）对无消费者的写绝对 DCE（§16.5/§16.6），B 组 iGPU GEMV 无法在
Vulkan 下验证。D3D12 走 AMD 的 DX 驱动（非 LLPC 编译栈），测试其是否同样消除计算写。

### 18.3.1 结论：D3D12 写存活，GEMV 全链路正确可量化

- **t_d3d12_full.cpp**（最小 compute 写测试）：Dispatch 写 1M floats 后读回，
  out[0..3] = 0.5/1.5/2.5/3.5 精确匹配期望 —— **AMD D3D12 驱动不消除无消费者写**
  （与 LLPC 行为相反）。
- **t_d3d12_gemv.cpp + d3d12_gemv_sk.hlsl**（完整 W4A8 GEMV 链）：
  - 数据：真实布局（packed uint2[8B/块]、scl uint8、act int8[16/块]、asb float）；
  - split-K 并行：每组 256 线程 = 1 行，每线程按 stride-256 循环多块，共享内存树归约；
  - 正确性：M=4096×K=4096 **bad=0/4096**；M=32768×K=4096 bad=1/32768（浮点阈值噪音）；
  - 计时：fence 值必须递增（Signal(1) 重复会导致第 2+ 次假快 ~0.006ms）；
    GPU timestamp resolve 在本机返回 0（放弃，改用 CPU 等待计时）。

### 18.3.2 性能（AMD Radeon 780M，weights+act 读，CPU 等待计时）

| M | K | 时间 | 带宽 GB/s | G MAC/s | 正确性 |
|---|---|---|---|---|---|
| 4096 | 4096 | 0.482 ms | 26.1 | 34.8 | 0/4096 bad |
| 8192 | 8192 | 1.851 ms | 27.2 | 36.3 | 0/8192 bad |
| 32768 | 4096 | 3.375 ms | 29.8 | 39.8 | 1/32768 bad |
| 4096 | 16384 | 1.704 ms | 29.6 | 39.4 | 1/4096 bad |
| 32768 | 16384 | 11.21 ms | 35.9 | 47.9 | 16/32768 bad |

- **纯读上限对照**（d3d12_copy.hlsl，upload 直读）：41-44 GB/s（接近 DRAM 51 GB/s）；
  GEMV 达 26-36 GB/s = 读上限的 **63-87%**，已带宽受限。
- **对比 CPU C 组**（AVX-512 W4A8 多行内核）：42.5 G MAC/s —— iGPU 35-48 G MAC/s
  **同一量级**，且 B 组场景（权重驻留 DRAM）**免 copy 直读**，单段完成。
- 对照 B 组带宽参考（15.1-28.4 GB/s）：**26-36 GB/s 达成并超过上限**。

### 18.3.3 工程教训（D3D12 ctypes/Python 路径放弃）
- Python 3.14 的 ctypes 调 COM 栈参数方法（≥4 个参数）不可靠：
  CreateCommittedResource/CreatePlacedResource 全部 E_INVALIDARG（同参数 C++ 成功）；
  寄存器参数方法（CreateHeap 等）正常。D3D12 验证一律走 C++（WRL）。
- ctypes 的 vtable 读取必须"对象 offset 0 = vtable 指针"再解引用（直接 cast 到
  POINTER(c_void_p) 读到的是对象字段）。
- D3D12 坑位速查：SRV 必须设 Shader4ComponentMapping=0x1688；structured buffer
  SRV stride 必须与 shader 元素匹配（uint2=8B）；cbuffer 需 16 字节对齐
  （root constants 传 4 值）；timestamp query 只用 EndQuery（BeginQuery 非法）；
  fence 每次递增。

### 18.3.4 B 组集成建议（更新 §16.2 结论）
- **B 组 iGPU 内核走 D3D12 计算管线**（HLSL），权重驻留共享内存（upload heap 直读
  41-44 GB/s），单段 GEMV 26-36 GB/s —— 不再依赖 copy 引擎两段式；
- VMA（vk_mem_alloc.h）用于宿主内存管理（权重 bank 分块、激活暂存）；
- 启动校准：CPU W4A8（42.5 G）vs iGPU D3D12（35-48 G）按批大小择优；
- 下一步：把 d3d12_gemv_sk 移植为 Python 可调用的 D3D12 服务（C++ DLL/进程），
  与 cpu_executor 集成做真实推理对比。
## 18.4 FreeToken 集成（--moe-backend igpu，2025-xx 实测）

**目标**：把 §18.3 验证的 D3D12 iGPU GEMV 接入 FreeToken 的 offload 家族，作为
B 组（DRAM 侧）计算的第二个引擎（与 CPU executor 并列），并适配 A/B 组架构
（A 组 dGPU 算 attention/常住专家；B 组 CPU 或 iGPU 算 DRAM 驻留权重）。

### 18.4.1 改动清单（全部已落地，语法/端到端验证通过）

| 文件 | 改动 |
|---|---|
| freetoken/moe/igpu_backend.py | **新增**：IgpuGemvService（常驻服务进程，stdio 二进制协议）、IgpuMoeExecutor（decode：量化激活→服务 GEMV→swiglu→合并）、igpu_available() 探活（启动服务→自检 numpy 参考 → maxerr<1e-2） |
| freetoken/moe/igpu_offload.py | **新增**：IgpuMoeBackend marker（类比 CpuOffloadMoeBackend） |
| freetoken/moe/__init__.py | OFFLOAD_MOE_BACKENDS 加 "igpu"；注册 create_igpu_moe_backend（CLI choices 自动出现） |
| freetoken/moe/offload_cache.py | decode_target 断言加 "igpu"；igpu_executor 槽位 + set_igpu_executor / is_igpu_layer |
| freetoken/engine/config.py | igpu_service（服务 exe 路径）、igpu_fallback（失败回退 cpu）、dense_ffn_engine（稠密 FFN 引擎：cpu|igpu|gpu） |
| freetoken/engine/engine.py | decode_target 决策加 igpu（全层走 igpu）；_init_igpu_executor（探活+自检+回退）；--moe-backend igpu 校验块（cache=2E、prefill overlap）；activation/格式校验对齐 cpu |
| freetoken/server/args.py | --igpu-service、--igpu-no-fallback、--dense-ffn-engine 面板选项 |
| freetoken/layers/moe.py | _decode_routed 加 is_igpu_layer → igpu_executor.decode(...) 分支 |

### 18.4.2 面板选项（新增）

- --moe-backend igpu：MoE decode 专家在 iGPU D3D12 服务上计算（native nvfp4 banks 直读 DRAM，免 PCIe 往返、不占 CPU 核）；
- --igpu-service <path>：服务 exe 路径（默认 repo microbench 或 $FREETOKEN_IGPU_SERVICE）；
- --igpu-no-fallback：服务不可用时**报错**而非回退 --moe-backend cpu；
- --dense-ffn-engine cpu|igpu|gpu：稠密模型（dense_host_offload 架构）FFN 计算引擎选择——B 组候选（cpu 默认、igpu 免 CPU、gpu 全驻留）。

### 18.4.3 判断逻辑（架构适配）

- auto 决策保持 offload/hybrid（不 auto 升 igpu——需要用户显式选择 + ft bench bw profile 支持，下轮加）；
- --moe-backend igpu 显式选择时：格式必须 native nvfp4；activation 集合对齐 CPU executor；服务不可用且 igpu_fallback 时**回退 CPU executor**（保证 boot 可用）；
- decode 分支：is_igpu_layer（全层）→ igpu_executor.decode——与 is_cpu_layer / decode_target=="hybrid" 并列，互斥。

### 18.4.4 验证结果（AMD 780M 实测）

- igpu_available()：True，adapter = "AMD Radeon 780M Graphics"，自检 maxerr=0.0016（服务数学 vs numpy 参考）；
- gate_up 单次投影：服务 vs numpy 参考 **maxerr=3.8e-06**；
- down 单次投影：**maxerr=5.7e-07**；
- 完整 decode（B=2, K=2, H=128）：输出有限、0.09s（含服务 spawn；纯计算 ~ms）；
- 服务协议 in-proc 调用（M=2048 K=4096）：10.9ms/次（stdio 管道主导）；**DLL 化后**（d3d12_gemv.dll，ctypes 指针传递）：**1.79ms/次 → 7.0 GB/s**（vs stdio 快 12×），正确性 rel=1e-7；
- DLL M=4096 K=4096 可复现带宽：**7.0 GB/s（Python 侧）**；纯 dispatch（无 Python/numpy 开销）0.5ms → 26-36 GB/s 带宽墙（见 §18.3）；

### 18.4.5 已知限制与 TODO

1. **e4m3 scale 近似**：服务 HLSL 当前用 0.01 × 位模式 近似权重 block scale（基准约定，服务与 numpy 参考自洽 rel=1e-7）；生产需在 d3d12_gemv_sk.hlsl 加 e4m3 LUT 才与 Triton/GPU 内核数学一致；
2. **act/asb 来源**：IgpuMoeExecutor 运行时用 per-16-block 量化（asb=block scale）；生产 checkpoint 若带 act/asb 字段需在 load 时对接；
3. **批量性能**：逐路由行×双投影调用，DLL 后 Python 侧 7 GB/s；C++ 宿主直调 + "act 按行"扩展协议（大矩阵单次调用）后可达 26-36 GB/s 带宽墙；
4. **ft bench bw 集成**：把 iGPU 带宽（26-36 GB/s）纳入 profile，支持 auto 决策升级 hybrid→igpu-hybrid（B 组 CPU∥iGPU∥PCIe 三路带宽匹配）；
5. **CUDA graph 接线**：IgpuMoeExecutor.decode 当前是 eager Python（D2H→numpy→服务→H2D 需 host sync）；CUDA-graph 捕获需 host-func 节点包服务调用（与 CPU executor 相同的 submit/sync 握手）。
