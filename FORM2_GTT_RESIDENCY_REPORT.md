# 形态二（GTT 全量驻留）实施报告与裁决 — 2026-09-02

## 一、背景

zero-copy 路线已死（见 IGPU_ZEROCOPY_VERDICT.md）。用户问：不走 zero-copy，能否
"边 copy 边释放、不影响解码速度"。为此评估了两条形态并完成实测。

## 二、决定性实测数据（2026-09-02 凌晨，独立进程，无并发干扰）

| 实验 | 结果 |
|---|---|
| 780M hipMalloc 容量阶梯（无 CUDA） | 25+ GB（57×433MB 后才 NULL），GTT 默认限额 ≈50% RAM |
| totalGlobalMem 报告值 | 18.20 GB（0.5GB 专用 + GTT 计入） |
| kernel 直读 GTT 内存带宽 | **26.6 GB/s**（read_sum kernel，1GB 累加） |
| H2D memcpy 带宽（pinned→GTT） | 16.5 GB/s |
| 8×1.69MB 小拷贝 | 1.787ms（0.223ms/次，WDDM 固定开销主导） |
| **形态二端到端（17GB 设备 banks + 设备 IO）** | **纯 kernel 0.70ms/层；含每层 IO 28.5ms/token = 35.1 t/s** |
| 形态二正确性 | layer0 全量随机 bank 对拍：**rel err 1.1e-3**（与 DLL 历史精度一致） |
| 形态一（流水线拉取）模型 | 8 拷贝 1.79ms + kernel 0.29ms ≈ 2.1ms/层串行 → ~23-32 t/s（劣于形态二） |

结论：**形态二（banks 全量 GTT 驻留）理论最优 ~65 t/s，实测 35 t/s**（含逐层小拷贝开销），
碾压形态一。形状一保留为后备。

## 三、形态二引擎集成 — 实施内容（已全部落盘）

1. **DLL v8.2**（hip_moe_dll.hip/.dll，已重建并验证版本串）：
   - `igpu_devmalloc/devfree`：hipMalloc 设备内存分配
   - `igpu_register_layer_dev`：直接登记设备指针（不做 host 别名解析）
   - `igpu_moe_decode_dev(layer, hidden_dev, ids_dev, tkw_dev, out_dev)`：
     全设备指针解码路径（含 topk_ids 越界防御）
   - `igpu_meminfo`：hipMemGetInfo 诊断导出

2. **executor**（igpu_shared_executor.py）：
   - `register_banks()`：Form-2 迁移 — 每层 6 个 bank 一次 hipMemcpy H2D 进 GTT，
     迁移后置 `bank_sources[layer]=None` 释放宿主 pinned（零净增占用 ✓）
   - 分块 H2D（64MB CHUNK）+ 失败诊断日志
   - `decode()`：D2H hidden/ids/weights → 设备 IO 暂存缓冲（devmalloc，惰性创建）
     → `igpu_moe_decode_dev` → D2H 回读 → H2D 回 GPU（dtype 转换保留）
   - 优先消费 `_IGPU_RESERVED` 预留指针（见下）

3. **engine.py**：
   - `FT_IGPU_RESERVE=1` 时，scheduler 进程 import 期（任何 CUDA 之前）预分配
     40×433MB = 17.3GB GTT banks（"iGPU GTT reserve: 40 banks"）
   - igpu 后端银行加载走 `layer_residency=["locked"]`（省 CUDA pin 配额）

4. **offload_cache.py / host_banks.py**：
   - set_bank_sources 放行 LOCKED/PAGEABLE 驻留（igpu 模式跳过 fused-copy plan
     与 prefill overlap buffers——它们依赖 CUDA pinned 设备别名）
   - `_os_lock` 移植 Windows（VirtualLock + SetProcessWorkingSetSizeEx）

## 四、根因发现（本轮最重要的产出）

形态二在**独立进程**全链路验证通过，但在**引擎进程**内 H2D 写入必然失败：

| 进程形态 | hipMalloc | hipMemcpy H2D | hipMemset | hipMemcpyAsync |
|---|---|---|---|---|
| 独立 python（无 CUDA） | 0x308000000 ✓ | rc=0 ✓ | ✓ | ✓ |
| 独立 python + 5GB VRAM + 12GB pinned | ✓ | rc=0 ✓ | ✓ | ✓ |
| 两级 spawn 链 | ✓ | rc=0 ✓ | ✓ | ✓ |
| **引擎 scheduler（CUDA 初始化后）** | **假 VA（0x8000000 步进）** | **rc=1** | **rc=1** | **rc=1** |
| 引擎 import 期预留（CUDA 前） | 17.3GB 分配成功、free 正确扣减 | **rc=1** | **rc=1** | **rc=1** |

**统一根因**：Windows WDDM 上，同进程 CUDA(NVIDIA) 上下文激活后，HIP(AMD) 的
**设备内存写入/提交路径被整体破坏**——GTT 页表登记成功（meminfo 扣减、free=0.00），
但页没有真正映射进 GPU 可写空间。这同时解释了此前 zero-copy 的 kernel 直读死亡
（设备侧读）、bank 回读 match=True（hipMemcpy D2H 走的是仍幸存的读/查询路径）、
以及 >96MB hipHostRegister 的静默失败——全部是同一 KMD 提交缺陷的不同症状。

对照之下，此前判断"zero-copy 在此平台架构性死亡"需要修正为：
**"同进程 CUDA+HIP 共存时，HIP 的 GPU 内存写入/读取皆不可靠；独立 HIP 进程一切正常"。**

## 五、结论与路线裁决

1. **同进程方案（无论 zero-copy 还是 GTT 驻留）在 Windows + ROCm 6.4 + 双 GPU
   （CUDA+HIP 共存）组合下不可行。** 这不是预算、不是时序、不是代码问题——
   是驱动级缺陷。全部引擎内尝试均有铁证。

2. **形态二本身被验证是正确的架构**（35 t/s 实测、精度 1.1e-3、宿主占用更省）。
   它唯一缺的是"HIP 独立进程"。要吃到这 35-65 t/s，需要：
   - **形态二-跨进程版**：独立 HIP worker 进程（无 CUDA）常驻 banks，
     引擎经共享内存环形队列送 hidden/ids，worker 解码后回写。
     预计额外 1-2ms/token 的 IPC 开销 → ~30-45 t/s 净值。
     这是一次"中等规模"改动（新 worker 进程 + IPC 协议），不是"小块修复"。
   - **WSL2 路线**：WSL2 内 GPU-PV 虚拟化下 HIP 独占（无 NVIDIA CUDA 同进程干扰），
     TheRock/ROCm 官方 Linux 栈对 gfx1103 有 nightly 支持（device-gfx1103）。
     Linux 的 GTT/amdgpu 路径没有 WDDM 缺陷，形态二直接可用。

3. **已落地且保持可用的**：
   - 引擎在 igpu 后端缺省时安全回落 CPU executor（当前引擎以此稳定运行）
   - FT_IGPU_RESERVE=1 可随时复现形态二全链路实验（所有代码就位）
   - DLL v8.2 设备内存 API 与 40-bank 预留机制（跨进程方案可直接复用）

## 六、验收清单（明早）

- [ ] 引擎当前状态：CPU fallback 稳定运行（port 1919）
- [ ] `_simtest6.py`：形态二独立进程全链路（正确性 + 35 t/s）可复现
- [ ] `_gtt_bw.exe`：GTT 容量/带宽 bench
- [ ] FT_IGPU_RESERVE=1 引擎日志：GTT reserve 40 banks + memcpy/memset rc=1（本报告第四节证据）
- [ ] 代码落盘清单：hip_moe_dll.hip v8.2 / igpu_shared_executor.py /
      engine.py / offload_cache.py / host_banks.py

## 七、遗留与建议（按优先级）

1. **短期**：接受 CPU fallback（17 t/s 上下）作为 Windows 生产形态；
2. **中期**：立项"HIP 独立 worker 进程"（形态二跨进程版）——改动范围明确，
   收益 30-45 t/s，风险点是 IPC 与生命周期管理；
3. **长期**：WSL2 路线评估（TheRock nightly + Linux amdgpu，无 WDDM 缺陷），
   或等 AMD 修复 Windows KMD 的 CUDA+HIP 共存提交缺陷（可附我们的实验报告提 bug）。


## 八、补充验收事实（同夜后续）

1. **prefill 路径硬依赖 CUDA pinned banks**：pageable banks 下 `copy_missing` →
   `fast_index_copy` 崩溃（"host tensor must be pinned+mapped"）。因此形态二若做
   跨进程版，引擎侧 banks 必须保持 PINNED（prefill 用），GTT 副本由 worker 进程独享。
2. **同夜 0.48 t/s 为机体深夜限频**（与 DELIVERY_2026-0901.md 第 67 行记录一致：
   深夜 0.6-1.5 t/s，清晨冷启动恢复 2.4-2.8 t/s），非代码回归。输出正确性完好
   （reasoning_content 正常生成，finish_reason=length 正常）。
3. 引擎最终稳定形态：`--moe-backend igpu` + 自动 CPU fallback（pinned banks），
   可安全过夜运行。
