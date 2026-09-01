# 780M HIP 共享内存带宽实测报告（P0 完成，2026-09-01）

## 测试环境与路径
- 设备：AMD Radeon 780M Graphics，gfx1103，6 CU，总显存 19.5GB（共享），canMapHostMemory=1
- 工具链：ROCm 6.4 HIP SDK（C:/Program Files/AMD/ROCm/6.4），hipcc --offload-arch=gfx1103
- 按用户决策采用 **HIP 路径**（放弃 D3D12/OpenCL；OpenCL 探测不到设备）
- 基准源码：benchmarks/cpu_moe_microbench/hip_shared_bw.hip（hipEvent 计时，
  checksum 校验内核真实执行，未做写入；纯读）

## 实测结果（best of 10，hipEvent）
| 测试 | 带宽 |
|---|---|
| A. hipMalloc（iGPU 本地）1GB 顺序读 | 32.9-34.4 GB/s |
| B. hipHostMalloc pinned 1GB 零拷贝顺序读 | **34.9 GB/s** |
| C. MoE token 模式：320 x 1.61MB 随机块读（515MB/token） | **35.2-38.6 GB/s（中位 37.4）** |

关键结论：
1. **零拷贝 pinned 读 = iGPU 本地读**（34.9 vs 33 GB/s）—— APU 统一内存下，
   共享池方案没有「远端惩罚」，bank 放主机内存性能等同于放 iGPU 显存。
2. 共享 DDR5 读墙约 **35-38 GB/s**（低于此前 60-90 的规划假设；与 CPU 侧
   16 线程实测 51GB/s 同源，iGPU 争用同一条双通道 DDR5-5600）。
3. MoE 随机块模式（真实路由形态）与顺序读几乎同速 → 1.61MB 块足够大，
   TLB/页粒度不是问题。

## 对吞吐预测的修正
每 token 专家读取量 515MB（40 层 x 8 路由 x 1.61MB）：
- 纯 MoE 读下限：515MB / 37GB/s ≈ **14.5ms/token → 69 t/s（MoE-only 上限）**
- 对照现状：slot-cache 路径每 token 有效 PCIe 搬运 ~2.6GB（868 槽池 miss 率
  下反复换入换出）→ 本机 OFF 基线 4.11 t/s 的主要成分。iGPU 共享池把这部分
  从 ~200ms 级压到 14.5ms（**MoE 环节 ~13x**）。
- 本机整步预测（dGPU dense/GDN ~10ms 与 iGPU MoE 14.5ms 重叠）：
  - 无 MTP：约 25ms/token → **~40 t/s**（本机基线 4.11 的 ~10x）
  - MTP K=3（B=4 行 verify，2.8 tok/step）：MoE 读 4x515MB/37GBps ≈ 59ms/step
    → **~47 t/s**
- 用户机（17 t/s 基线，主模型更快）按比例折算：MTP 后预计 60-100+ t/s。
  **门槛 >25 t/s：通过（余量 2x）。**

## 内存/显存总量（修正后结论不变，数字校准）
- dGPU 显存：~5.0GB（dense 2.0 + GDN 池 1.9 + KV 0.07 + 图/激活 ~1.0），
  8GB 卡可行，slot cache 1.43GB 省掉。
- 主机内存：20-22GB（bank 16.93 + MTP 头 0.49 + 运行时 2-3）。
- iGPU 专用显存：~0 附加（pinned 零拷贝映射，实测无性能差）。

## 传输瓶颈核验（满足「传输不是瓶颈」约束）
- iGPU 读共享内存 515MB/token **就是计算本身**（读密集 GEMV），非额外搬运；
  37GB/s 实测下占 14.5ms，其余 PCIe 流量 0.66MB/token（<1%）。
- 剩余瓶颈是 iGPU 侧 GEMV 计算与 kernel launch（下一阶段 P1 用层内融合
  kernel 摊销；HIP 免 D3D12 runtime，launch 开销更低）。

## P0 判定：**GO** —— 进入 P1（HIP BATCH_LAYER 协议 + gfx1103 融合 GEMV shader
+ CUDA-graph host-node 桥）。
