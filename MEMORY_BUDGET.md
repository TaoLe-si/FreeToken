# 内存/显存总量计算书 — iGPU 共享池方案 vs 现状

## 实测与清单数据来源
- 权重清单 freetoken_weight.json（精确字节）：专家 bank 16.93GB（40 层 x 433.5MB，
  每专家 1.69MB nvfp4 packed）；GPU 常驻 dense 2.00GB（embed 0.97 bf16 + lm_head
  0.27 + attn/GDN/shared/norm）；MTP 头 0.49GB（mtp.safetensors）。
- GDN 状态池：recurrent [40,24,32,128,128] f32 = 80MB/槽，conv 1.25MB/槽，
  24 槽 = 1.90GB。
- KV cache：8 个 full-attn 层，q4_0 约 4.5KB/token → 16384 页 = 0.07GB（本就
  放 WDDM shared）。
- 带宽实测（历史）：主机 DDR5-5600 双通道顺序读 51GB/s（16 线程）；780M 单
  GEMV kernel 0.06ms；v3 server dispatch 0.2-0.5ms；本机 CPU executor 全层
  decode 0.44 t/s（计算墙，排除）。
- 780M 读共享内存带宽：无本地 iGPU，需目标机跑 igpu_bw.py 取实数（按 60-90GB/s
  区间做规划）。

## 场景 A：现状（offload slot-cache 868 槽）
dGPU 显存：
  dense 常驻            2.00 GB
  slot cache 868 槽     1.43 GB (868 x 1.69MB)
  GDN 状态池 24 槽      1.90 GB
  KV (q4_0, 16k 页)     0.07 GB（放 shared，约占 0 VRAM）
  CUDA 图 + 激活        约 1.0-1.3 GB
  合计                  约 6.4-6.7 GB（8GB 卡余 0.6GiB —— 与启动日志吻合）
主机内存：
  专家 bank (pinned)    16.93 GB
  MTP 头                0.49 GB
  Python/运行时/栈      约 2-3 GB
  KV/GDN WDDM 溢出      0-2 GB（驱动管理）
  合计                  约 20-22 GB

## 场景 B：iGPU 共享池方案（目标架构）
dGPU 显存：
  dense 常驻            2.00 GB
  GDN 状态池            1.90 GB
  KV (q4_0, 16k 页)     0.07 GB
  pinned IO 环          约 0.01 GB
  CUDA 图 + 激活        约 1.0 GB
  合计                  约 5.0 GB（slot cache 1.43GB 省掉）
iGPU 显存：
  bank 为 host 指针映射（hipHostRegister / shared heap）→ 约 0 GB 附加
  （仅 shader/命令缓冲，100MB 级）
共享主机内存：
  专家 bank             16.93 GB（同一份 pinned RAM，iGPU 直读，无每步拷贝）
  MTP 头                0.49 GB
  Python/运行时         约 2-3 GB
  合计                  约 20-22 GB

## 数据传输核验（传输不是瓶颈）
- iGPU 直读共享内存：508MB/token（40 层 x 8 路由 x 1.69MB）。60-90GB/s 下
  = 5.6-8.5ms/token —— 这是计算主路径本身（读密集 GEMV），不是额外传输。
- PCIe（dGPU 与 host 间）每步仅路由表+激活+输出 约 0.66MB/token → 20GB/s 下
  33us/token，占比 <1%，非瓶颈。
- CUDA graph 内 D2H/H2D 与 host node 开销已由 CpuMoeExecutor 模式验证可捕获。

## 吞吐预测（叠加 MTP，d1 命中 79%）
- iGPU MoE 每层融合 dispatch：40 x 0.3-0.45ms = 12-18ms/token → 55-85 t/s 纯解码
- MTP K=3 有效（2.8 tok/step，主模型 5-10ms 与 iGPU 并行重叠）→ 目标 40-70 t/s
  （保守下限 vs 用户基线 17 t/s = 2.4x+）

## 结论（总量答案）
- 显存（dGPU 8GB）：方案 B 需约 5.0GB —— 现有 8GB 卡即可，比现状省 1.4GB。
- 内存：两案均约 20-22GB —— 32GB 机器舒适；24GB 机器需压 WDDM 溢出余量并
  限 max_running_req=1-2。
- iGPU 专用显存：约 0 附加（bank 是 host 指针映射，不占 iGPU VRAM）。
- 前提验证（P0，目标机）：igpu_bw.py 实测共享读带宽 + 单层批量 GEMV 原型，
  推算 > 25 t/s 才开工 P1。
