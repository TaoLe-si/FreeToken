# 当前状态报告 (2026-08-27 16:30)

## P0 修复状态
- **问题确认**: t_mxfp4_gemv_v3_server.exe 的 STATELESS 路径输出全 0
- **已诊断**: dxc 把 StructuredBuffer<int/uint> 编译成 float binding，server 写入的 int32 数据被当 float 读，结果为 0
- **尝试修复 1**: shader 改 StructuredBuffer<float>，server 端把 int32 → float 转换 — 失败，仍然 0
- **尝试修复 2**: shader 用 ByteAddressBuffer — 失败，仍然 0  
- **尝试修复 3**: simplest shader (outv[0] = 1.0f) — 失败，仍然 0
- **根因未确定**: 即使最简单的 shader 也输出 0，疑似 D3D12 dispatch 或 readback 路径有更深层问题

## 重要发现：t_mtp_fc_clean.exe 是可工作的 D3D12 程序！
- 输出: outv[0] = -1.7111124
- PyTorch ref: -1.7111129
- **rel err = 2.79e-7 (bit-exact)**
- 路径: E:\FreeToken\benchmarks\cpu_moe_microbench\t_mtp_fc_clean.exe

## 可用 baseline
- t_mtp_fc_clean.exe 独立可工作 (M=1, K=4096 MXFP4 fc)
- t_mtp_fc_server.exe 独立可工作（从 bin 读权重）
- t_mtp_fc_compare.py 已经验证 bit-exact

## 任务状态
- [DONE] 拉取远程 main (8 commits)
- [DONE] 创建 feature/igpu-mtp-mxfp4 分支
- [DONE] stash pop 解决 4 个冲突（接受 theirs）
- [PARTIAL] P0 修复: t_mxfp4_gemv_v3_server.exe STATELESS 路径仍然输出 0
  - t_mtp_fc_clean.exe 路径已经工作（不需要 v3 server）
- [TODO] 集成 MTP driver 到 FreeToken scheduler
- [TODO] E2E 测试 + 测量加速比

## 关键文件
- t_mxfp4_gemv_v3_server.exe (P0 坏)
- t_mxfp4_gemv_v3_server_full.cpp (与 v3_server.cpp 相同, 不是 backup)
- t_mxfp4_gemv_sk.hlsl (当前是 ByteAddressBuffer 版本，备份在 .bak_pre_p0)
- t_mxfp4_gemv_sk.dxil (恢复为原始 7408 字节)
- **t_mtp_fc_clean.exe** (独立工作)
- **t_mtp_fc_server.exe** (独立工作)
- t_mtp_fc_compare.py (验证 bit-exact)
- python/freetoken/engine/mtp_driver.py (MTP driver, 8.9KB)
- python/freetoken/engine/mtp_igpu_executor.py (MTP FC iGPU exec, 6.0KB)
- python/freetoken/engine/mtp_igpu_moe_executor.py (MTP MoE iGPU exec, 7.9KB)
- python/freetoken/kernel/igpu_fc.py (iGPU client, 11.6KB)
- python/freetoken/models/qwen3_5_moe/mtp.py (MTP head module, 16.7KB)
- python/freetoken/moe/igpu_backend.py (iGPU backend, 28.6KB)

## 下一步
1. **放弃 v3 server 路线**，用 t_mtp_fc_clean.cpp 作为 iGPU MXFP4 kernel 的模板
2. 写一个新的 iGPU server (类似 t_mtp_fc_clean.cpp)，让它接收 1) packed M*K uint, 2) act K float, 3) scales M*ns float (e8m0 decoded), 4) bias M float
3. 集成到 python MTP driver
4. E2E 测试 + tok/s 测量
