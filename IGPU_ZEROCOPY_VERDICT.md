# iGPU Zero-Copy 判死报告 (P2.5)

## 结论
本机 (RTX 4070 Laptop + 780M gfx1103, ROCm 6.4 Windows, CUDA+HIP 同进程) 上,
**HIP 零拷贝 (hipHostMalloc/hipHostRegister + hipHostGetDevicePointer) 的 kernel 直读不可用**。

## 铁证链
1. DLL 内核数值正确性: numpy 逐位对拍 max rel err 4.8e-4 (随机 NVFP4 全流程)
2. bank 迁移 hipHostMalloc 后宿主/别名 hipMemcpy 读回一致 (match=True)
3. **kernel 视角**: probe kernel 读 hidden → 全 0; 读 bank → 全 0 (宿主同地址 82 71 42 44 非零)
4. hipHostGetDevicePointer 返回值 == host 地址 (伪别名; 正常应为独立设备地址)
5. hipMemcpy API 读通但 kernel 读不通 → 页未进入 GPU 页表
6. 独立进程复现: 时好时坏 (非确定性); 引擎环境 100% 失败
7. hipHostRegister >96MB CUDA-pinned 页: 静默返回 0 且读全 0 (尺寸相关失效)

## 症状链回放
输出重复 ("here, here...") ← MoE 输出全 0 ← kernel 读不到 host 页 ← HIP 映射失效

## 引擎修复沉淀 (已完成)
- app.py: --moe-backend igpu 不再被强转 hybrid
- graph.py: _capture_graphs 尊重 FT_SKIP_CUDA_GRAPH
- igpu_shared_executor: bank 迁移式替换 (稳态零净增), IO 共享内存化, dtype 修复
- hip_moe_dll v5-v7: hostmalloc/hostfree 导出, 三层诊断 (文件日志/内核 probe/bank probe)
- _io_for 返回 bug, 输出 dtype cast 修复

## 路线判定
- A. iGPU 显存槽位 (memcpy 拉取 + LRU): Bug A 重现 + 8GB UMA < 17GB → 不可行
- B. 图外分段 (主干 replay + MoE 图外每步 memcpy 拉取): 可行, 30-45 t/s 上限, 中等工程量
- C. CPU executor: 0.44 t/s 已否决
- D. 回 slot cache 修 Bug A: VRAM 不足, 回到原点

下一步等用户决策 (B 或其他)。
