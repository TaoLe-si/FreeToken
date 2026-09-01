# iGPU 共享池 MoE — 阶段总览与状态报告（2026-09-01）

## 路线：HIP 路径（用户拍板，放弃 D3D12/OpenCL）

| 阶段 | 内容 | 验收标准 | 状态 |
|---|---|---|---|
| P0 判定 | HIP 带宽实测（hip_shared_bw.hip） | 投影 >25 t/s | ✅ 完成（3c9e278）：中位 37.4 GB/s → MoE-only 69 t/s，GO |
| P1a GEMV 内核 | gfx1103 HIP NVFP4 GEMV：8 专家 gate_up+silu+down，零拷贝读 pinned bank | 数值对拍 max rel err < 3e-2（vs fp32 参考）；实测单层延迟 → 投影整步 | ⏳ 进行中（子代理） |
| P1b 常驻 server | HIP server 进程：BATCH_LAYER 协议（一次提交=一层全部路由+激活）；hipHostRegister 注册 freetoken host banks；done/ready flag 页（供 CUDA host node 轮询） | server 往返延迟 < 1ms/层（含提交）；bank 注册零拷贝生效 | 待 P1a |
| P1c 层内融合 | 8 专家 gate_up+down 融合单 kernel/单提交，launch 开销摊销 | 每层总耗时 ≤ 读带宽下限 1.36ms × 1.15 | 待 P1b |
| P2 engine 集成 | IgpuSharedMoeExecutor（对齐 CpuMoeExecutor 的 decode_submit/decode_sync + flag-sync 图桥）；pinned IO 每层 x 2 乒乓；moe.py dispatch 接入；decode_target=igpu；replay/verify-graph 门控按 executor 能力放行 | eager 前向数值正确；CUDA graph 捕获成功；replay 输出与 eager 位级一致 | 待 P1c |
| P3 验收 | planets/france 逐字 == MTP-OFF；200-token 单次基准；state-hash 位级对比；文档+推送 | 等价性通过 且 >25 t/s（目标 40+）；否则回退门控 | 待 P2 |

## 关键实测基准（P0，已定案）
- 780M gfx1103，ROCm 6.4 HIP：pinned 零拷贝读 = 本地读（34.9 vs 33 GB/s）
- MoE token 形态（320 x 1.61MB 随机块，515MB/token）：中位 37.4 GB/s
- → 每 token 专家读取下限 14.5ms（69 t/s）；>25 t/s 门槛 2x 余量

## 风险与回退
- 数值对拍不过 → 检查 scale/bias 公式顺序（以仓库 nvfp4_backends 为准），必要时 P1a 改用合成 bank 先锁性能、真权重对拍推迟到 P1b。
- server 往返超预算 → 把 BATCH_LAYER 升级为整步提交（40 层流水在 server 内串行，done 页一次翻转）。
- 图捕获失败 → host node 改轮询 done 页的 busy-wait 变体（仍图安全，参考 ModelVerifyGraphBackend pinned 重放模式）。
- P3 不达标 → decode_target 门控回退（replay 仅非 MTP），保留 executor 供迭代。

## 分派
- P1a 已派发子代理（独立、自包含：内核 + 数值对拍 + 性能投影）。
- P1b/P1c 依赖 P1a 内核接口；P2 依赖 P1b 协议；P3 依赖 P2。
