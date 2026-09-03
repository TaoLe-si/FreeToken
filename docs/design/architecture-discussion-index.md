# FreeToken 架构讨论索引 (2026-09-03 session)

## 讨论产出

| # | 文件 | 内容 |
|---|---|---|
| 1 | `docs/design/igpu-pcie-async-refactor.md` | iGPU PCIe 异步重构原始设计 |
| 2 | `docs/design/cpu-managed-expert-cache.md` | CPU-managed expert cache 架构提案 |
| 3 | `docs/design/phase1.5-validation-results.md` | Phase 1.5 验证结果与失败分析 |
| 4 | `docs/design/architecture-discussion-index.md` | 本文档 |

## 关键架构结论

### A. 带宽层级 (本机)

| 介质 | 带宽 | 备注 |
|---|---|---|
| dGPU VRAM | 256 GB/s | RTX 4070 GDDR6 |
| CPU ↔ DDR5 | 89 GB/s | 双通道 + cache |
| iGPU GTT | 26 GB/s | APU coherent |
| PCIe 3.0 x16 | 6-12 GB/s real | dGPU↔RAM |

### B. 50 tok/s 的物理需求

- top-8 expert 读: 320 MB/token × 50 = **16 GB/s** 权重带宽
- 只有 VRAM (256 GB/s), CPU RAM (89 GB/s), iGPU GTT (26 GB/s) 可达
- PCIe (6-12 GB/s) 不够

### C. iGPU 路径的真实瓶颈

- iGPU compute: 7.4 TFLOPS (FP16), 利用率仅 2% (160 GFLOPS 需求)
- iGPU GTT: 26 GB/s, 实际读 41 MB/token = 1.6 ms (fast)
- **真正瓶颈: kernel launch overhead, 120 launches × 400 µs = 48 ms/token (96%)**

### D. Phase 1 改动

1. `hip_moe_dll.hip`: 删除 `hipStreamSynchronize`, 添加 `igpu_get_stream` 导出
2. `igpu_shared_executor.py`: decode() 改为 pinned staging + 单次 sync
3. **集成验证失败**: iGPU MoE 路径 `hipMemcpy H2D rc=1`

### E. 历史数据

| 路径 | tok/s | 来源 |
|---|---|---|
| 用户实测 `--moe-backend=igpu` | 7 | per-layer sync + iGPU compute |
| CPU fallback | ~3 | iGPU H2D 失败 |
| Phase 0 isolated | 16.5 | _igpu_phase0.py |
| Phase 1 isolated | 22.4 | _phase1_test.py |
| Phase 1 e2e | ? | 无法验证, H2D 失败 |
| Phase 2 目标 | 40-50 | kernel 融合 |

## 待解决问题优先级

1. **P0**: 修复 `hipMalloc returns unusable low VAs` 问题
   - 选项 A: `hipHostRegister` + reuse pinned host memory (零额外分配)
   - 选项 B: 完全跳过 GTT, zero-copy kernel 读 pinned host
2. **P0**: 修复 `FT_IGPU_RESERVE=1` OOM 问题 (与 P0 同源)
3. **P1**: 验证 Phase 1 集成达到 22 tok/s
4. **P1**: Phase 2 kernel 融合 (3 → 1 kernel per layer)
5. **P2**: CPU-side async prefetch (cache top-K experts on dGPU)
6. **P2**: CUDA Graph capture (替换 launch overhead)

## 用户洞察 (10 条关键)

记录于 `phase1.5-validation-results.md` 末尾

## 下次会话入口

1. 阅读 `phase1.5-validation-results.md`
2. 决定: 先解决 hipMalloc unusable address 问题 (P0)
3. 选项 A vs B 选型
4. 实施 + 测试