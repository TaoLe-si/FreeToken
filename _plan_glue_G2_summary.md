# 明早验收 - MTP 优化 + C++ 胶水层骨架 (2026-08-29 23:30)

## 改动文件清单

1. **E:/FreeToken/python/freetoken/attention/linear.py**
   - FLAMetadata dataclass 新增 4 个 fields (G.2 persistent buffers):
     - mtp_verify_cu_seqlens_varlen: torch.Tensor | None
     - mtp_verify_has_initial_state: torch.Tensor | None
     - mtp_verify_snap_host_slots: list[int] | None
     - mtp_verify_live_slot: int

2. **E:/FreeToken/python/freetoken/scheduler/scheduler.py**
   - _prepare_batch L1671-1700 改为 promote 4 个 persistent fields 到 FLAMetadata
   - snap_host_slots 用 list(slot_list) 防御 copy (避免 _mtp_release_gdn_snap mutation)
   - live_slot 从 batch.reqs[0] 拿 (hybrid-radix linear_slot_idx 优先)

3. **E:/FreeToken/python/freetoken/models/qwen3_5_moe/gdn.py**
   - _forward_mtp_verify 重写使用 persistent fields
   - 移除 2 次 torch.tensor(... device=...) 分配 (生产路径下)
   - 移除 2+1 次 .item() syncs (live_slot + 每 step dst_slot)
   - 保留 fallback 分支给 direct-op callers (tests)

4. **E:/FreeToken/python/freetoken/kernel/csrc/glue/igpu_service.h** (新增)
5. **E:/FreeToken/python/freetoken/kernel/csrc/glue/igpu_service.cpp** (新增)
6. **E:/FreeToken/python/freetoken/kernel/csrc/glue/pybind_module.cpp** (新增)
7. **E:/FreeToken/setup.py**
   - 新增 CppExtension "freetoken.kernel._freetoken_igpu" (IgpuService + pybind11 module)
   - 修复 _cuda_runtime_paths() 加 v13 lib/x64 路径 (cudart.lib)

## 编译产物

- _freetoken_igpu.cp312-win_amd64.pyd (259 KB) - 在 E:/FreeToken/python/freetoken/kernel/
- FreeToken.exe (18.5 MB, mtime 2026-08-29 23:26:00) - 在 E:/FreeToken/dist/
- 备份: FreeToken_preG2glue_1788017143067.exe (17.5MB, 旧 C.4+ C.7 baseline)

## 验证状态

- ✅ Python syntax check OK (gdn.py + linear.py + scheduler.py)
- ✅ C++ module 编译成功 (BUILD_EXIT=0)
- ✅ C++ module import OK (version 0.1.0, IgpuService class registered)
- ✅ PyInstaller pack 成功 (FreeToken.exe 18.5MB)
- ⏸️ Runtime test: 用户自己开 daemon 测 (panel setting sMtp=true, sMtpIgpuFc=true)

## 预期效果 (基于设计分析)

| 路径 | 改动 | 预期加速 |
|---|---|---|
| G.2 sync 消除 | 移除 .item() + torch.tensor() 在生产路径 | verify forward Python overhead -50% |
| G.2 进 graph (后续) | persistent buffers 让 CUDA graph capture 可行 | verify forward kernel launch -90% (5ms vs 40ms) |
| B.1 MTP_LAYER (后续) | iGPU D3D12 整层 fused | MTP step -40% (50ms → 30ms) |

今晚仅交付 **G.2 sync 消除** + **胶水层编译骨架**。完整 ROI 需要后续 B/E/G 线 PR 落地。

## 后续待办 (Phase 2)

1. **IgpuService 真实实现** (Windows CreateProcessW + overlapped pipe)
2. **GdnDispatcher C++ 类** (24 层 GDN forward, 3 paths)
3. **MtpHead C++ 类** (单步 fused forward_with_state)
4. **GDN verify path 进 CUDA graph** (利用 G.2 持久化 buffer)
5. **MoE kernel port 到 D3D12 HLSL** (E.1 调研结论: 256 routed top-8, 512MB sticky weights)
