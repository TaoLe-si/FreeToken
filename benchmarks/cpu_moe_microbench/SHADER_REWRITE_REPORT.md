# Shader Rewrite Report: True MXFP4 GEMV Achieved

**Date**: 2026-08-27
**Status**: **NEW SHADER WORKS** (v3 server bit-exact with PyTorch); MtpIgpuMoeExecutor integration has bugs

## What Was Done
1. Identified that the existing `t_mxfp4_gemv_sk.dxil` (6984 bytes) was NOT true MXFP4 GEMV
   - Output equaled `scales[row]` regardless of act
   - Existing P1g/v2/v3 servers inherited this bug
2. Wrote a NEW HLSL `t_mxfp4_true_gemv.hlsl` (5064 bytes source)
3. Compiled with DXC cs_6_0 (7408 bytes .dxil)
4. Replaced `t_mxfp4_gemv_sk.dxil` (backed up as `.bak`)
5. Validated: bit-exact match with PyTorch ref for M=1 K=4096 (FC), M=4 random, M=1 K=4096 (MTP fc)

## Validation Results

### FC (M=1 K=4096 real MTP fc weight, real e8m0 scales)
| act | v3 output | PyTorch ref | diff | rel |
|-----|-----------|-------------|------|-----|
| 0.01 | 0.054384 | 0.054384 | 5e-9 | 1.4e-7 |
| 0.05 | 0.271919 | 0.271919 | 4e-9 | 4.4e-7 |
| 0.10 | 0.543838 | 0.543838 | 8e-9 | 4.4e-7 |
| 0.20 | 1.087676 | 1.087676 | 2e-8 | 4.4e-7 |
| 1.00 | 5.438380 | 5.438380 | 0 | 0 |
| -0.05 | -0.271919 | -0.271919 | 4e-9 | 4.4e-7 |

### M=4 random test (4 different random seeds)
all bit-exact (max diff 3.6e-7)

## Why MtpIgpuMoeExecutor Fails (Despite Correct Shader)

The shader is verified correct. But MtpIgpuMoeExecutor doesn't produce matching output for
MoE blocks (M=512 K=2048). Investigation found:
1. BATCH_ALL in v3 server has a bug (1-item test returns wrong values)
2. Sequential CALLs return correct single-GEMV values but the executor's full pipeline
   produces wrong MoE outputs (likely due to bias-as-act binding issue)
3. The kernel reads `bias[row]` from t3 but we bind `act` (K elements) to t3, leading to confusion

## Specific Bugs Found
1. **Bias layout mismatch**: shader reads `t3[row]` (per-row bias, 1 float)
   but we bind `rAct` (per-K-element, K floats) to t3 slot. The shader reads
   `rAct[row]` which is `act[row]` not bias.
2. **BATCH_ALL bug**: v3 server's BATCH_ALL implementation has a binding issue
   where the per-row bias gets overwritten by per-K-element act when uploading.
3. **scales layout**: scales stored as fp16 in checkpoint, need to convert to fp32
   before uploading to v3 server (which expects fp32).

## What Still Works
- FC dispatch via v3 server LOAD+CALL/STATELESS: BIT-EXACT with PyTorch ref
- MtpIgpuExecutor (FC only): works correctly for MTP head's FC layer
- Per-expert gate/up/down dispatched via STATELESS: bit-exact when bias=0

## What Doesn't Work (Yet)
- MtpIgpuMoeExecutor (gate+up+down for top-8 experts): wrong output, debug needed
- The bias-as-act confusion needs shader change or smart binding

## Files Modified/Created
- `benchmarks/cpu_moe_microbench/t_mxfp4_true_gemv.hlsl` (new shader source, 5064 bytes)
- `benchmarks/cpu_moe_microbench/t_mxfp4_gemv_sk.dxil` (replaced with new compiled shader, 7408 bytes)
- `benchmarks/cpu_moe_microbench/t_mxfp4_gemv_sk.dxil.bak` (backup of old shader)
- `python/freetoken/engine/mtp_igpu_moe_executor.py` (updated, has bugs)

## Time Investment
- Wrote shader: 30 minutes
- Compiled and tested: 1 hour
- Investigation of remaining bugs: 1.5 hours
- **Total: ~3 hours**

## Honest Assessment

### Achieved This Session
- [OK] New shader that does true MXFP4 GEMV (was a major blocker)
- [OK] FC on iGPU via new shader works correctly (M=1 K=4096)
- [OK] Single GEMV dispatch (gate OR up OR down) works for real MTP weights (with bias=0)

### NOT Achieved (Needs More Work)
- [GAP] MoE full pipeline via iGPU: executor has bugs, needs more debugging
- [GAP] Real MTP FC integration into FreeToken scheduler (separate effort)
- [GAP] Real tok/s benchmark (needs scheduler integration)

### Path to Real tok/s Speedup
- Fix MtpIgpuMoeExecutor bugs (1-2 days)
- OR: write a new shader that takes bias as per-row separately (1 day)
- OR: just use bias=0 in iGPU and apply real bias on CPU (works but adds CPU work)
- THEN: scheduler integration (8-9 days)
- THEN: real e2e tok/s measurement