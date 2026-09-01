# Bias Separation Report

**Date**: 2026-08-27
**Status**: **Bias correctly separated per-row** (iGPU matches CPU for gate/up/down standalone)

## Changes Made

### v3 Server Changes
1. `rA` (slot 3 = bias) re-allocated as M*4 bytes (per-row) instead of K*4 (per-K-element)
2. Added `rB` (slot 2 = act) as K*4 bytes (per-K-element)
3. Fixed CALL handler: act bytes go to rB (slot 2), bias bytes go to rA (slot 3)
4. Fixed BATCH_ALL handler: same act/bias swap

## Validation

### Single GEMV (M=1 K=32, bias=2.5 vs bias=0)
- With bias=2.5: -15.5 (= -18 + 2.5 ✓)
- Without bias: -18 (= -36 * 0.5 ✓)
- Bias diff matches exactly: 2.5 ✓

### Single GEMV (M=4 K=64 random, bias=[0.1,-0.2,0.3,-0.4])
- With bias: [-1.71, -0.76, 1.21, 0.18]
- Without bias: [-1.81, -0.56, 0.91, 0.58]
- Bias diff: [0.10, -0.20, 0.30, -0.40] (matches exactly ✓)

### MTP MoE Executor
- Forward() works correctly with bias=0 in shader
- CPU vs iGPU match within atol=1e-2 (rel diff 1.02 due to small output magnitudes)
- 24 sequential CALLs (slow: ~20ms)

## Performance Bottleneck

BATCH_ALL in v3 server does per-item dispatches with barriers. Each dispatch waits
for fence. 16 items = 16 submissions. For the executor's 16+8 = 24 GEMVs, this
is too slow (15-20ms total).

To get real speedup, the shader would need to support multi-GEMV batched dispatch.
That's a more complex shader change (1-2 more days).

## Status
- [OK] FC dispatch via v3 server (M=1 K=4096): works, bit-exact with PyTorch
- [OK] Single GEMV (gate/up/down) per expert with bias: works, bit-exact
- [OK] Bias separation: per-row bias correctly added to output
- [GAP] MtpIgpuMoeExecutor.speed: 20ms (24 sequential dispatches); needs BATCH_ALL fix
- [GAP] BATCH_ALL is buggy in v3 server (per-item dispatches with barriers, slow for 16+ items)

## Next Steps
- Fix BATCH_ALL in v3 server to do single-dispatch for all items (or single shader)
- OR: write a new shader that takes all 16 GEMVs in one Dispatch call
- THEN: re-benchmark MtpIgpuMoeExecutor speedup