# Project Status: ABANDONED at [P1b] numerical correctness check

## Timeline

[P0] ✓ MTP head weights extracted, real shape confirmed.
  - hidden=2048, vocab=248320, 1 MTP layer
  - MXFP4 format: 4-bit e2m1 weights + bf16 scale + bf16 bias per 32-element micro-block (NOT pure e8m0)
  - Total MTP head weight: ~405 MB (~1.8% of main 21.5 GB)

[P1a] ✓ MXFP4 GEMV kernel on AMD Radeon 780M D3D12
  - HLSL kernel + host C++ + build scripts + Python reference
  - Synthetic random data test (M=2048, K=4096, 100 iters):
    * Latency p50: **0.30 ms / iter**
    * Throughput: **55 GFLOPs** (vs theoretical 780M W4A8 ~7 TFLOPs, ~0.8% utilization)
    * Numerical accuracy: max rel diff **3.7e-4** vs PyTorch (within FP accumulation noise)
  - Single-row sanity test (M=1, K=32, all weights=acts=1, scale=1, bias=0): exact match 32

[P1b] ✗ NVFP4 GEMV with REAL model weights — NUMERICAL CORRECTNESS UNVERIFIED
  - Format on real weights: bf16 scale (~0.002) + bf16 bias (~0.003) per micro-block
  - Random act + real weights: D3D12 output vs PyTorch ref — fail (diff ~constant factor, root cause not isolated)
  - Single-row M=1 K=32 test: D3D12 = -3.93 expected -27 (debug override scale=1, bias=0)
  - Single-row M=1 K=32 test: D3D12 = -3.5M expected -0.073 (real bias/scale)
  - Root cause: not isolated. Suspected causes:
    * bf16 precision loss in scale/bias multiplication
    * nibble order / byte ordering discrepancy between Python and HLSL unpack
    * constant-factor error in D3D12 reduce or NVFP4 formula application
  - Time spent: many iterations, unable to isolate root cause

## Phase status

[P0] ✓ DONE
[P1a] ✓ DONE (synthetic data verified)
[P1b] ✗ FAIL — numerical correctness not verified
[Phase C: scheduler integration] — NOT STARTED (depended on [P1b])
[Final: tok/s speedup measurement] — NOT REACHABLE

## Final decision per user's rule ("失败即放弃项目")

Project is **ABANDONED** at [P1b] stage.

The core issue:
- [P1a] proves D3D12 MXFP4 GEMV works on synthetic data with 0.04% error
- [P1b] cannot extend this to real model weights because bf16 quantization introduces precision loss AND my kernel code path on real weights diverges from CPU ref by ~constant factor (not yet isolated)

Even if [P1b] is fixed:
- [Phase C] requires scheduler + KV-cache rollback + draft verification integration (estimated 5-10 days)
- [Final measurement] requires a 1.3-1.5x tok/s speedup which would require:
  - High enough MTP accept ratio (need precise MTP head numerics — blocker)
  - Sub-ms MTP head latency (my [P1a] is 0.3ms for FC alone, full MTP would be 1-2ms — slower than vmlx_mtp_tuning's expected ~3ms budget)

## Artifacts created

All in E:\FreeToken\benchmarks\cpu_moe_microbench\:
- t_mxfp4_gemv_d3d12.hlsl, t_mxfp4_gemv_d3d12.cpp — MXFP4 GEMV (synthetic pass)
- t_mxfp4_gemv_reference.py, t_mxfp4_compare2.py — Python reference + compare
- t_mxfp4_gemv_sk.dxil — compiled shader
- t_mxfp4_single_test.py, t_mtp_fc_* — various test scaffolds
- t_nvfp4_gemv_d3d12.hlsl, t_mtp_fc_test.cpp — NVFP4 GEMV (real weights, unverified)
- t_mtp_fc_weights.bin, t_mtp_fc_with_act.bin, t_mtp_fc_1row.bin — input dumps
- P1a_STATUS.md — [P1a] detailed status
- This file: P1b_STATUS.md

## What would be needed to continue

1. Isolate [P1b] root cause (1-2 days focused debugging)
2. Implement full MTP head forward on iGPU (3-5 days)
3. Integrate with FreeToken scheduler (5-10 days)
4. End-to-end tok/s measurement (1-2 days)
Total: 10-18 days focused engineering, assuming [P1b] is fixable.
