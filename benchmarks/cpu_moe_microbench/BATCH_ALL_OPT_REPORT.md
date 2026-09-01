# Final Status: BATCH_ALL Optimization Attempt

**Date**: 2026-08-27
**Status**: **Shader correct**, v3 server STATELESS bit-exact, BATCH_ALL has issues

## What Was Found
1. v3 server STATELESS is BIT-EXACT for M=1 AND M=4 (verified with real MTP weights)
2. STATELESS works with bias=0 AND bias!=0 (per-row bias correctly added)
3. MtpIgpuMoeExecutor's BATCH_ALL approach has issues (16 separate dispatches, slow)
4. Multi-GEMV shader compiled but cbuffer layout differs from v3 server's PSO
5. The original "missing file" error was due to cbuffer mismatch in PSO creation

## Key Findings
- STATELESS is the reliable path (verified bit-exact)
- BATCH_ALL with new shader needs server changes (which we already made)
- BATCH_ALL is slow because v3 server does 16 separate dispatches
- For real speedup, need a shader that processes all 16 GEMVs in ONE dispatch

## What Works
- [OK] v3 server STATELESS for FC (M=1, K=4096): bit-exact with PyTorch ref
- [OK] v3 server STATELESS for gate/up/down (M=512, K=2048): bit-exact
- [OK] MtpIgpuMoeExecutor via STATELESS (not BATCH_ALL): works correctly
- [OK] Bias per-row correctly handled in shader's t3 slot

## What's Broken
- [GAP] MtpIgpuMoeExecutor via BATCH_ALL: hangs (server cbuffer mismatch)
- [GAP] MtpIgpuMoeExecutor via 24 sequential CALLs: works but slow (~20ms vs CPU 3.7ms)

## Path Forward
To get real speedup, the v3 server needs to dispatch all 16 GEMVs in ONE shader call.
That's a non-trivial change to the shader (need shared memory layout for multiple GEMVs)
AND to v3 server (need to handle the BATCH_ALL payload format properly).
Estimated 2-3 more days.

## Honest Assessment
The foundation is solid:
- True MXFP4 GEMV shader works (verified bit-exact)
- v3 server handles per-row bias correctly
- Single GEMV dispatch works (FC, gate, up, down) with bit-exact output

But the MoE pipeline is still slow because of 24 sequential Python subprocess calls.
Each call has ~0.5ms overhead (Python + struct + pipe write + pipe read + GPU dispatch).
Total: 12ms minimum overhead for 24 calls.

For a real demo:
1. Write a true multi-GEMV shader (single dispatch for B GEMVs)
2. Update v3 server to use single dispatch for BATCH_ALL
3. Use that path from MtpIgpuMoeExecutor

Without this, the MtpIgpuMoeExecutor path is functionally correct but performance is bad.
Time spent: significant. Time remaining: limited.

## Deliverables This Phase
- New shader (t_mxfp4_true_gemv.hlsl): correct MXFP4 GEMV math
- New shader (t_mxfp4_multi_gemv.hlsl): batched design (not yet integrated)
- v3 server: per-row bias separation works
- v3 server: BATCH_ALL has bias/act swap fix (but slow with new shader)
- MtpIgpuMoeExecutor: uses STATELESS-style dispatch (24 sequential, correct but slow)

## Recommended Next Action
Use v3 server STATELESS (single GEMV) for now. It's correct and bit-exact.
For real speedup, either:
1. Write multi-GEMV shader + integrate (2-3 days) - gives actual speedup
2. Move to scheduler integration (8-9 days) - uses current single-GEMV path, gives small speedup
3. Accept current state and document - FC on iGPU works, MTP MoE iGPU slow