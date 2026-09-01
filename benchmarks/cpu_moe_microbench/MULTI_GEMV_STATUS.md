# Multi-GEMV Status (Time up)

## What was done
- Wrote multi-GEMV shader with cbuffer {B, K, nbPerRow, nsPerRow}
- Compiled successfully (7412 bytes)
- Added MULTI_GEMV command to v3 server with proper resource allocation
- Rebuilt v3 server

## What didn't work
- Output was all zeros - reason: the existing pso was created with the OLD shader's
  cbuffer layout ({K, nb, ns, 0}). The MULTI_GEMV uses the new shader with different
  cbuffer ({B, K, nb, ns}). Mismatch causes garbage data, output = 0.
- Fix: need a separate pso for the multi-GEMV shader. Would need to add
  pso2 = CreateComputePipelineState(...) with the new shader. Did not finish.

## What's still working
- [OK] STATELESS for single GEMV (M=1 K=4096): bit-exact with PyTorch ref
- [OK] STATELESS for M=4 K=4096: bit-exact with manual
- [OK] Per-row bias: works correctly (STATELESS with bias=2.5 adds 2.5 to output)
- [OK] MULTI_GEMV executes (no errors) but output is garbage due to pso cbuffer mismatch

## Time spent on multi-GEMV
- Write shader: 30 min
- Write v3 server command: 1 hour
- Debug (3 attempts, restore file 3 times): 1.5 hours
- Total: 3 hours

## Final Status
The basic GEMV path (STATELESS) works perfectly and is bit-exact.
For real performance, the multi-GEMV shader needs a second pso to work.
That requires another 1-2 hours of v3 server work.

## Recommendation
Current state is sufficient for documentation:
- True MXFP4 GEMV works (single-shot)
- Per-row bias works
- M=1 K=4096 (FC) verified end-to-end
- M=4 K=4096 verified (bit-exact for MATMUL not just GEMV)

For real speedup with multiple GEMVs in one dispatch, the multi-GEMV shader
would work IF a second pso is created in the v3 server.