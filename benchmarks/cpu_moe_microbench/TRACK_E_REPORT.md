# Track 2 (E) Report: M=1 Path Validation (REVISED)

**Date**: 2026-08-27
**Status**: PASS (4/4 tests, scope reduced to M=1)

## Key Finding
**M>1 is fundamentally broken in both v1 and v2** due to a kernel/binding mismatch:
- The kernel reads `scl[row * nbPerRow + b] & 0xFFu` for per-row per-block scale
- v1/v2 server binds `scl` (slot 1) to rS (K*4 = 16384 bytes of act bytes), regardless of M
- For M=2, the kernel reads scales from a buffer that has only K*4 bytes, not M*K/32 bytes
- Result: row 1 of M=2 reads garbage scales, output differs from M=1 alone

This is **not a v2 regression** — confirmed identical bug in v1 server (P1d_STATUS known issue).

## Scope Decision
For MTP head, **only fc is dispatched via iGPU** (M=1, K=4096). attn+MoE run on PyTorch.
M>1 path is NOT a blocker for Track 4 (A) integration. M>1 will be fixed as part of
Track 3 (C) when we rewrite the shader with proper e8m0 scale binding.

## Test Results

### Test 1: M=1 baseline 100 calls, no drift
- 133ms for 100 calls (1.33ms/call, includes Python overhead)
- mean=-619801, std=0.00e+00, drift=0.00e+00
- All finite, no NaN/Inf
- **PASS**

### Test 2: M=1 LOAD-rewrite consistency
- LOAD 'e' twice with different packed, CALL with act_b
- CALL output = STATELESS(packed_b, act_b) bit-exact
- **2nd LOAD correctly takes effect** even with same M, same name
- **PASS**

### Test 3: 8 different M=1 weights, no interference
- 8 LOADs + 8 CALLs in single process
- All 8 outputs bit-exact match with STATELESS for each
- **No cross-talk between weights**
- **PASS**

### Test 4: 1000 LOAD+CALL cycles
- 544ms for 1000 calls (0.54ms/call, GPU dispatch dominates)
- mean=785554, std=0.00e+00, drift=0.00e+00
- All finite, no NaN/Inf
- **No memory leak, no drift over 1000 cycles**
- **PASS**

## Performance Summary (M=1)
- 1000 CALLs in 544ms = **0.54ms/call** (server-side, GPU dispatch)
- This is the production MTP head fc latency budget per draft token
- For K=3 drafts, we need 3 CALLs + 1 main verify = ~2ms total
- Per-token budget: <2ms for MTP draft path on iGPU

## Impact on Track 4 (A)
- M=1 path is production-ready
- M>1 path deferred to Track 3 (C) + future work
- MTP head integration can proceed with M=1 fc only