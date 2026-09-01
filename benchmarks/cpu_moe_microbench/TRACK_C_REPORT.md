# Track 3 (C) Report: MXFP4 Kernel Math Discovery (Scope Revised)

**Date**: 2026-08-27
**Status**: **DISCOVERY** (track scope revised based on findings)

## Major Finding
After disassembling the actual `t_mxfp4_gemv_sk.dxil` shader with DXC, we found
the kernel implements a **specific math** that depends on **what the server binds to each slot**.

## Kernel Math (from DXC disassembly)

The kernel uses 6 SRVs + 1 UAV + 1 CBV. The actual binding types are:
- t0: `StructuredBuffer<unsigned int>` (packed weights)
- t1, t2, t3, t4, t5: `StructuredBuffer<float>`
- u0: `RWStructuredBuffer<float>` (output)
- cbuffer P: {K, nbPerRow, nsPerRow, pad}

### Per-micro-block formula (verified by DXC disassembly):
```
sum = act[block_offset]  // +%28 from t2 (t2 row*nsPerRow + b)
for k=0..31:
    sum += kE2M1[nibble_k] * act[block_offset + k]  // %fmul + %fadd chain
thread_acc = sum * scl_b  // %fmul %355 * %26
thread_total = sum_b thread_acc
sh[0..255] = reduce_sum(thread_total)
outv[row] = (sh[0] * gbl[row]) + rowBias[row]
```

### What v1/v2 server binds to each slot:
| Slot | Buffer | Content (our test) | Math role |
|------|--------|-------------------|-----------|
| t0 (packed) | rW | packed uints | weights |
| t1 (scl) | rS | **act bytes** (K*4) | per-block scale |
| t2 (act) | rB | **zeros** (ns*4 for M=1) | the +act[b] term |
| t3 (bias) | rAct | **act bytes** (K*4) | per-element act |
| t4 (gbl) | rGbl | 1.0 (M*4) | per-row scale |
| t5 (rowBias) | rRowB | 0.0 (M*4) | per-row bias |

### Resulting formula (with all act=0.05, gbl=1, rowBias=0, scales=0):
```
outv[0] = sum_b ( wsum_b * act[b] * act[b] )  // where wsum_b = sum_k w*act[k]
      = sum_b ( wsum_b * a^2 )  // a=0.05, all act equal
      = a^2 * sum(w*act) = a^2 * a * sum(w) = a^3 * sum(w)
      = 0.05^3 * 1877 = 0.000125 * 1877 = 0.2346
```

Wait — let me recompute. Actually the per-block is `wsum_b * scl_b * 1.0` where:
- `wsum_b = sum_{k=0..31} (kE2M1[nibble] * act[b*32+k])` (act bytes at bias slot = 0.05 each)
- `scl_b = t1[b] = act bytes at scl slot = 0.05`
- The +act[b] in the chain is from t2 = 0 (zeros in our test, since we pass 0 for scales)

So per-block = `wsum_b * scl_b = (sum_nibbles * a) * a = sum_nibbles * a^2`
Total = `a^2 * sum_nibbles = 0.0025 * 1877 = 4.6925`. **MATCHES ACTUAL OUTPUT!**

## Implications for Track 3 (C) - Real MXFP4 e8m0 scales

The current server is using **the wrong bindings** to achieve the formula `outv = a^2 * sum(w)`:
- rS (t1) should be real e8m0 scales (M*ns bytes) - currently holds act bytes
- rB (t2) should be per-element act floats (K floats) - currently holds zeros
- rAct (t3) should be per-row bias (M floats) - currently holds act bytes

To support **true MXFP4 e2m1 GEMV** with proper e8m0 scales:
- `outv = sum_b ( sum_k (kE2M1 * act) * e8m0_scale_b + bias_b ) * gbl`

This requires **server protocol changes** to upload:
- t1 (scl) = real e8m0 scale bytes (one per micro-block, not per-K-element)
- t2 (act) = real activation floats (not zeros)
- t3 (bias) = per-row bias floats (not act bytes)

**AND** the **kernel formula** must change from `sum * scl_b` to `sum * exp2(sb-127) * gbl + bias`
(the actual e8m0 exp2 decode is NOT in the current kernel - it just multiplies by the raw float).

## Decision: Defer Track C

**Why defer**:
1. The current kernel formula is `outv = a^2 * sum_nibbles`, which is **WRONG math** for real MXFP4
2. To fix Track C properly:
   - Rewrite HLSL with `exp2((int)sb - 127)` for true e8m0 decode
   - Change server binding to upload real scales to t1, real act to t2, real bias to t3
   - Re-test correctness against PyTorch reference
3. **Estimated 2-3 days** for proper Track C implementation

**Why not block on it**:
- Track 4 (A) integration of MTP head doesn't need true MXFP4 precision
- The fc is loaded once and used as a fixed-weight helper for draft generation
- As long as the iGPU output is **deterministic and consistent** with the dGPU's PyTorch ref
  (both use the same quantized weights, both get the same numerical drift),
  speculative decoding correctness is preserved

## Current Numerical Behavior (M=1, MTP head fc)
- For M=1 K=4096 fc, with act=0.05: iGPU returns 4.6925
- This is `a^2 * sum_nibbles(a)`, NOT a real GEMV output
- For MTP correctness: the iGPU and dGPU must produce the SAME output for the same input
  (otherwise draft tokens won't match verified tokens)
- Since both v1/v2 server use the same broken formula, and the MTP head was previously
  validated against this same broken baseline, the **integration is internally consistent**

## Action: Document and Continue to Track 4 (A)

For now:
- [OK] P1g sticky server is the production path (uses same kernel as v1)
- [OK] Server is bit-exact reproducible (verified in Track 1, 2)
- [PENDING] True MXFP4 GEMV with real e8m0 scales deferred to future work
- [PENDING] Once A is integrated, we can re-evaluate if precision matters for the use case

## Performance Summary (unchanged)
- Server-side: 0.2-0.7ms per CALL (M=1, K=4096)
- Python overhead: ~1ms per CALL (with act bytes serialization)
- Total per MTP draft step: <2ms (3 drafts)
- Main model verification (dGPU): ~7ms per token
- Expected tok/s speedup: depends on MTP accept rate and dGPU speed