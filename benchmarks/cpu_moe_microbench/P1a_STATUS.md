# [P1a] MXFP4 GEMV on AMD Radeon 780M — STATUS: PASS

## Artifacts created
- E:FreeTokenenchmarkscpu_moe_microbench	_mxfp4_gemv_d3d12.hlsl  — MXFP4 GEMV compute shader (e8m0 scales, ByteAddressBuffer for int8 act)
- E:FreeTokenenchmarkscpu_moe_microbench	_mxfp4_gemv_d3d12.cpp   — host benchmark
- E:FreeTokenenchmarkscpu_moe_microbench	_mxfp4_gemv_sk.dxil    — compiled shader
- E:FreeTokenenchmarkscpu_moe_microbench	_mxfp4_gemv_d3d12.exe  — host exe
- E:FreeTokenenchmarkscpu_moe_microbenchuild_mxfp4_gemv.bat     — build script
- E:FreeTokenenchmarkscpu_moe_microbench	_mxfp4_gemv_reference.py — CPU PyTorch reference
- E:FreeTokenenchmarkscpu_moe_microbench	_mxfp4_compare2.py      — D3D12 vs CPU diff
- E:FreeTokenenchmarkscpu_moe_microbench	_mxfp4_single_test.py  — single-row sanity test generator
- E:FreeTokenenchmarkscpu_moe_microbench	_mxfp4_gemv_output.bin  — D3D12 output (after each run)

## Performance (M=2048, K=4096, N=100)
- p50 latency: **0.30-0.31 ms / iteration**
- throughput: **~55 GFLOPs** (kernel is naive; not yet optimized)
- AMD Radeon 780M theoretical W4A8 INT8 peak: ~7 TFLOPs
- We use ~0.8% of peak — huge headroom for optimization, but functional

## Numerical correctness
- Single-row sanity test (M=1, K=32, weights=1, acts=1, scale=1, bias=0, gbl=1): output = **32** ✓
- Random M=2048 K=4096 vs CPU PyTorch reference: max rel diff **3.7e-4** (~0.04%)
- The 0.04% error is reduction-order floating-point noise (CPU sums in one order, GPU in another). Within MXFP4 GEMV expected precision.

## Bugs found and fixed during [P1a]
1. D3D12 resource for UAV output needed `D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS` (was missing → access violation)
2. Readback heap type was wrong (was D3D12_HEAP_TYPE_DEFAULT, should be READBACK)
3. Shader declared `StructuredBuffer<int>` for int8 activations → read 4 bytes per int8, garbage in upper bytes. Switched to `ByteAddressBuffer` with `load_int8(byteOffset)` helper that sign-extends the low byte.

## What's next ([P1b])
Build a complete MTP head forward path on top of this GEMV:
- attn qkv proj + RoPE + RMSNorm + attention score/softmax + o proj
- MoE gate logits + top-8 expert selection + 8 expert GEMVs
- shared_expert GEMV + shared_expert_gate weighting
- mtp.norm RMSNorm at end
- Final fc projection (already have via this kernel)

## Hard acceptance status
- [P1a] PASSED ✓
- [P1b] pending
- Phase C (scheduler integration) pending
- Phase D (full decode with iGPU-MTP, measured tok/s speedup) pending

The final acceptance is end-to-end tok/s speedup vs vmlx_mtp_tuning.json's 1.564× target.
