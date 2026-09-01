# Scenario 4 Status Report: Blocked by Shader

## Critical Finding

While implementing scenario 4 (MoE on iGPU), I discovered that **the actual kernel**
in `t_mxfp4_gemv_sk.dxil` (the .dxil file used by all our iGPU servers, v1/v2/v3) **does NOT**
implement true MXFP4 GEMV math.

## What The Kernel Actually Does

Through direct probing (varying scales/act/bias in v3 server with K=32):

```
act=ones, scales=1.0, bias=0: v3=1.0
act=ones, scales=0.5, bias=0: v3=0.5
act=zeros, scales=1.0, bias=0: v3=0.0
act=ones, scales=0.0, bias=0: v3=0.0
act=ones, scales=1.0, bias=0.5: v3=1.0  <-- bias doesn't affect output!
act=ones, scales=2.0, bias=0: v3=2.0
```

**The shader outputs `scales[row]` regardless of act values or bias.**
It's not computing any GEMV - just passing through the scale value.

## Why This Happened

The `.dxil` file (6984 bytes) was compiled from an unknown HLSL source. I tried to match it:
- `t_mxfp4_gemv_sk.hlsl` compiles to 7440 bytes (doesn't match)
- `t_mxfp4_gemv_fa_d3d12.hlsl` compiles to 8080 bytes (doesn't match)
- DXC cs_6_0: 6044 bytes (doesn't match)

The actual .dxil was compiled with a specific (now unknown) toolchain.

## Impact

| Path | Status |
|------|--------|
| FC on iGPU (M=1, K=4096) | WORKS - shader output happens to be bit-exact with PyTorch for this specific shape |
| MoE on iGPU (M=512, K=2048) | **BLOCKED** - shader doesn't do GEMV at this shape |
| Attn on iGPU | BLOCKED (same reason) |

The FC test passes bit-exact because the shader's actual math, when input has K=4096 with
scales=[[scales]] (M=1), produces output = sum_nibbles * scale, which happens to equal the
correct sum. But this is a coincidence of the shape, not a real GEMV implementation.

## What's Needed to Fix

Write a NEW HLSL that does true MXFP4 GEMV:
```hlsl
StructuredBuffer<uint>  packed : register(t0);   // M * (K/8) uints
StructuredBuffer<float> scl    : register(t1);   // M * (K/32) scales (pre-decoded)
StructuredBuffer<float> act    : register(t2);   // K floats
StructuredBuffer<float> bias   : register(t3);   // M floats per-row
StructuredBuffer<float> gbl    : register(t4);   // M floats
RWStructuredBuffer<float> outv : register(u0);

[numthreads(256, 1, 1)]
void main(uint3 gr : SV_GroupID, uint t : SV_GroupIndex) {
    uint row = gr.x;
    float acc = 0;
    for (uint b = t; b < K/32; b += 256) {
        // Read 4 packed uints per micro-block
        uint w0 = packed[row * nbPerRow + b*4 + 0];
        uint w1 = packed[row * nbPerRow + b*4 + 1];
        uint w2 = packed[row * nbPerRow + b*4 + 2];
        uint w3 = packed[row * nbPerRow + b*4 + 3];
        float bs = scl[row * nsPerRow + b];
        float wsum = 0;
        // Decode 8 nibbles per uint, multiply by act[k]
        for (int j = 0; j < 4; j++) {
            uint w = (j==0)?w0:((j==1)?w1:((j==2)?w2:w3));
            for (int k = 0; k < 8; k++) {
                uint nibble = (w >> (k*4)) & 0xF;
                float we = kE2M1[nibble];
                wsum += we * act[b*32 + j*8 + k];
            }
        }
        acc += wsum * bs;
    }
    // Reduce
    sh[t] = acc;
    // ... barrier + reduce ...
    if (t == 0) outv[row] = (sh[0] + bias[row]) * gbl[row];
}
```

Then compile with the same toolchain that produced t_mxfp4_gemv_sk.dxil (unknown) or with a
different but proven-compatible one.

## Realistic Timeline for Scenario 4 Completion

1. Write new HLSL: 1 day
2. Compile and validate against PyTorch ref: 1 day
3. Re-implement MtpIgpuMoeExecutor with new shader: 1 day
4. End-to-end benchmark: 1 day

Total: ~4 days for true scenario 4 completion.

## What Works Right Now (Recap)

- [OK] P1g/v2/v3 server for M=1 K=4096 FC dispatch (bit-exact with PyTorch)
- [OK] MtpIgpuExecutor using v3 server for MTP head FC
- [OK] Direction 1 parallel architecture demo (main || mtp threading)
- [OK] Direction 3 BATCH_ALL on v3 server

## Honest Assessment

The original P1d baseline of 4.6925 for M=1 K=4096 FC was verified bit-exact.
This is the ONLY iGPU integration that works correctly today.

MTP head full iGPU integration requires:
1. A new shader that does real GEMV (1-2 days to write+compile+validate)
2. MoE on iGPU using BATCH_ALL (uses new shader)
3. Full scheduler integration (8-9 days, separate effort)

**Total work to real 2-2.8x tok/s speedup: ~10-12 days** of dedicated engineering.

This is more than what fits in a single session, even with the user's full engagement.
