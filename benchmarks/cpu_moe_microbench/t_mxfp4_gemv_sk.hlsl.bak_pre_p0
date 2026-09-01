// MXFP4 GEMV kernel (e8m0 block scales, no per-block bias)
// Reuses W4A8 layout (1 byte per K element, two elements packed per byte)
// but replaces e4m3 scale decoding with e8m0 (exponent-only).
//
// Layout:
//   packed : StructuredBuffer<uint> of weight, 2 K-elements per byte, K/2 bytes per row
//            (W4A8 used uint2 = 8 elements; we use uint = 4 elements for simpler addressing)
//   scl    : StructuredBuffer<uint8_t> of per-K/32 block scales (e8m0)
//   act    : StructuredBuffer<int> of activations, K per row
//   bias   : StructuredBuffer<float> of per-output bias (added once per row)
//   outv   : RWStructuredBuffer<float> of outputs, M
//   gbl    : StructuredBuffer<float> of per-output global scale (multiplied last)

StructuredBuffer<uint>  packed : register(t0);   // M * (K/4) uints (each uint = 4 nibbles)
StructuredBuffer<uint>  scl    : register(t1);   // M * (K/32) uints (each uint = 4 e8m0 scales)
StructuredBuffer<int>   act    : register(t2);   // K ints
StructuredBuffer<float> bias   : register(t3);   // M floats
RWStructuredBuffer<float> outv : register(u0);
StructuredBuffer<float> gbl    : register(t4);

cbuffer P : register(b0) { uint K; uint nbPerRow; uint nsPerRow; }

// e2m1 nibble decode (same as W4A8 path)
static const int kE2M1[16] = {0,1,2,3,4,6,8,12, 0,-1,-2,-3,-4,-6,-8,-12};

groupshared float sh[256];

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID,
          uint  t    : SV_GroupIndex,
          uint3 gr   : SV_GroupID) {
    uint row = gr.x;
    float acc = 0.0f;

    // For each micro-block of 32 K-elements (8 uints in packed, 1 byte in scl)
    for (uint b = t; b < nbPerRow; b += 256) {
        uint pbase = row * nbPerRow + b;     // base into packed
        uint sbase = row * nsPerRow + b;     // base into scl (1 byte per micro-block)

        // Load scale byte once; decode e8m0 → scale = 2^(b - 127)
        // Pack 4 e8m0 bytes into a uint and extract one byte at this b position.
        uint packIdx = b >> 2;       // which uint in scl
        uint byteIdx = b & 3u;       // which byte in that uint (0..3)
        uint sPack = scl[sbase & ~3u];   // safe when nbPerRow is mult of 4; for safety do per-byte read:
        uint sb;
        if      (byteIdx == 0) sb =  sPack        & 0xFFu;
        else if (byteIdx == 1) sb = (sPack >>  8) & 0xFFu;
        else if (byteIdx == 2) sb = (sPack >> 16) & 0xFFu;
        else                   sb = (sPack >> 24) & 0xFFu;
        // e8m0: scale = 2^(byte - 127).  byte=127 → 1.0; byte=128 → 2.0; byte=126 → 0.5
        float bs = (sb == 0) ? 0.0f : exp2((float)((int)sb - 127));

        // Walk 8 nibbles from this uint (each uint = 4 bytes = 4 K-elements at 4-bit each).
        // Actually we want K/2 bytes per micro-block = 32 bytes = 8 uints.
        // For simplicity, we load 1 uint per iteration step and decode 8 nibbles.
        // nbPerRow = K/32 micro-blocks. Each micro-block has 32 K-elements.
        // 32 K-elements * 4-bit = 128 bits = 4 uints.
        // So per micro-block we read 4 uints and 1 scale byte.
        uint words[4];
        words[0] = packed[pbase * 4u + 0u];
        words[1] = packed[pbase * 4u + 1u];
        words[2] = packed[pbase * 4u + 2u];
        words[3] = packed[pbase * 4u + 3u];

        int wsum = 0;
        uint abase = b * 32u;
        for (int j = 0; j < 4; j++) {
            uint w = words[j];
            int w0 = kE2M1[w        & 0xFu];
            int w1 = kE2M1[(w >>  4) & 0xFu];
            int w2 = kE2M1[(w >>  8) & 0xFu];
            int w3 = kE2M1[(w >> 12) & 0xFu];
            int w4 = kE2M1[(w >> 16) & 0xFu];
            int w5 = kE2M1[(w >> 20) & 0xFu];
            int w6 = kE2M1[(w >> 24) & 0xFu];
            int w7 = kE2M1[(w >> 28) & 0xFu];
            uint ai = abase + (uint)j * 8u;
            wsum += w0 * act[ai + 0] + w1 * act[ai + 1] + w2 * act[ai + 2] + w3 * act[ai + 3]
                  + w4 * act[ai + 4] + w5 * act[ai + 5] + w6 * act[ai + 6] + w7 * act[ai + 7];
        }
        acc += (float)wsum * bs;
    }

    sh[t] = acc;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = 128; s > 0; s >>= 1) {
        if (t < s) sh[t] += sh[t + s];
        GroupMemoryBarrierWithGroupSync();
    }
    if (t == 0) outv[row] = (sh[0] + bias[row]) * gbl[row];
}
