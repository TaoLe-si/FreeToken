// MXFP4 GEMV kernel (e8m0 block scales, no per-block bias)
//
// Layout (server writes float32 to t1/t2; shader reads as float32):
//   packed : StructuredBuffer<uint> of weight, K/8 uints per row (each uint = 4 nibbles)
//   scl    : StructuredBuffer<float> of per-micro-block e8m0 scale
//            (server decodes byte 127 to 1.0, byte 128 to 2.0, etc.)
//            M rows of (K/32) floats
//   act    : StructuredBuffer<float> of activations, K per row (server casts int32 -> float)
//   bias   : StructuredBuffer<float> of per-output bias (M floats)
//   outv   : RWStructuredBuffer<float> of outputs, M
//   gbl    : StructuredBuffer<float> of per-output global scale (M floats)

StructuredBuffer<uint>    packed : register(t0);
StructuredBuffer<float>   scl    : register(t1);
StructuredBuffer<float>   act    : register(t2);
StructuredBuffer<float>   bias   : register(t3);
RWStructuredBuffer<float> outv   : register(u0);
StructuredBuffer<float>   gbl    : register(t4);

cbuffer P : register(b0) { uint K; uint nbPerRow; uint nsPerRow; };

// e2m1 nibble decode
static const int kE2M1[16] = {0,1,2,3,4,6,8,12, 0,-1,-2,-3,-4,-6,-8,-12};

groupshared float sh[256];

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID,
          uint  t    : SV_GroupIndex,
          uint3 gr   : SV_GroupID) {
    uint row = gr.x;
    float acc = 0.0f;

    for (uint b = t; b < nbPerRow; b += 256) {
        uint pbase = row * nbPerRow + b;
        uint sbase = row * nsPerRow + b;  // float index into scl

        // Load float scale for this micro-block (server has already decoded e8m0 byte to float)
        float bs = scl[sbase];

        // Load 4 uints (16 bytes) = 32 nibbles
        uint words[4];
        words[0] = packed[pbase * 4u + 0u];
        words[1] = packed[pbase * 4u + 1u];
        words[2] = packed[pbase * 4u + 2u];
        words[3] = packed[pbase * 4u + 3u];

        int wsum = 0;
        uint abase = b * 32u;  // element index in act
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
            // Load 8 acts as floats
            float a0 = act[row * K + ai + 0u];
            float a1 = act[row * K + ai + 1u];
            float a2 = act[row * K + ai + 2u];
            float a3 = act[row * K + ai + 3u];
            float a4 = act[row * K + ai + 4u];
            float a5 = act[row * K + ai + 5u];
            float a6 = act[row * K + ai + 6u];
            float a7 = act[row * K + ai + 7u];
            wsum += (int)round(w0 * a0 + w1 * a1 + w2 * a2 + w3 * a3
                              + w4 * a4 + w5 * a5 + w6 * a6 + w7 * a7);
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
