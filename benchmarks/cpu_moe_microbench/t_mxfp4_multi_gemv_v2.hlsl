// [P1k] Multi-GEMV MXFP4 kernel: B independent GEMVs in one Dispatch.
// B is in cbuffer, so v3 server can pass B per call.
//
// Inputs:
//   packed:  StructuredBuffer<uint>  - [B, K/8]  B rows of packed weights
//   scl:     StructuredBuffer<float> - [B, K/32] B rows of pre-decoded e8m0 scales
//   act:     StructuredBuffer<float> - [B, K]   B rows of activations
//   bias:    StructuredBuffer<float> - [B]      B biases (one per item)
//   gbl:     StructuredBuffer<float> - [B]      B per-row global scales
//   outv:    RWStructuredBuffer<float> - [B]    B outputs
//
// cbuffer:
//   B:        number of items
//   K:        K-dim
//   nbPerRow: K/8
//   nsPerRow: K/32
//
// Formula:
//   for each b in 0..B-1:
//     wsum = sum_{micro-block} kE2M1[w_b,k] * act_b[k]
//     outv[b] = (wsum + bias_b) * gbl_b

StructuredBuffer<uint>  packed : register(t0);
StructuredBuffer<float> scl    : register(t1);
StructuredBuffer<float> act    : register(t2);
StructuredBuffer<float> bias   : register(t3);
StructuredBuffer<float> gbl    : register(t4);
RWStructuredBuffer<float> outv : register(u0);

cbuffer P : register(b0) {
    uint B;
    uint K;
    uint nbPerRow;
    uint nsPerRow;
};

static const int kE2M1[16] = {0,1,2,3,4,6,8,12, 0,-1,-2,-3,-4,-6,-8,-12};

groupshared float sh[256];

[numthreads(256, 1, 1)]
void main(uint3 gr : SV_GroupID, uint t : SV_GroupIndex) {
    uint b = gr.x;  // batch index
    float acc = 0.0f;

    [loop]
    for (uint mb = t; mb < nsPerRow; mb += 256u) {
        uint pBase = b * nbPerRow + mb * 4u;
        uint w0 = packed[pBase + 0u];
        uint w1 = packed[pBase + 1u];
        uint w2 = packed[pBase + 2u];
        uint w3 = packed[pBase + 3u];

        float bs = scl[b * nsPerRow + mb];

        uint abase = b * K + mb * 32u;

        float wsum = 0.0f;
        wsum += (float)kE2M1[(w0      ) & 0xFu] * act[abase +  0u]
              + (float)kE2M1[(w0 >>  4) & 0xFu] * act[abase +  1u]
              + (float)kE2M1[(w0 >>  8) & 0xFu] * act[abase +  2u]
              + (float)kE2M1[(w0 >> 12) & 0xFu] * act[abase +  3u]
              + (float)kE2M1[(w0 >> 16) & 0xFu] * act[abase +  4u]
              + (float)kE2M1[(w0 >> 20) & 0xFu] * act[abase +  5u]
              + (float)kE2M1[(w0 >> 24) & 0xFu] * act[abase +  6u]
              + (float)kE2M1[(w0 >> 28) & 0xFu] * act[abase +  7u];
        wsum += (float)kE2M1[(w1      ) & 0xFu] * act[abase +  8u]
              + (float)kE2M1[(w1 >>  4) & 0xFu] * act[abase +  9u]
              + (float)kE2M1[(w1 >>  8) & 0xFu] * act[abase + 10u]
              + (float)kE2M1[(w1 >> 12) & 0xFu] * act[abase + 11u]
              + (float)kE2M1[(w1 >> 16) & 0xFu] * act[abase + 12u]
              + (float)kE2M1[(w1 >> 20) & 0xFu] * act[abase + 13u]
              + (float)kE2M1[(w1 >> 24) & 0xFu] * act[abase + 14u]
              + (float)kE2M1[(w1 >> 28) & 0xFu] * act[abase + 15u];
        wsum += (float)kE2M1[(w2      ) & 0xFu] * act[abase + 16u]
              + (float)kE2M1[(w2 >>  4) & 0xFu] * act[abase + 17u]
              + (float)kE2M1[(w2 >>  8) & 0xFu] * act[abase + 18u]
              + (float)kE2M1[(w2 >> 12) & 0xFu] * act[abase + 19u]
              + (float)kE2M1[(w2 >> 16) & 0xFu] * act[abase + 20u]
              + (float)kE2M1[(w2 >> 20) & 0xFu] * act[abase + 21u]
              + (float)kE2M1[(w2 >> 24) & 0xFu] * act[abase + 22u]
              + (float)kE2M1[(w2 >> 28) & 0xFu] * act[abase + 23u];
        wsum += (float)kE2M1[(w3      ) & 0xFu] * act[abase + 24u]
              + (float)kE2M1[(w3 >>  4) & 0xFu] * act[abase + 25u]
              + (float)kE2M1[(w3 >>  8) & 0xFu] * act[abase + 26u]
              + (float)kE2M1[(w3 >> 12) & 0xFu] * act[abase + 27u]
              + (float)kE2M1[(w3 >> 16) & 0xFu] * act[abase + 28u]
              + (float)kE2M1[(w3 >> 20) & 0xFu] * act[abase + 29u]
              + (float)kE2M1[(w3 >> 24) & 0xFu] * act[abase + 30u]
              + (float)kE2M1[(w3 >> 28) & 0xFu] * act[abase + 31u];

        acc += wsum * bs;
    }

    sh[t] = acc;
    GroupMemoryBarrierWithGroupSync();
    [unroll(8)]
    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (t < s) sh[t] += sh[t + s];
        GroupMemoryBarrierWithGroupSync();
    }

    if (t == 0u) {
        outv[b] = (sh[0] + bias[b]) * gbl[b];
    }
}
