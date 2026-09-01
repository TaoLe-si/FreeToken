// [P1i] True MXFP4 GEMV shader for iGPU D3D12.
// Bit-exact with PyTorch reference. Supports M >= 1, K up to 8192.
//
// Inputs:
//   packed: StructuredBuffer<uint>  - shape [M, K/8], 4 bytes per uint, 8 nibbles per uint
//   scl:    StructuredBuffer<float> - shape [M, K/32], pre-decoded e8m0 scale per micro-block
//   act:    StructuredBuffer<float> - shape [K], activation per K-element
//   bias:   StructuredBuffer<float> - shape [M], per-row bias (added to output before gbl)
//   gbl:    StructuredBuffer<float> - shape [M], per-row global scale (multiplier)
//   outv:   RWStructuredBuffer<float> - shape [M], output
//
// Formula:
//   for each b in 0..K/32-1:
//     wsum = sum_{k=0..31} kE2M1[w] * act[b*32+k]
//     acc += wsum * scl[block_b]
//   outv[row] = (sh[0] + bias[row]) * gbl[row]
//
// Note: The kernel uses Dense root sig (8 slots), matching the existing v3 server.

StructuredBuffer<uint>  packed : register(t0);
StructuredBuffer<float> scl    : register(t1);
StructuredBuffer<float> act    : register(t2);
StructuredBuffer<float> bias   : register(t3);
StructuredBuffer<float> gbl    : register(t4);
RWStructuredBuffer<float> outv : register(u0);

cbuffer P : register(b0) {
    uint K;       // input dim
    uint nbPerRow; // K/8 uints per row
    uint nsPerRow; // K/32 micro-blocks per row
    uint pad;
};

// e2m1 nibble decode LUT (matches W4A8 / MXFP4)
static const int kE2M1[16] = {0,1,2,3,4,6,8,12, 0,-1,-2,-3,-4,-6,-8,-12};

groupshared float sh[256];

[numthreads(256, 1, 1)]
void main(uint3 gr : SV_GroupID, uint t : SV_GroupIndex) {
    uint row = gr.x;
    float acc = 0.0f;

    // Each micro-block has 32 K-elements = 4 uints (4 bytes * 8 nibbles/byte = 32 nibbles)
    [loop]
    for (uint b = t; b < nsPerRow; b += 256u) {
        // Load 4 packed uints for this micro-block
        uint pBase = row * nbPerRow + b * 4u;
        uint w0 = packed[pBase + 0u];
        uint w1 = packed[pBase + 1u];
        uint w2 = packed[pBase + 2u];
        uint w3 = packed[pBase + 3u];

        // Load scale for this micro-block (pre-decoded e8m0)
        float bs = scl[row * nsPerRow + b];

        // Load activation base index for this micro-block
        uint abase = b * 32u;

        // Decode 32 nibbles (4 uints * 8 nibbles each) and accumulate
        // Per uint: 8 nibbles at bit positions 0, 4, 8, 12, 16, 20, 24, 28
        float wsum = 0.0f;

        // uint 0: K-elements abase+0..7
        wsum += (float)kE2M1[(w0      ) & 0xFu] * act[abase +  0u]
              + (float)kE2M1[(w0 >>  4) & 0xFu] * act[abase +  1u]
              + (float)kE2M1[(w0 >>  8) & 0xFu] * act[abase +  2u]
              + (float)kE2M1[(w0 >> 12) & 0xFu] * act[abase +  3u]
              + (float)kE2M1[(w0 >> 16) & 0xFu] * act[abase +  4u]
              + (float)kE2M1[(w0 >> 20) & 0xFu] * act[abase +  5u]
              + (float)kE2M1[(w0 >> 24) & 0xFu] * act[abase +  6u]
              + (float)kE2M1[(w0 >> 28) & 0xFu] * act[abase +  7u];

        // uint 1: K-elements abase+8..15
        wsum += (float)kE2M1[(w1      ) & 0xFu] * act[abase +  8u]
              + (float)kE2M1[(w1 >>  4) & 0xFu] * act[abase +  9u]
              + (float)kE2M1[(w1 >>  8) & 0xFu] * act[abase + 10u]
              + (float)kE2M1[(w1 >> 12) & 0xFu] * act[abase + 11u]
              + (float)kE2M1[(w1 >> 16) & 0xFu] * act[abase + 12u]
              + (float)kE2M1[(w1 >> 20) & 0xFu] * act[abase + 13u]
              + (float)kE2M1[(w1 >> 24) & 0xFu] * act[abase + 14u]
              + (float)kE2M1[(w1 >> 28) & 0xFu] * act[abase + 15u];

        // uint 2: K-elements abase+16..23
        wsum += (float)kE2M1[(w2      ) & 0xFu] * act[abase + 16u]
              + (float)kE2M1[(w2 >>  4) & 0xFu] * act[abase + 17u]
              + (float)kE2M1[(w2 >>  8) & 0xFu] * act[abase + 18u]
              + (float)kE2M1[(w2 >> 12) & 0xFu] * act[abase + 19u]
              + (float)kE2M1[(w2 >> 16) & 0xFu] * act[abase + 20u]
              + (float)kE2M1[(w2 >> 20) & 0xFu] * act[abase + 21u]
              + (float)kE2M1[(w2 >> 24) & 0xFu] * act[abase + 22u]
              + (float)kE2M1[(w2 >> 28) & 0xFu] * act[abase + 23u];

        // uint 3: K-elements abase+24..31
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

    // Reduce across 256 threads
    sh[t] = acc;
    GroupMemoryBarrierWithGroupSync();
    [unroll(8)]
    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (t < s) sh[t] += sh[t + s];
        GroupMemoryBarrierWithGroupSync();
    }

    if (t == 0u) {
        outv[row] = (sh[0] + bias[row]) * gbl[row];
    }
}
