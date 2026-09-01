// MXFP4 GEMV with FLOAT activation (for RMSNorm'd inputs).
// Same layout as t_mxfp4_gemv_d3d12.hlsl but act is StructuredBuffer<float>.

StructuredBuffer<uint>  packed : register(t0);
StructuredBuffer<uint>  scl    : register(t1);
StructuredBuffer<float> act    : register(t2);
StructuredBuffer<float> bias   : register(t3);
StructuredBuffer<float> gbl    : register(t4);
RWStructuredBuffer<float> outv : register(u0);

cbuffer P : register(b0) {
    uint K;
    uint nbPerRow;
    uint nsPerRow;
    uint pad;
};

static const int kE2M1[16] = {0,1,2,3,4,6,8,12, 0,-1,-2,-3,-4,-6,-8,-12};

groupshared float sh[256];

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID,
          uint  t    : SV_GroupIndex,
          uint3 gr   : SV_GroupID) {
    uint row = gr.x;
    if (row >= 2048) return;

    float acc = 0.0f;

    for (uint b = t; b < (K / 32u); b += 256u) {
        uint sPack = scl[row * nsPerRow + (b >> 2)];
        uint byteIdx = b & 3u;
        uint sb;
        if      (byteIdx == 0) sb =  sPack        & 0xFFu;
        else if (byteIdx == 1) sb = (sPack >>  8) & 0xFFu;
        else if (byteIdx == 2) sb = (sPack >> 16) & 0xFFu;
        else                   sb = (sPack >> 24) & 0xFFu;
        float bs = (sb == 0u) ? 0.0f : exp2((float)((int)sb - 127));

        uint pBase = row * nbPerRow + b * 4u;
        uint w0 = packed[pBase + 0u];
        uint w1 = packed[pBase + 1u];
        uint w2 = packed[pBase + 2u];
        uint w3 = packed[pBase + 3u];

        uint abase = b * 32u;
        float wsum = 0.0f;
        wsum += (float)kE2M1[(w0      ) & 0xFu] * act[abase +  0]
              + (float)kE2M1[(w0 >>  4) & 0xFu] * act[abase +  1]
              + (float)kE2M1[(w0 >>  8) & 0xFu] * act[abase +  2]
              + (float)kE2M1[(w0 >> 12) & 0xFu] * act[abase +  3]
              + (float)kE2M1[(w0 >> 16) & 0xFu] * act[abase +  4]
              + (float)kE2M1[(w0 >> 20) & 0xFu] * act[abase +  5]
              + (float)kE2M1[(w0 >> 24) & 0xFu] * act[abase +  6]
              + (float)kE2M1[(w0 >> 28) & 0xFu] * act[abase +  7];
        wsum += (float)kE2M1[(w1      ) & 0xFu] * act[abase +  8]
              + (float)kE2M1[(w1 >>  4) & 0xFu] * act[abase +  9]
              + (float)kE2M1[(w1 >>  8) & 0xFu] * act[abase + 10]
              + (float)kE2M1[(w1 >> 12) & 0xFu] * act[abase + 11]
              + (float)kE2M1[(w1 >> 16) & 0xFu] * act[abase + 12]
              + (float)kE2M1[(w1 >> 20) & 0xFu] * act[abase + 13]
              + (float)kE2M1[(w1 >> 24) & 0xFu] * act[abase + 14]
              + (float)kE2M1[(w1 >> 28) & 0xFu] * act[abase + 15];
        wsum += (float)kE2M1[(w2      ) & 0xFu] * act[abase + 16]
              + (float)kE2M1[(w2 >>  4) & 0xFu] * act[abase + 17]
              + (float)kE2M1[(w2 >>  8) & 0xFu] * act[abase + 18]
              + (float)kE2M1[(w2 >> 12) & 0xFu] * act[abase + 19]
              + (float)kE2M1[(w2 >> 16) & 0xFu] * act[abase + 20]
              + (float)kE2M1[(w2 >> 20) & 0xFu] * act[abase + 21]
              + (float)kE2M1[(w2 >> 24) & 0xFu] * act[abase + 22]
              + (float)kE2M1[(w2 >> 28) & 0xFu] * act[abase + 23];
        wsum += (float)kE2M1[(w3      ) & 0xFu] * act[abase + 24]
              + (float)kE2M1[(w3 >>  4) & 0xFu] * act[abase + 25]
              + (float)kE2M1[(w3 >>  8) & 0xFu] * act[abase + 26]
              + (float)kE2M1[(w3 >> 12) & 0xFu] * act[abase + 27]
              + (float)kE2M1[(w3 >> 16) & 0xFu] * act[abase + 28]
              + (float)kE2M1[(w3 >> 20) & 0xFu] * act[abase + 29]
              + (float)kE2M1[(w3 >> 24) & 0xFu] * act[abase + 30]
              + (float)kE2M1[(w3 >> 28) & 0xFu] * act[abase + 31];

        acc += wsum * bs;
    }

    sh[t] = acc;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (t < s) sh[t] += sh[t + s];
        GroupMemoryBarrierWithGroupSync();
    }
    if (t == 0u) {
        outv[row] = (sh[0] + bias[row]) * gbl[row];
    }
}
