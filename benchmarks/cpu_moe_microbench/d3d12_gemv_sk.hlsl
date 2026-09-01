StructuredBuffer<uint2> packed : register(t0);
StructuredBuffer<uint> scl : register(t1);
StructuredBuffer<int> act : register(t2);
StructuredBuffer<float> asb : register(t3);
StructuredBuffer<float> gbl : register(t4);
RWStructuredBuffer<float> outv : register(u0);
cbuffer P : register(b0) { uint K; uint nbPerRow; float gs; float pad; }

static const int kE2M1x2[16] = {0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};
groupshared float sh[256];

[numthreads(256, 1, 1)]
void main(uint3 g : SV_DispatchThreadID, uint t : SV_GroupIndex, uint3 gr : SV_GroupID) {
    uint row = gr.x;
    float acc = 0.0;
    for (uint b = t; b < nbPerRow; b += 256) {
        uint2 pk = packed[row * nbPerRow + b];
        uint sb = scl[row * nbPerRow + b] & 0xFFu;
        int wsum = 0;
        uint bytes[8];
        bytes[0] = pk.x & 0xFFu; bytes[1] = (pk.x >> 8) & 0xFFu; bytes[2] = (pk.x >> 16) & 0xFFu; bytes[3] = (pk.x >> 24) & 0xFFu;
        bytes[4] = pk.y & 0xFFu; bytes[5] = (pk.y >> 8) & 0xFFu; bytes[6] = (pk.y >> 16) & 0xFFu; bytes[7] = (pk.y >> 24) & 0xFFu;
        uint abase = b * 16;
        for (int j = 0; j < 8; j++) {
            int wlo = kE2M1x2[bytes[j] & 0xFu];
            int whi = kE2M1x2[(bytes[j] >> 4) & 0xFu];
            wsum += wlo * act[abase + j] + whi * act[abase + 8 + j];
        }
        acc += (float)wsum * 0.01f * (float)sb + asb[b];
    }
    sh[t] = acc;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = 128; s > 0; s >>= 1) {
        if (t < s) sh[t] += sh[t + s];
        GroupMemoryBarrierWithGroupSync();
    }
    if (t == 0) outv[row] = sh[0] * gs * gbl[row];
}
