StructuredBuffer<uint2> packed : register(t0);
StructuredBuffer<uint> scl : register(t1);
StructuredBuffer<int> act : register(t2);       // 每元素 1 个 int8 值（int32 存储）
StructuredBuffer<float> asb : register(t3);
RWStructuredBuffer<float> outv : register(u0);
cbuffer P : register(b0) { uint K; uint nbPerRow; float gs; float pad; }

static const int kE2M1x2[16] = {0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};

[numthreads(256, 1, 1)]
void main(uint3 g : SV_DispatchThreadID) {
    uint row = g.x;
    uint base = row * nbPerRow;
    float acc = 0.0;
    for (uint b = 0; b < nbPerRow; b++) {
        uint2 pk = packed[base + b];
        uint sb = scl[base + b] & 0xFFu;
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
    outv[row] = acc * gs;
}
