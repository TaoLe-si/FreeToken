// MXFP4 GEMV: ByteAddressBuffer for act (byte-addressable).
// act.Load(byteOffset) returns uint32 (4 bytes). We extract low byte, sign-extend to int.

StructuredBuffer<uint> packed : register(t0);
StructuredBuffer<uint> scl    : register(t1);
ByteAddressBuffer act        : register(t2);
StructuredBuffer<float> bias  : register(t3);
StructuredBuffer<float> gbl   : register(t4);
RWStructuredBuffer<float> outv : register(u0);

cbuffer P : register(b0) {
    uint K;
    uint nbPerRow;
    uint nsPerRow;
    uint pad;
};

static const int kE2M1[16] = {0,1,2,3,4,6,8,12, 0,-1,-2,-3,-4,-6,-8,-12};

// Read signed int8 from ByteAddressBuffer byte offset
int load_int8(uint byteOff) {
    uint b = act.Load(byteOff & ~3u);  // align to 4
    uint shift = (byteOff & 3u) * 8u;
    uint val = (b >> shift) & 0xFFu;
    // sign extend: if high bit set, subtract 256
    return (val >= 128u) ? (int)(val - 256u) : (int)val;
}

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
        int wsum = 0;
        wsum += kE2M1[(w0      ) & 0xFu] * load_int8(abase +  0)
              + kE2M1[(w0 >>  4) & 0xFu] * load_int8(abase +  1)
              + kE2M1[(w0 >>  8) & 0xFu] * load_int8(abase +  2)
              + kE2M1[(w0 >> 12) & 0xFu] * load_int8(abase +  3)
              + kE2M1[(w0 >> 16) & 0xFu] * load_int8(abase +  4)
              + kE2M1[(w0 >> 20) & 0xFu] * load_int8(abase +  5)
              + kE2M1[(w0 >> 24) & 0xFu] * load_int8(abase +  6)
              + kE2M1[(w0 >> 28) & 0xFu] * load_int8(abase +  7);
        wsum += kE2M1[(w1      ) & 0xFu] * load_int8(abase +  8)
              + kE2M1[(w1 >>  4) & 0xFu] * load_int8(abase +  9)
              + kE2M1[(w1 >>  8) & 0xFu] * load_int8(abase + 10)
              + kE2M1[(w1 >> 12) & 0xFu] * load_int8(abase + 11)
              + kE2M1[(w1 >> 16) & 0xFu] * load_int8(abase + 12)
              + kE2M1[(w1 >> 20) & 0xFu] * load_int8(abase + 13)
              + kE2M1[(w1 >> 24) & 0xFu] * load_int8(abase + 14)
              + kE2M1[(w1 >> 28) & 0xFu] * load_int8(abase + 15);
        wsum += kE2M1[(w2      ) & 0xFu] * load_int8(abase + 16)
              + kE2M1[(w2 >>  4) & 0xFu] * load_int8(abase + 17)
              + kE2M1[(w2 >>  8) & 0xFu] * load_int8(abase + 18)
              + kE2M1[(w2 >> 12) & 0xFu] * load_int8(abase + 19)
              + kE2M1[(w2 >> 16) & 0xFu] * load_int8(abase + 20)
              + kE2M1[(w2 >> 20) & 0xFu] * load_int8(abase + 21)
              + kE2M1[(w2 >> 24) & 0xFu] * load_int8(abase + 22)
              + kE2M1[(w2 >> 28) & 0xFu] * load_int8(abase + 23);
        wsum += kE2M1[(w3      ) & 0xFu] * load_int8(abase + 24)
              + kE2M1[(w3 >>  4) & 0xFu] * load_int8(abase + 25)
              + kE2M1[(w3 >>  8) & 0xFu] * load_int8(abase + 26)
              + kE2M1[(w3 >> 12) & 0xFu] * load_int8(abase + 27)
              + kE2M1[(w3 >> 16) & 0xFu] * load_int8(abase + 28)
              + kE2M1[(w3 >> 20) & 0xFu] * load_int8(abase + 29)
              + kE2M1[(w3 >> 24) & 0xFu] * load_int8(abase + 30)
              + kE2M1[(w3 >> 28) & 0xFu] * load_int8(abase + 31);

        acc += (float)wsum * bs;
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
