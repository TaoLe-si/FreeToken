StructuredBuffer<uint>    packed  : register(t0);
StructuredBuffer<float>   scl     : register(t1);
StructuredBuffer<float>   bias_pb : register(t2);
StructuredBuffer<float>   act     : register(t3);
StructuredBuffer<float>   gbl     : register(t4);
StructuredBuffer<float>   rowBias : register(t5);
RWStructuredBuffer<float> outv    : register(u0);
cbuffer P : register(b0) { uint K; uint nbPerRow; uint nsPerRow; };

int getE2M1(uint nibble) {
    if (nibble == 0) return 0;
    if (nibble == 1) return 1;
    if (nibble == 2) return 2;
    if (nibble == 3) return 3;
    if (nibble == 4) return 4;
    if (nibble == 5) return 6;
    if (nibble == 6) return 8;
    if (nibble == 7) return 12;
    if (nibble == 8) return 0;
    if (nibble == 9) return -1;
    if (nibble == 10) return -2;
    if (nibble == 11) return -3;
    if (nibble == 12) return -4;
    if (nibble == 13) return -6;
    if (nibble == 14) return -8;
    if (nibble == 15) return -12;
    return 0;
}

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID, uint t : SV_GroupIndex, uint3 gr : SV_GroupID) {
    uint row = gr.x;
    if (t == 0) {
        uint b = 0;
        uint pbase = row * nbPerRow + b;
        uint w = packed[pbase];
        outv[0] = (float)w;
        uint n0 = w & 0xFu;
        uint n1 = (w >> 4) & 0xFu;
        outv[1] = (float)n0;
        outv[2] = (float)n1;
        outv[3] = (float)getE2M1(n0);
        outv[4] = (float)getE2M1(n1);
        outv[5] = (float)(w & 0xF);
        outv[6] = (float)((w >> 4) & 0xF);
        outv[7] = act[row * K + 0];
        outv[8] = act[row * K + 1];
        outv[9] = act[row * K + 2];
        outv[10] = act[row * K + 3];
    }
}
