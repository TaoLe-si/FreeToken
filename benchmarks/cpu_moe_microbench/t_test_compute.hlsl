
// Test: compute one micro-block (b=0) and output
StructuredBuffer<uint>    packed  : register(t0);
StructuredBuffer<float>   scl     : register(t1);
StructuredBuffer<float>   bias_pb : register(t2);
StructuredBuffer<float>   act     : register(t3);
StructuredBuffer<float>   gbl     : register(t4);
StructuredBuffer<float>   rowBias : register(t5);
RWStructuredBuffer<float> outv    : register(u0);
cbuffer P : register(b0) { uint K; uint nbPerRow; uint nsPerRow; };

static const int kE2M1[16] = {0,1,2,3,4,6,8,12, 0,-1,-2,-3,-4,-6,-8,-12};

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID, uint t : SV_GroupIndex, uint3 gr : SV_GroupID) {
    uint row = gr.x;
    float result = 0.0f;
    if (t == 0) {
        uint b = 0;
        uint pbase = row * nbPerRow + b;
        uint sbase = row * nsPerRow + b;
        float bs2 = scl[sbase];
        float bb = bias_pb[sbase];
        uint words[4];
        words[0] = packed[pbase * 4u + 0u];
        words[1] = packed[pbase * 4u + 1u];
        words[2] = packed[pbase * 4u + 2u];
        words[3] = packed[pbase * 4u + 3u];
        float wsum = 0.0f;
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
            float a0 = act[row * K + ai + 0u];
            float a1 = act[row * K + ai + 1u];
            float a2 = act[row * K + ai + 2u];
            float a3 = act[row * K + ai + 3u];
            float a4 = act[row * K + ai + 4u];
            float a5 = act[row * K + ai + 5u];
            float a6 = act[row * K + ai + 6u];
            float a7 = act[row * K + ai + 7u];
            wsum += w0 * a0 + w1 * a1 + w2 * a2 + w3 * a3
                     + w4 * a4 + w5 * a5 + w6 * a6 + w7 * a7;
        }
        result = (wsum + bb) * bs2;
        outv[row] = result;
    }
}
