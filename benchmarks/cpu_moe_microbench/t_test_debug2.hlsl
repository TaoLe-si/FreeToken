
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
    if (t == 0) {
        uint b = 0;
        uint pbase = row * nbPerRow + b;
        
        uint w = packed[pbase];  // packed[0]
        
        // Output the raw uint as float
        outv[0] = (float)w;
        
        // Output each nibble value
        outv[1] = (float)kE2M1[w        & 0xFu];  // nibble 0
        outv[2] = (float)kE2M1[(w >>  4) & 0xFu];  // nibble 1
        outv[3] = (float)kE2M1[(w >>  8) & 0xFu];  // nibble 2
        outv[4] = (float)kE2M1[(w >> 12) & 0xFu];  // nibble 3
        outv[5] = (float)kE2M1[(w >> 16) & 0xFu];  // nibble 4
        outv[6] = (float)kE2M1[(w >> 20) & 0xFu];  // nibble 5
        outv[7] = (float)kE2M1[(w >> 24) & 0xFu];  // nibble 6
        outv[8] = (float)kE2M1[(w >> 28) & 0xFu];  // nibble 7
        
        // Output act[0..3]
        outv[9]  = act[row * K + 0];
        outv[10] = act[row * K + 1];
        outv[11] = act[row * K + 2];
        outv[12] = act[row * K + 3];
    }
}
