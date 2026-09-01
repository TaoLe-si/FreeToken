
// Test: read packed[0], extract nibble 0, output as float
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
        uint w = packed[row * nbPerRow];
        int nibble = kE2M1[w & 0xFu];
        outv[row] = (float)nibble;
    }
}
