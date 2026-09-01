RWStructuredBuffer<uint> outv : register(u0);
StructuredBuffer<uint> packed : register(t0);
StructuredBuffer<uint> scl : register(t1);
StructuredBuffer<int> act : register(t2);
StructuredBuffer<float> bias : register(t3);
StructuredBuffer<float> gbl : register(t4);
cbuffer P : register(b0) { uint K; uint nbPerRow; uint nsPerRow; uint pad; }
[numthreads(256, 1, 1)]
void main(uint3 gr : SV_GroupID, uint t : SV_GroupIndex) {
    outv[gr.x] = (uint)(gr.x * 10u + t);
}
