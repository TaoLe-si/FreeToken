StructuredBuffer<uint2> packed : register(t0);
RWStructuredBuffer<float> outv : register(u0);
cbuffer P : register(b0) { uint K; uint nbPerRow; float gs; float pad; }
[numthreads(256, 1, 1)]
void main(uint3 g : SV_DispatchThreadID) {
    uint2 v = packed[g.x];
    outv[g.x] = (float)(v.x & 0xFFu) + (float)(v.y & 0xFFu);
}
