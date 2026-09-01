RWStructuredBuffer<float> outv : register(u0);
cbuffer P : register(b0) { uint K; uint nbPerRow; float g; float pad; }
[numthreads(256, 1, 1)]
void main(uint3 g : SV_DispatchThreadID) {
    outv[g.x] = float(g.x) * g;
}
