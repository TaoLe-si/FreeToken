StructuredBuffer<float> asb : register(t3);
RWStructuredBuffer<float> outv : register(u0);
cbuffer P : register(b0) { uint K; uint nbPerRow; float g; float pad; }
[numthreads(256, 1, 1)]
void main(uint3 g : SV_DispatchThreadID) {
    float acc = 0.0;
    for (uint b = 0; b < nbPerRow; b++) acc += asb[b];
    outv[g.x] = acc * g;
}
