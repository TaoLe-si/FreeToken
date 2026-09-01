RWStructuredBuffer<float> outv : register(u0);
[numthreads(256, 1, 1)]
void main(uint3 g : SV_DispatchThreadID) {
    outv[g.x] = float(g.x) * 1.0001 + 0.5;
}
