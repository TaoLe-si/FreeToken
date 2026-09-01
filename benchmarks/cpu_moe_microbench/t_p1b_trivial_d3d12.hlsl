RWStructuredBuffer<float> outv : register(u0);
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    outv[0] = 32.0f;
}
