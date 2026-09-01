RWStructuredBuffer<float> outv : register(u0);

[numthreads(1, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    outv[0] = 1.0f;
}
