// Minimal hello-world D3D12 compute shader.
// One thread, one group, write constant to UAV.
RWStructuredBuffer<uint> outv : register(u0);
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    outv[0] = 42u;
}
