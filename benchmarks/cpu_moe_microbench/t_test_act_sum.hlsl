StructuredBuffer<int> act : register(t2);
RWStructuredBuffer<float> outv : register(u0);
groupshared float sh[256];
[numthreads(256, 1, 1)]
void main(uint3 gr : SV_GroupID, uint t : SV_GroupIndex) {
    if (gr.x != 0u) return;
    float acc = 0.0f;
    if (t < 32u) acc = (float)act[t];
    sh[t] = acc;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (t < s) sh[t] += sh[t + s];
        GroupMemoryBarrierWithGroupSync();
    }
    if (t == 0u) outv[0] = sh[0];
}
