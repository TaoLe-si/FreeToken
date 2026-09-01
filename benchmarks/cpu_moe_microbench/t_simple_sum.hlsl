// Minimal test: just sum 32 act values.
StructuredBuffer<int> act : register(t2);
RWStructuredBuffer<float> outv : register(u0);
[numthreads(256, 1, 1)]
void main(uint3 gr : SV_GroupID, uint t : SV_GroupIndex) {
    groupshared float sh[256];
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
