RWStructuredBuffer<float> outv : register(u0);
[numthreads(256, 1, 1)]
void main(uint3 gr : SV_GroupID, uint t : SV_GroupIndex) {
    if (t == 0u) {
        outv[gr.x] = (float)gr.x * 100.0f + 7.0f;
    }
}
