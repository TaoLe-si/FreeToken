// MTP MoE combine: out[k] = shared_out[k] + sum_e top8_w[e] * expert_out[e, k]
StructuredBuffer<float>   c_expert_out  : register(t0);
StructuredBuffer<float>   c_top8_w      : register(t1);
StructuredBuffer<float>   c_shared_out  : register(t2);
RWStructuredBuffer<float> c_final_out   : register(u0);
cbuffer CombCB : register(b0) { uint H; };

[numthreads(512, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID, uint t : SV_GroupIndex,
         uint3 gr : SV_GroupID) {
    uint k = gr.x * 512 + t;
    if (k >= H) return;
    float acc = c_shared_out[k];
    for (uint e = 0; e < 8; e++) {
        acc += c_top8_w[e] * c_expert_out[e * H + k];
    }
    c_final_out[k] = acc;
}
