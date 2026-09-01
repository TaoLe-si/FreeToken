// MTP MoE 8-expert SwiGLU: one dispatch, 8 thread groups (one per selected expert).
StructuredBuffer<float>   e_hidden       : register(t0);
StructuredBuffer<float>   e_expert_gate  : register(t1);
StructuredBuffer<float>   e_expert_up    : register(t2);
StructuredBuffer<float>   e_expert_down  : register(t3);
StructuredBuffer<uint>    e_top8_idx     : register(t4);
RWStructuredBuffer<float> e_expert_out   : register(u0);
cbuffer ExpertCB : register(b0) { uint H; uint I; };

groupshared float gate_act[512];
groupshared float up_act[512];

[numthreads(512, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID, uint t : SV_GroupIndex,
         uint3 gr : SV_GroupID) {
    uint e_grp = gr.x;          // 0..7
    if (e_grp >= 8) return;
    uint e = e_top8_idx[e_grp]; // actual expert idx
    float g = 0.0f, u = 0.0f;
    uint gbase = e * I * H + t * H;
    uint ubase = e * I * H + t * H;
    for (uint k = 0; k < H; k += 4) {
        float4 h4 = float4(e_hidden[k], e_hidden[k+1], e_hidden[k+2], e_hidden[k+3]);
        float4 g4 = float4(e_expert_gate[gbase+k], e_expert_gate[gbase+k+1], e_expert_gate[gbase+k+2], e_expert_gate[gbase+k+3]);
        float4 u4 = float4(e_expert_up  [ubase+k], e_expert_up  [ubase+k+1], e_expert_up  [ubase+k+2], e_expert_up  [ubase+k+3]);
        g += dot(h4, g4);
        u += dot(h4, u4);
    }
    gate_act[t] = g;
    up_act[t] = u;
    GroupMemoryBarrierWithGroupSync();

    // Phase 2: silu(gate) * up -> store in up_act.
    if (t < I) {
        float x = gate_act[t];
        float silu = x / (1.0f + exp(-x));
        up_act[t] = silu * up_act[t];
    }
    GroupMemoryBarrierWithGroupSync();

    // Phase 3: out[e_grp, k] = sum_i up_act[i] * expert_down[e, k, i].
    for (uint k = t; k < H; k += 512) {
        float acc = 0.0f;
        uint dbase = e * H * I + k * I;
        for (uint i = 0; i < I; i++) {
            acc += up_act[i] * e_expert_down[dbase + i];
        }
        e_expert_out[e_grp * H + k] = acc;
    }
}
