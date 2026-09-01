// MTP attn Q@K^T + softmax + AV + sigmoid gate.
StructuredBuffer<float>   at_qg       : register(t0);
StructuredBuffer<float>   at_kc       : register(t1);
StructuredBuffer<float>   at_vc       : register(t2);
RWStructuredBuffer<float> at_gate_out : register(u0);
cbuffer AttnCB : register(b0) { uint QO; uint KV; uint HD; uint KV_LEN; uint KV_MAX; };

groupshared float probs[4096];   // max kv_len
groupshared float scale_sh;
groupshared float maxv_sh;
groupshared float esum_sh;

[numthreads(128, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID, uint t : SV_GroupIndex,
         uint3 gr : SV_GroupID) {
    uint h = gr.x;             // 0..15
    uint kv_h = h / 8;
    if (t == 0) scale_sh = 1.0f / sqrt((float)128);
    GroupMemoryBarrierWithGroupSync();
    // Phase 1: scores[i] = sum_d qg[h, d] * kc[kv_h, i, d]
    for (uint i = t; i < KV_LEN; i += 128) {
        float s = 0.0f;
        uint kbase = kv_h * KV_MAX * 128 + i * 128;
        for (uint d = 0; d < 128; d += 4) {
            float4 q4 = float4(at_qg[h*128+d], at_qg[h*128+d+1], at_qg[h*128+d+2], at_qg[h*128+d+3]);
            float4 k4 = float4(at_kc[kbase+d], at_kc[kbase+d+1], at_kc[kbase+d+2], at_kc[kbase+d+3]);
            s += dot(q4, k4);
        }
        probs[i] = s * scale_sh;
    }
    GroupMemoryBarrierWithGroupSync();

    // Phase 2: softmax. Single-thread serial (slow but correct for skeleton).
    if (t == 0) {
        float maxv = -1e30f;
        for (uint i = 0; i < KV_LEN; i++) if (probs[i] > maxv) maxv = probs[i];
        float esum = 0.0f;
        for (uint i = 0; i < KV_LEN; i++) { probs[i] = exp(probs[i] - maxv); esum += probs[i]; }
        for (uint i = 0; i < KV_LEN; i++) probs[i] /= esum;
    }
    GroupMemoryBarrierWithGroupSync();

    // Phase 3: out[d] = sum_i probs[i] * vc[kv_h, i, d], then apply sigmoid gate.
    for (uint d = t; d < 128; d += 128) {
        float acc = 0.0f;
        for (uint i = 0; i < KV_LEN; i++) {
            acc += probs[i] * at_vc[kv_h * KV_MAX * 128 + i * 128 + d];
        }
        // gate: sigmoid(qg[h, 128+d]) * acc (qg has q in [0,128), gate in [128,256))
        float g = 1.0f / (1.0f + exp(-at_qg[h*256 + 128 + d]));
        at_gate_out[h * 128 + d] = g * acc;
    }
}
