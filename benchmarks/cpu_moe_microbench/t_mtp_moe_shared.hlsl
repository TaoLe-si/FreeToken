// MTP MoE shared expert SwiGLU + sigmoid gate.
StructuredBuffer<float>   sh_hidden     : register(t0);
StructuredBuffer<float>   sh_sgate_w    : register(t1);
StructuredBuffer<float>   sh_sup_w      : register(t2);
StructuredBuffer<float>   sh_sdown_w    : register(t3);
StructuredBuffer<float>   sh_sgw        : register(t4);  // (1,) bf16 linear gate
RWStructuredBuffer<float> sh_out        : register(u0);
cbuffer SharedCB : register(b0) { uint H; uint I; };

groupshared float sg_act[512];
groupshared float su_act[512];
groupshared float sg_scalar_sh;

[numthreads(512, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID, uint t : SV_GroupIndex) {
    float g = 0.0f, u = 0.0f;
    for (uint k = 0; k < H; k += 4) {
        float4 h4 = float4(sh_hidden[k], sh_hidden[k+1], sh_hidden[k+2], sh_hidden[k+3]);
        float4 g4 = float4(sh_sgate_w[t*H+k], sh_sgate_w[t*H+k+1], sh_sgate_w[t*H+k+2], sh_sgate_w[t*H+k+3]);
        float4 u4 = float4(sh_sup_w  [t*H+k], sh_sup_w  [t*H+k+1], sh_sup_w  [t*H+k+2], sh_sup_w  [t*H+k+3]);
        g += dot(h4, g4);
        u += dot(h4, u4);
    }
    sg_act[t] = g;
    su_act[t] = u;
    GroupMemoryBarrierWithGroupSync();

    if (t < I) {
        float x = sg_act[t];
        float silu = x / (1.0f + exp(-x));
        su_act[t] = silu * su_act[t];
    }
    GroupMemoryBarrierWithGroupSync();

    // out[k] = sum_i su_act[i] * sh_sdown_w[k, i]
    for (uint k = t; k < H; k += 512) {
        float acc = 0.0f;
        for (uint i = 0; i < I; i++) {
            acc += su_act[i] * sh_sdown_w[k * I + i];
        }
        sh_out[k] = acc;
    }

    // Sigmoid gate: scalar = sigmoid(sum_k sh_sgw[k] * sh_hidden[k])
    if (t == 0) {
        float s = 0.0f;
        for (uint k = 0; k < H; k++) s += sh_sgw[k] * sh_hidden[k];
        sg_scalar_sh = 1.0f / (1.0f + exp(-s));
    }
    GroupMemoryBarrierWithGroupSync();

    // Apply gate.
    for (uint k = t; k < H; k += 512) {
        sh_out[k] *= sg_scalar_sh;
    }
}
