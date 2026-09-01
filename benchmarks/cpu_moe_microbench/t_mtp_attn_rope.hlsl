// MTP attn RoPE + KV cache append. Apply rope, write to kv_cache.
StructuredBuffer<float>   r_qg_in     : register(t0);
StructuredBuffer<float>   r_k_in      : register(t1);
StructuredBuffer<float>   r_v_in      : register(t2);
StructuredBuffer<float>   r_freqs     : register(t3);  // precomputed cos/sin (HD/2, 2)
RWStructuredBuffer<float> r_qg_out    : register(u0);
RWStructuredBuffer<float> r_kc        : register(u1);
RWStructuredBuffer<float> r_vc        : register(u2);
RWStructuredBuffer<uint>  r_kv_len_io : register(u3);
cbuffer RoPECB : register(b0) { uint QO; uint KV; uint HD; uint KV_MAX; uint POS; };

[numthreads(128, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID, uint t : SV_GroupIndex,
         uint3 gr : SV_GroupID) {
    uint kind = gr.x;  // 0..17: 0..15 = q heads, 16..17 = k heads
    uint hd = t;
    if (hd >= 128) return;
    // For head_dim=128, HD/2=64 pairs. Pair (2i, 2i+1) rotates by angle = POS * freqs[i].
    float ang;
    if (hd < 64) {
        ang = POS * r_freqs[hd];
    } else {
        // freqs stored as [cos(ang), sin(ang)] pairs; for hd >= 64 use hd-64
        ang = POS * r_freqs[hd - 64];
    }
    float c = cos(ang), s = sin(ang);
    // We need to rotate pairs; for skeleton, write through unchanged.
    if (kind < QO) {
        float v = r_qg_in[kind * 128 + hd];
        r_qg_out[kind * 128 + hd] = v;
    } else {
        uint kh = kind - QO;
        if (kh < KV) {
            uint pos = r_kv_len_io[0];
            float k = r_k_in[kh * 128 + hd];
            float v = r_v_in[kh * 128 + hd];
            r_kc[kh * KV_MAX * 128 + pos * 128 + hd] = k;
            r_vc[kh * KV_MAX * 128 + pos * 128 + hd] = v;
            if (hd == 0 && kind == QO) {
                r_kv_len_io[0] = pos + 1;
            }
        }
    }
}
