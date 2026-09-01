// MTP MoE routing kernel: 256 logits -> top-8 (idx, weights via softmax).
StructuredBuffer<float>   r_hidden    : register(t0);
StructuredBuffer<float>   r_router_w  : register(t1);
RWStructuredBuffer<uint>  r_top8_idx  : register(u0);
RWStructuredBuffer<float> r_top8_w    : register(u1);
cbuffer RouteCB : register(b0) { uint H; uint E; };

groupshared float logits_sh[256];
groupshared uint  idx_sh[256];

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID, uint t : SV_GroupIndex) {
    uint e = t;
    if (e < E) {
        float acc = 0.0f;
        uint base = e * H;
        for (uint k = 0; k < H; k += 4) {
            float4 h4 = float4(r_hidden[k], r_hidden[k+1], r_hidden[k+2], r_hidden[k+3]);
            float4 w4 = float4(r_router_w[base+k], r_router_w[base+k+1], r_router_w[base+k+2], r_router_w[base+k+3]);
            acc += dot(h4, w4);
        }
        logits_sh[e] = acc;
        idx_sh[e] = e;
    }
    GroupMemoryBarrierWithGroupSync();

    // Bitonic sort top-8 in descending order.
    // For E=256: log2(256)=8 stages, each stage has sub-stages.
    [unroll] for (uint k = 2; k <= 256; k <<= 1) {
        [unroll] for (uint j = k >> 1; j > 0; j >>= 1) {
            for (uint i = t; i < 256; i += 256) {
                uint ixj = i ^ j;
                if (ixj > i) {
                    bool asc = ((i & k) == 0);
                    float li = logits_sh[i], lj = logits_sh[ixj];
                    bool swap = asc ? (li < lj) : (li > lj);
                    if (swap) {
                        float tl = logits_sh[i]; logits_sh[i] = logits_sh[ixj]; logits_sh[ixj] = tl;
                        uint  ti = idx_sh[i]; idx_sh[i] = idx_sh[ixj]; idx_sh[ixj] = ti;
                    }
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }
    }
    // After sort, logits_sh is descending [0..255]. Top-8 = [0..7].
    if (t < 8) {
        r_top8_idx[t] = idx_sh[t];
        float maxv = logits_sh[0];
        float esum = 0.0f;
        for (uint i = 0; i < 8; i++) esum += exp(logits_sh[i] - maxv);
        r_top8_w[t] = exp(logits_sh[t] - maxv) / esum;
    }
}
