// MTP attn o_proj: out[k] = sum_d gate_in[d] * o_w[k, d]
StructuredBuffer<float>   op_gate_in : register(t0);
StructuredBuffer<float>   op_o_w     : register(t1);
RWStructuredBuffer<float> op_out     : register(u0);
cbuffer OpCB : register(b0) { uint H; };

[numthreads(512, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID, uint t : SV_GroupIndex,
         uint3 gr : SV_GroupID) {
    uint k = gr.x * 512 + t;
    if (k >= H) return;
    float acc = 0.0f;
    uint base = k * H;
    for (uint d = 0; d < H; d += 4) {
        float4 g4 = float4(op_gate_in[d], op_gate_in[d+1], op_gate_in[d+2], op_gate_in[d+3]);
        float4 w4 = float4(op_o_w[base+d], op_o_w[base+d+1], op_o_w[base+d+2], op_o_w[base+d+3]);
        acc += dot(g4, w4);
    }
    op_out[k] = acc;
}
