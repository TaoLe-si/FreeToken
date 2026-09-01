cbuffer Params : register(b0) {
    uint totalUint;
    uint stride;
    uint pad0;
    uint pad1;
};
RWStructuredBuffer<uint> dummy : register(u0);
StructuredBuffer<uint> data : register(t0);

[numthreads(256, 1, 1)]
void main(uint3 gid3 : SV_DispatchThreadID) {
    uint gid = gid3.x;
    uint per = (totalUint + stride - 1) / stride;
    uint start = gid * per;
    uint end = min(start + per, totalUint);
    uint acc = 0;
    for (uint i = start; i < end; i += 8) {
        acc ^= data[i] ^ data[i+1] ^ data[i+2] ^ data[i+3]
             ^ data[i+4] ^ data[i+5] ^ data[i+6] ^ data[i+7];
    }
    if (acc == 0xDEADBEEFu) dummy[gid & 0xFFFFu] = acc;
}
