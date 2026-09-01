#version 450

// 奇数 1023 次 XOR：真读 → acc = 0x5A5A5A5A；读被消除 → 0
layout(binding = 1) readonly buffer data_buf { uint data[]; };
layout(binding = 2) writeonly buffer out_buf { uint out_data[]; };

void main()
{
    const uint gid = gl_GlobalInvocationID.x;
    const uint base = gid * 1024u;
    uint acc = 0u;
    for (int k = 0; k < 1023; k++)
    {
        acc ^= data[base + uint(k)];
    }
    out_data[acc & 1023u] = acc;
}
