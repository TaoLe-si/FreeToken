#version 450

// 纯读 4KB/项；out 不带 writeonly → 写有观测 → 读写链保留
layout(binding = 1) readonly buffer data_buf { uint data[]; };
layout(binding = 2) buffer out_buf { uint out_data[]; };

void main()
{
    const uint gid = gl_GlobalInvocationID.x;
    const uint base = gid * 1024u;
    uint acc = 0u;
    for (int k = 0; k < 1024; k++)
    {
        acc ^= data[base + uint(k)];
    }
    out_data[acc & 1023u] = acc;
}
