#version 450

// volatile 阻止 LLPC 消除读；每项 4KB
layout(binding = 1) volatile readonly buffer data_buf { uint data[]; };
layout(binding = 2) writeonly buffer out_buf { uint out_data[]; };

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
