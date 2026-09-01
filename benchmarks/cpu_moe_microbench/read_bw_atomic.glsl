#version 450

// 原子写：真副作用，LLPC 不可消除 → 读链保留
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
    atomicXor(out_data[acc & 1023u], 1u);
}
