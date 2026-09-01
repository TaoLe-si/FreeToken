#version 450

// 诊断：每项读 1024 uint (4KB)，写固定位置便于 CPU 校验
layout(binding = 1) readonly buffer data_buf { uint data[]; };
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
    out_data[0] = acc;
}
