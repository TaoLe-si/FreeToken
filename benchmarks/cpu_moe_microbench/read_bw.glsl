#version 450

// SM 读带宽测试：每 work-item 顺序读 256 uint (1KB)，XOR 累加，写 1 个 uint 防优化
layout(binding = 1) readonly buffer data_buf { uint data[]; };
layout(binding = 2) writeonly buffer out_buf { uint out_data[]; };

void main()
{
    const uint gid = gl_GlobalInvocationID.x;
    const uint base = gid * 256u;
    uint acc = 0u;
    for (int k = 0; k < 256; k++)
    {
        acc ^= data[base + uint(k)];
    }
    out_data[gid & 1023u] = acc;
}
