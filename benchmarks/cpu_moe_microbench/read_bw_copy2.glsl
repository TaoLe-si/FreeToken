#version 450

// copy 读1写1；out 不带 writeonly
layout(binding = 1) readonly buffer data_buf { uint data[]; };
layout(binding = 2) buffer out_buf { uint out_data[]; };

void main()
{
    const uint gid = gl_GlobalInvocationID.x;
    out_data[gid & 65535u] = data[gid];
}
