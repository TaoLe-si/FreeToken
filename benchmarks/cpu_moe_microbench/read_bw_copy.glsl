#version 450

// copy 模式：每项读 1 uint 写 1 uint（与 vkpeak copy-d2d 同型）
layout(binding = 1) readonly buffer data_buf { uint data[]; };
layout(binding = 2) writeonly buffer out_buf { uint out_data[]; };

void main()
{
    const uint gid = gl_GlobalInvocationID.x;
    out_data[gid & 262143u] = data[gid];
}
