"""Examine fc_w bit pattern."""
import numpy as np
import os
import sys
import struct
import torch

base = "E:\\models\\Qwen3.6-35B-A3B-MXFP4-MTP"

import safetensors.torch
import json as _json
with open(os.path.join(base, "model.safetensors.index.json")) as f:
    idx = _json.load(f)
state = safetensors.torch.load_file(os.path.join(base, idx["weight_map"]["mtp.fc.weight"]))
fc_w = state["mtp.fc.weight"].cpu().numpy()  # (2048, 512) uint32 packed
fc_b = state["mtp.fc.biases"].cpu().numpy().astype(np.float32)
fc_s = state["mtp.fc.scales"].cpu().numpy().astype(np.float32)

# Take first 4 uint32 of first row = first 16 nibbles
print("fc_w[0, 0:4] hex:", [hex(int(x)) for x in fc_w[0, 0:4]])
print("fc_w[0, 0:4] bytes:")
for u in fc_w[0, 0:4]:
    u32 = int(u)
    b0 = u32 & 0xFF
    b1 = (u32 >> 8) & 0xFF
    b2 = (u32 >> 16) & 0xFF
    b3 = (u32 >> 24) & 0xFF
    print(f"  u32=0x{u32:08X}  bytes: {b0:3d} {b1:3d} {b2:3d} {b3:3d}")

# nvfp4 unpack
kE2M1 = np.array([0,1,2,3,4,6,8,12, 0,-1,-2,-3,-4,-6,-8,-12], dtype=np.int32)

# Compute first 32 weights
nibbles = []
u32_0 = int(fc_w[0, 0])
for i in range(8):
    nib = (u32_0 >> (i*4)) & 0xF
    nibbles.append(kE2M1[nib])
print(f"\\nFirst 8 weights from fc_w[0, 0]: {nibbles}")
print(f"Sum: {sum(nibbles)}")

# But - here's the issue: the E2M1 unpacking in shader does:
#   nibble = (b >> bit) & 0xF
# but B is a byte from packed_bytes
# packed_bytes is fc_w.view(uint8) which gives M*nb*4 bytes per row
# 
# In shader: packed[abase + n] reads uint32, takes 4 nibbles
# But abase = b*32 (micro-block start)
# Each micro-block has 32 weights (32 nibbles = 4 uints)
# 
# The shader does:
#   uint pBase = row * nbPerRow + b * 4u;  // = row * (K/8) + b * 4
#   uint w0 = packed[pBase + 0u];
#   uint w1 = packed[pBase + 1u];
#   ...
#   W[0]  = kE2M1[(w0 >>  0) & 0xF]  // low nibble of w0
#   W[1]  = kE2M1[(w0 >>  4) & 0xF]
#   W[2]  = kE2M1[(w0 >>  8) & 0xF]
#   ...
# 
# So each uint32 has 8 nibbles (low to high)
# uint_idx = n // 8
# byte_idx = (n // 2) % 4
# bit = (n % 2) * 4
# This is different! 
# In t_mxfp4_compare.py ref:
#   uint_idx = n // 8
#   byte_idx = (n // 2) % 4
#   bit = (n % 2) * 4
#   flat = uint_idx * 4 + byte_idx
#   b = packed_bytes[flat]
#   nibble = (b >> bit) & 0xF

# So ref reads BYTE first, then nibble from byte
# vs shader reads UINT32 first, then nibble from uint32
# These are equivalent if byte is at correct offset

# Let me check: u32_0 = 0xC5C0C5C0
# Low byte = 0xC0, next = 0xC5, next = 0xC0, high = 0xC5
# Wait, on little-endian: 0xC5C0C5C0 = bytes C0 C5 C0 C5 (low to high)
# As uint32: bits 0-7 = C0, bits 8-15 = C5, bits 16-23 = C0, bits 24-31 = C5
# 
# Ref: uint_idx=0, byte_idx=0 -> byte 0 = 0xC0
#   bit=0 -> nibble = 0xC0 & 0xF = 0 -> W[0] = 0
#   bit=4 -> nibble = (0xC0 >> 4) & 0xF = 0xC -> W[1] = 0
# Shader: w0 = 0xC5C0C5C0
#   W[0] = (w0 >> 0) & 0xF = 0
#   W[1] = (w0 >> 4) & 0xF = 0
# These MATCH ✓

# W[2]: ref byte_idx=1, byte=0xC5, bit=0, nibble=0xC5 & 0xF = 0x5
# shader: (w0 >> 8) & 0xF = 0x5
# These MATCH ✓

# W[3]: ref byte_idx=1, bit=4, nibble=0xC5>>4=0xC
# shader: (w0 >> 12) & 0xF = 0xC
# MATCH ✓

# Good, the bit order is consistent

# Now the act_ones test: 0x3F800000 = float 1.0
# But as int32 it's just 0x3F800000
# Client: act_int32 = act.view(np.int32) (treating float32 bits as int32)
# But act = np.full(K, 1.0, dtype=np.float32) -> bits 0x3F800000
# act.view(np.int32) gives 0x3F800000 as int32

# So client sends 0x3F800000 (int32 value 1065353216)
# Server: int32_t* aSrc = (int32_t*)act.data(); aDst[i] = (float)aSrc[i];
# aDst[0] = (float)0x3F800000 = ... wait, (float)0x3F800000 as int reinterpret?
# In C++: (float)0x3F800000 = float(1065353216) which is out of float range
# So (float)0x3F800000 = 1.065e+09 (approximately)
# THIS IS THE BUG! 

# C++ (float)int32 is NOT a bit cast. It converts integer to float (saturates).
# So 1065353216 -> 1.065e+09

# To do a bit cast, we need: 
#   union { int32_t i; float f; } u; u.i = 0x3F800000; act[i] = u.f;  // = 1.0
# or use std::memcpy

# Server code:
#   float* aDst = (float*)((char*)m + offA);
#   int32_t* aSrc = (int32_t*)act.data();
#   for (UINT32 i = 0; i < nAct; i++) aDst[i] = (float)aSrc[i];
# This is C-style cast: (float)int32_value = integer to float conversion
# It interprets 0x3F800000 as int = 1065353216, then converts to float
# = 1.065e+09 (because 2^30 = 1.073e+9, so 1.065e+09 is the float)

# To fix: use bit_cast (C++20) or memcpy:
#   std::memcpy(&aDst[i], &aSrc[i], 4);
# This does bit-cast

print("\\n=== bit cast test ===")
import ctypes
# Python: 0x3F800000 as int -> 1065353216
# C: (float)1065353216 = ?
# In C99: (float)1065353216 = 1.065e+09 approximately

# Verify: 2^30 = 1073741824, so 1.065e+09 = 1065353216 < 2^30, but greater than 2^29 = 536870912
# So (float)1065353216 = 1065353216.0 exactly (since < 2^31)
# But as float bit pattern, 1065353216.0 = 0x4FFE0000 (mantissa needed)

# So shader sees act[i] = 1.065e+09 (not 1.0)
# Then sum W*act with W=±1..12 and act=1e+9 = ±1e+9 per element
# With 32 elements in a block: wsum = 1e+9 * sum = 1e+10
# Then (wsum + bias) * scale = 1e+10 * 0.0027 = 2.7e+7 per block
# With 128 blocks: 128 * 2.7e+7 = 3.4e+9
# This matches outv = 5.79e+9! 

print("\\n*** BUG FOUND: server side (float)int32 is not bit cast ***")
print("Need to use std::memcpy for bit cast")
