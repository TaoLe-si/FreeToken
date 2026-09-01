"""Debug: inspect row 0, micro-block 0 of D3D12 inputs."""
import numpy as np
import os
base = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(base, "t_mxfp4_inputs.bin"), "rb") as f:
    f.read(4)  # magic
    import struct
    M, K = struct.unpack("II", f.read(8))
    nbPerRow = K // 8
    nsPerRow = K // 32
    packed = np.frombuffer(f.read(M * nbPerRow * 4), dtype=np.uint32).reshape(M, nbPerRow)
    scl    = np.frombuffer(f.read(M * nsPerRow * 4), dtype=np.uint32).reshape(M, nsPerRow)
    act    = np.frombuffer(f.read(K), dtype=np.int8)
    bias   = np.frombuffer(f.read(M * 4), dtype=np.float32)
    gbl    = np.frombuffer(f.read(M * 4), dtype=np.float32)

# Row 0 micro-block 0:
print(f"M={M} K={K} nbPerRow={nbPerRow} nsPerRow={nsPerRow}")
print(f"row 0 packed[0]  = 0x{packed[0,0]:08x}")
print(f"row 0 scl[0]     = 0x{scl[0,0]:08x} (byte0={scl[0,0] & 0xFF}, byte1={(scl[0,0]>>8)&0xFF})")
print(f"row 0 scl[0] byte 0 (micro-block 0) = {scl[0,0] & 0xFF}  → exp2({(scl[0,0] & 0xFF) - 127})")
print(f"row 0 act[:32]  = {act[:32].tolist()}")
print(f"row 0 bias      = {bias[0]}")
print(f"row 0 gbl       = {gbl[0]}")

# Manual micro-block 0 GEMV:
packed_row = packed[0]
packed_bytes = packed_row.view(np.uint8)
# Micro-block 0: 4 uints starting at uint 0
mb_packed = packed_bytes[:16]  # 16 bytes = 4 uints
print(f"micro-block 0 packed bytes = {list(mb_packed)}")

# Decode nibbles
kE2M1 = [0,1,2,3,4,6,8,12, 0,-1,-2,-3,-4,-6,-8,-12]
wsum = 0
for j in range(4):
    w = int.from_bytes(mb_packed[j*4:j*4+4].tobytes(), 'little')  # not right, mb_packed is bytes
# Simpler: 16 bytes = 16 nibble pairs
nibbles = []
for byte in mb_packed:
    nibbles.append(byte & 0xF)
    nibbles.append((byte >> 4) & 0xF)
print(f"32 nibbles = {nibbles}")
weights = [kE2M1[n] for n in nibbles]
print(f"32 weights = {weights}")

# 32 act values
acts = act[:32].astype(np.int32)
print(f"32 acts = {acts.tolist()}")

# weight * act sum
prod_sum = sum(w*a for w,a in zip(weights, acts))
print(f"prod_sum (before scale) = {prod_sum}")

# scale = exp2(byte0 - 127)
scale = 2.0 ** ((scl[0,0] & 0xFF) - 127)
print(f"scale = {scale}")

mb_sum = prod_sum * scale
print(f"micro-block 0 contribution = {mb_sum}")

# bias + gbl applied once
out = (mb_sum + bias[0]) * gbl[0]
print(f"row 0 final (only mb 0) = {out}")
print(f"NOTE: this only accounts for micro-block 0, not all 128 micro-blocks")
