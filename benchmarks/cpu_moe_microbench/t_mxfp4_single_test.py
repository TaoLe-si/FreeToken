"""Single-row minimal test."""
import numpy as np
import os
base = os.path.dirname(os.path.abspath(__file__))

# One row, K=32, packed = 4 uint = 16 bytes (we'll fill with simple data)
# 32 weights all = 1 (nibble 0001 → weight 1)
# 32 acts all = 1
# scale = 1 (byte = 127)
# bias = 0, gbl = 1
# Expected output = 32 * 1 * 1 * 1 + 0 + 0 = 32

M, K = 1, 32
nbPerRow = K // 8  # 4
nsPerRow = K // 32  # 1

packed = np.zeros((M, nbPerRow), dtype=np.uint32)
# Each uint32 packs 8 nibbles. For weight = 1, nibble = 1 (kE2M1[1] = 1).
# So each byte = 0x11, each uint = 0x11111111.
packed[:] = 0x11111111

# scl: 1 uint with byte 0 = 127 (scale=1), other bytes = 0
scl = np.zeros((M, nsPerRow), dtype=np.uint32)
scl[0, 0] = 127  # byte 0 = 127, rest = 0

# act: all 1
act = np.ones(K, dtype=np.int8)

# bias: 0
bias = np.zeros(M, dtype=np.float32)

# gbl: 1
gbl = np.ones(M, dtype=np.float32)

with open(os.path.join(base, "t_mxfp4_single_inputs.bin"), "wb") as f:
    f.write(b"SGL1")  # magic
    import struct
    f.write(struct.pack("II", M, K))
    f.write(packed.tobytes())
    f.write(scl.tobytes())
    f.write(act.tobytes())
    f.write(bias.tobytes())
    f.write(gbl.tobytes())

# Compute reference
sys_path = os.path.join(base, "t_mxfp4_gemv_reference.py")
import sys
sys.path.insert(0, base)
from importlib import reload
import t_mxfp4_gemv_reference
reload(t_mxfp4_gemv_reference)
ref = t_mxfp4_gemv_reference.mxfp4_gemv_reference(packed, scl, act, bias, gbl)
print(f"Expected = 32, ref = {ref}")
