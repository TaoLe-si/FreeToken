import os, struct, numpy as np
out = "E:\\FreeToken\\benchmarks\\cpu_moe_microbench"

# All-1 test
w_packed = np.array([0x11111111, 0x11111111, 0x11111111, 0x11111111], dtype=np.uint32)
b_one = np.array([0.0], dtype=np.float32)
s_one = np.array([1.0], dtype=np.float32)
act = np.ones(32, dtype=np.float32)
M, K = 1, 32
nbPerRow = K // 8  # = 4 — CORRECT
nsPerRow = K // 32  # = 1
print(f"nbPerRow={nbPerRow}, nsPerRow={nsPerRow}")
with open(out + "/t_p1b_all1.bin", "wb") as f:
    f.write(struct.pack("IIII", M, K, nbPerRow, nsPerRow))
    f.write(w_packed.tobytes())
    f.write(b_one.tobytes())
    f.write(s_one.tobytes())
    f.write(act.tobytes())
print("saved t_p1b_all1.bin (expected result: 32)")
