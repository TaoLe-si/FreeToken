"""Generate M=128 act to debug."""
import numpy as np, struct
out = "E:\\FreeToken\\benchmarks\\cpu_moe_microbench"
M, K = 128, 4096
nbPerRow = K // 8
nsPerRow = K // 32
rng = np.random.default_rng(42)
act = rng.normal(0, 1, size=K).astype(np.float32)
with open(out + "/t_mtp_fc_128.bin", "wb") as f:
    f.write(struct.pack("IIII", M, K, nbPerRow, nsPerRow))
    f.write(np.zeros(M * nbPerRow, dtype=np.uint32).tobytes())  # dummy weights (zeros)
    f.write(np.zeros(M * nsPerRow, dtype=np.float32).tobytes())  # dummy biases
    f.write(np.ones(M * nsPerRow, dtype=np.float32).tobytes())   # scale = 1
    f.write(act.tobytes())
print("done")
