import struct, numpy as np
out = "E:\\FreeToken\\benchmarks\\cpu_moe_microbench"
with open(out + "/t_mtp_fc_weights.bin", "rb") as f:
    M, K, nbPerRow, nsPerRow = struct.unpack("IIII", f.read(16))
    f.read(M * nbPerRow * 4)
    biases = np.frombuffer(f.read(M * nsPerRow * 4), dtype=np.float32)
    scales = np.frombuffer(f.read(M * nsPerRow * 4), dtype=np.float32)

print("biases[:5] =", biases[:5])
print("scales[:5] =", scales[:5])
print("Are they identical?", np.array_equal(biases, scales))
print("Same first 100?", (biases[:100] == scales[:100]).all())
