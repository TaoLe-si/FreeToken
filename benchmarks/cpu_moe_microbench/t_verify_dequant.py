"""Verify MXFP4 dequant matches expected output."""
import sys
sys.path.insert(0, r'E:\\\\FreeToken\\\\benchmarks\\\\cpu_moe_microbench')
import torch, safetensors.torch, os, json, struct
from t_mxfp4_dequant import dequant_mxfp4_packed_row, dequant_mxfp4_expert_block

base = r"E:\\\\models\\\\Qwen3.6-35B-A3B-MXFP4-MTP"
state = safetensors.torch.load_file(os.path.join(base, "model-00022-of-00023.safetensors"))
fc_w = state["mtp.fc.weight"][0]  # row 0
fc_b = state["mtp.fc.biases"][0]
fc_s = state["mtp.fc.scales"][0]
print("fc_w shape:", fc_w.shape, "fc_b shape:", fc_b.shape, "fc_s shape:", fc_s.shape)

# Dequant row 0 with my function
W_dequant = dequant_mxfp4_packed_row(fc_w, 4096)
print("W_dequant[:10]:", W_dequant[:10].tolist())
print("expected: [-6, -3, -2, -3, -1, -2, 6, -4, 12, 8] (from earlier [P0] analysis)")

# Verify the dequant matches what we extracted earlier
# From [P0] analysis, MTP fc row 0 first 32 weights = [-6, -3, -2, -3, -1, -2, 6, -4, 12, 8, -4, 4, 0, 6, -2, -4, -1, -1, 4, 0, 12, -4, -8, -1, -6, 0, 0, -4, -12, -8, -1, -2]
expected = [-6, -3, -2, -3, -1, -2, 6, -4, 12, 8, -4, 4, 0, 6, -2, -4, -1, -1, 4, 0, 12, -4, -8, -1, -6, 0, 0, -4, -12, -8, -1, -2]
got = W_dequant[:32].tolist()
print("match:", got == expected)
if got != expected:
    for i in range(32):
        if got[i] != expected[i]:
            print(f"  mismatch at {i}: got {got[i]}, expected {expected[i]}")
