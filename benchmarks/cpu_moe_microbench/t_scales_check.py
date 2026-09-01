"""Check fc_s for nan/inf."""
import os, json, safetensors.torch
import torch
mdl = "E:\\models\\Qwen3.6-35B-A3B-MXFP4-MTP"
with open(os.path.join(mdl, "model.safetensors.index.json")) as f: idx = json.load(f)
state = safetensors.torch.load_file(os.path.join(mdl, idx["weight_map"]["mtp.fc.weight"]))
fc_s = state["mtp.fc.scales"].cpu().numpy().astype("float32")
fc_b = state["mtp.fc.biases"].cpu().numpy().astype("float32")
import numpy as np
print(f"fc_s.shape={fc_s.shape}")
print(f"fc_s[0, 0:5] = {fc_s[0, :5]}")
print(f"fc_s any nan: {np.isnan(fc_s).any()}")
print(f"fc_s any inf: {np.isinf(fc_s).any()}")
print(f"fc_s min: {fc_s.min()}, max: {fc_s.max()}")
print(f"fc_b[0, 0:5] = {fc_b[0, :5]}")
print(f"fc_b any nan: {np.isnan(fc_b).any()}")
