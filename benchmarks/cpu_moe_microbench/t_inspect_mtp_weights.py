"""Inspect MTP head weight structure in the model."""
import os, json, torch, safetensors.torch
base = r"E:\models\Qwen3.6-35B-A3B-MXFP4-MTP"
with open(os.path.join(base, "model.safetensors.index.json")) as f:
    idx = json.load(f)
mtp_keys = sorted([k for k in idx["weight_map"] if k.startswith("mtp.")])
print("MTP keys (count):", len(mtp_keys))
for k in mtp_keys[:50]:
    print(" ", k, "->", idx["weight_map"][k])
print("...")
for k in mtp_keys[-5:]:
    print(" ", k, "->", idx["weight_map"][k])
