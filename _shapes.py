import sys, json, torch
sys.path.insert(0, "E:/FreeToken/python")
from safetensors import safe_open
src = "E:/models/Qwen3.6-35B-A3B-MXFP4-MTP"
wmap = json.load(open(f"{src}/model.safetensors.index.json", encoding="utf-8"))["weight_map"]
for k in ["language_model.model.layers.3.self_attn.q_proj",
          "language_model.model.layers.3.self_attn.k_proj",
          "language_model.model.layers.0.linear_attn.in_proj_qkv",
          "language_model.lm_head"]:
    row = []
    for suf in (".weight", ".scales", ".biases"):
        with safe_open(f"{src}/{wmap[k+suf]}", framework="pt") as fh:
            t = fh.get_tensor(k + suf)
            row.append(f"{suf[1:]} {tuple(t.shape)} {t.dtype}")
    print(k.split(".")[-1] + ": " + "  ".join(row))
