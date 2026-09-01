
import torch, numpy as np, ctypes, os
os.add_dll_directory(r"C:\Program Files\AMD\ROCm\6.4\bin")
d = torch.load("E:/FreeToken/igpu_layer_dump.pt", weights_only=False)
hidden, ids, weights, out = d["hidden"], d["ids"], d["weights"], d["out"]
print("hidden", hidden.shape, hidden.dtype, "norm", hidden.float().norm().item())
print("ids", ids[0].tolist())
print("weights", weights[0].tolist())
print("out norm", out.float().norm().item(), "out[0,:4]", out[0,:4].tolist())
# 加载真实 layer0 banks
from safetensors import safe_open
import json
# 从 ftw manifest 找 layer0 bank 文件
mf = json.load(open(r"E:/models/Qwen3.6-35B-A3B-MXFP4-MTP-NVFP4/freetoken_weight.json"))
def find_layer0(mf):
    # 打印结构线索
    if isinstance(mf, dict):
        return list(mf.keys())[:8]
    return type(mf)
print("manifest keys:", find_layer0(mf))
