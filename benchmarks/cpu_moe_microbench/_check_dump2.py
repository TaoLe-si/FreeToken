
import torch, numpy as np
d = torch.load("E:/FreeToken/igpu_layer_dump.pt", weights_only=False)
hidden, ids, weights, out = d["hidden"], d["ids"], d["weights"], d["out"]
print("out norm:", out.float().norm().item())
print("out nonzero:", (out != 0).float().mean().item())
print("ids:", ids[0].tolist())
# hidden 分布
print("hidden stats: mean", hidden.mean().item(), "std", hidden.std().item(), "norm", hidden.norm().item())
