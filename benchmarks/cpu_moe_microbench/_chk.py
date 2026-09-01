
import torch
d = torch.load("E:/FreeToken/igpu_layer_dump.pt", weights_only=False)
print("out norm:", d["out"].float().norm().item())
print("out[0,:4]:", d["out"][0,:4].tolist())
print("hidden norm:", d["hidden"].norm().item())
