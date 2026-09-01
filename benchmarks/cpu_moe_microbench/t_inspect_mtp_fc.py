import torch, safetensors.torch, os, json
base = "E:\\models\\Qwen3.6-35B-A3B-MXFP4-MTP"
with open(os.path.join(base, "model.safetensors.index.json")) as f: idx = json.load(f)
fc_file = idx["weight_map"]["mtp.fc.weight"]
state = safetensors.torch.load_file(os.path.join(base, fc_file))
W = state["mtp.fc.weight"]
B = state["mtp.fc.biases"]
S = state["mtp.fc.scales"]
print("W:", W.shape, W.dtype)
print("B:", B.shape, B.dtype, "min/max/mean:", B.min().item(), B.max().item(), B.float().mean().item())
print("S:", S.shape, S.dtype, "min/max/mean:", S.min().item(), S.max().item(), S.float().mean().item())
print("W[0,:4] hex:", [hex(int(v)) for v in W[0,:4].tolist()])
print("B[0,:8]:", B[0,:8].tolist())
print("S[0,:8]:", S[0,:8].tolist())
