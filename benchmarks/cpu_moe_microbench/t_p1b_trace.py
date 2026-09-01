import numpy as np, struct, json, torch, safetensors.torch, os
base = "E:\\models\\Qwen3.6-35B-A3B-MXFP4-MTP"
with open(os.path.join(base, "model.safetensors.index.json")) as f: idx = json.load(f)
state = safetensors.torch.load_file(os.path.join(base, idx["weight_map"]["mtp.fc.weight"]))
fc_w = state["mtp.fc.weight"].cpu().numpy()
w_packed = fc_w[0, :4]
print("packed uints:", [hex(int(v)) for v in w_packed])

kE2M1 = [0,1,2,3,4,6,8,12, 0,-1,-2,-3,-4,-6,-8,-12]
def nibbles_from_uint(u):
    return [(u >> s) & 0xF for s in (0, 4, 8, 12, 16, 20, 24, 28)]

K = 32
act = np.ones(K, dtype=np.float32)
wsum_total = 0.0
for j in range(4):
    w = int(w_packed[j])
    n = nibbles_from_uint(w)
    print(f"uint[{j}] = 0x{w:08x}, nibbles = {n}")
    abase = j * 8
    for k in range(8):
        w_n = kE2M1[n[k]]
        a_val = act[abase + k]
        contrib = w_n * a_val
        wsum_total += contrib
        print(f"  nibble[{k}]={n[k]} -> w={w_n}, act[{abase+k}]={a_val}, contrib={contrib}")

print(f"\nTotal wsum = {wsum_total}")
print(f"Expected if all W=-1: {-K}")
