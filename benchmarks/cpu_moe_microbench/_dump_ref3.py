
import torch, numpy as np, ctypes, os, json
os.add_dll_directory(r"C:\Program Files\AMD\ROCm\6.4\bin")
hip = ctypes.CDLL(r"C:\Program Files\AMD\ROCm\6.4\bin\amdhip64_6.dll")
d = torch.load("E:/FreeToken/igpu_layer_dump.pt", weights_only=False)
hidden = d["hidden"][0].numpy().astype(np.float32)
ids = d["ids"][0].numpy().astype(np.int64)
weights = d["weights"][0].numpy().astype(np.float32)
dll_out = d["out"][0].numpy().astype(np.float32)

mf = json.load(open(r"E:/models/Qwen3.6-35B-A3B-MXFP4-MTP-NVFP4/freetoken_weight.json"))
entries = {t["name"]: t for t in mf["tensors"]}
H, I, NE = 2048, 512, 256
MODEL = r"E:/models/Qwen3.6-35B-A3B-MXFP4-MTP-NVFP4"

def read_raw(name):
    e = entries[name]
    off, nb = e["global_off"], e["nbytes"]
    for s in mf["shards"]:
        if s["global_off"] <= off < s["global_off"] + s["nbytes"]:
            with open(MODEL + "/" + s["file"], "rb") as f:
                f.seek(off - s["global_off"])
                return np.frombuffer(f.read(nb), dtype=np.uint8)
    raise KeyError(name)

E2M1 = np.array([0.0,0.5,1.0,1.5,2.0,3.0,4.0,6.0,-0.0,-0.5,-1.0,-1.5,-2.0,-3.0,-4.0,-6.0], dtype=np.float32)
def e4m3(u8):
    s = np.where(u8 & 0x80, -1.0, 1.0).astype(np.float32)
    e = ((u8 >> 3) & 0xF).astype(np.int32); m = (u8 & 7).astype(np.int32)
    val = np.where(e == 0, m/8.0*np.float32(2.0**-6), (1+m/8.0)*np.float32(2.0)**(e-7).astype(np.float32))
    return s*val
def f16(u16):
    return torch.from_numpy(u16.astype(np.uint16)).view(torch.float16).float().numpy()
def dequant(packed, scale, glb, N, K):
    lo = (packed & 0xF).astype(np.int32); hi = (packed >> 4).astype(np.int32)
    codes = np.empty((N, K), dtype=np.int32); codes[:, 0::2] = lo; codes[:, 1::2] = hi
    return E2M1[codes] * np.repeat(e4m3(scale), 16, axis=1) * glb[:, None]

out_ref = np.zeros(H, dtype=np.float32)
for i, eid in enumerate(ids):
    gp = read_raw(f"gate_up_packed#L00000").reshape(NE, 2*I, H//2)[eid]
    gs = read_raw(f"gate_up_scale#L00000").reshape(NE, 2*I, H//16)[eid]
    gg = read_raw(f"gate_up_global#L00000").view(np.uint16).reshape(NE, 2*I)[eid]
    gu = dequant(gp, gs, f16(gg), 2*I, H) @ hidden
    gate, up = gu[:I], gu[I:]
    act = (gate / (1 + np.exp(-np.clip(gate, -30, 30)))) * up
    dp = read_raw(f"down_packed#L00000").reshape(NE, H, I//2)[eid]
    ds = read_raw(f"down_scale#L00000").reshape(NE, H, I//16)[eid]
    dg = read_raw(f"down_global#L00000").view(np.uint16).reshape(NE, H)[eid]
    out_ref += weights[i] * (dequant(dp, ds, f16(dg), H, I) @ act)

print("ref out norm:", np.linalg.norm(out_ref), "dll out norm:", np.linalg.norm(dll_out))
print("ref[0,:4]:", out_ref[:4])
# 归一化对比（数值精度被 scale 吃掉时 rel err 失真）— 用余弦
cos = float(np.dot(out_ref, dll_out) / (np.linalg.norm(out_ref)*np.linalg.norm(dll_out) + 1e-9))
print("cos:", cos)
