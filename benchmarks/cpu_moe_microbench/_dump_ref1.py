
import torch, numpy as np, ctypes, os, json, glob
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

def read_bank(name):
    e = entries[name]
    for s in mf["shards"]:
        # bank 可能在任一 shard; 计算 shard 内偏移
        pass
    # 简化: shards 顺序连续, file 由 global_off 决定
    off = e["global_off"]
    # 找所属 shard
    cur = 0
    for s in mf["shards"]:
        if off < s["global_off"] + s["nbytes"]:
            fpath = r"E:/models/Qwen3.6-35B-A3B-MXFP4-MTP-NVFP4/" + s["file"]
            local = off - s["global_off"]
            with open(fpath, "rb") as f:
                f.seek(local)
                return np.fromfile(f, dtype=e["dtype_np"] if "dtype_np" in e else None, count=e["nbytes"]) if False else None
    return None

# 更直接: 每个 shard 一个文件, global_off 是该 shard 内? 看 shard1 的 global_off
for s in mf["shards"]:
    print("shard:", s["file"], "goff:", s["global_off"], "nbytes:", s["nbytes"])
