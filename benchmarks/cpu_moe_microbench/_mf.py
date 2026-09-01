
import json
mf = json.load(open(r"E:/models/Qwen3.6-35B-A3B-MXFP4-MTP-NVFP4/freetoken_weight.json"))
ts = mf["tensors"]
print(type(ts), len(ts) if hasattr(ts, "__len__") else "")
t0 = ts[0]
print("entry sample:", json.dumps(t0)[:300])
# 找 layer 0 的 expert bank
l0 = [t for t in ts if str(t.get("name", "")).find("layers.0") >= 0 or str(t.get("name", "")).find(".0.") >= 0]
print("layer0 entries:", len(l0))
for t in l0[:12]: print(json.dumps(t)[:200])
