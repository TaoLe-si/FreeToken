
import json
mf = json.load(open(r"E:/models/Qwen3.6-35B-A3B-MXFP4-MTP-NVFP4/freetoken_weight.json"))
ts = mf["tensors"]
l0 = [t for t in ts if "layers.0" in t["name"] and ("expert" in t["name"] or "pack" in t["name"] or "global" in t["name"])]
for t in l0[:15]: print(json.dumps(t)[:220])
print("---")
# 所有 bank 类名字
names = sorted({t["name"].replace("layers.0.", "layers.N.") for t in ts if "pack" in t["name"] or "experts" in t["name"]})
for n in names[:12]: print(n)
