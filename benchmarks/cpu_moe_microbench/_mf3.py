
import json
mf = json.load(open(r"E:/models/Qwen3.6-35B-A3B-MXFP4-MTP-NVFP4/freetoken_weight.json"))
ts = mf["tensors"]
for t in ts:
    if t["name"] in ("down_packed#L00000", "gate_up_packed#L00000", "gate_up_scale#L00000", "down_global#L00000"):
        print(json.dumps(t)[:250])
