import json
from safetensors import safe_open
out = r"E:\models\Qwen3.6-35B-A3B-NVFP4-MTP-debf"
idx = json.load(open(out + r"\model.safetensors.index.json"))
shard = idx["weight_map"]["model.layers.0.mlp.experts.0.gate_proj.weight"]
with safe_open(out + "\\" + shard, framework="pt") as f:
    keys = list(f.keys())
    t = f.get_tensor("model.layers.0.mlp.experts.0.gate_proj.weight")
    print("expert gate:", tuple(t.shape), t.dtype)
    e = f.get_tensor("model.embed_tokens.weight")
    print("embed:", tuple(e.shape), e.dtype)
    lh = None
    for k in keys[:500000]:
        if k == "lm_head.weight": lh = k; break
    if lh:
        t2 = f.get_tensor(lh)
        print("lm_head:", tuple(t2.shape), t2.dtype)
    # 统计本分片总字节
    tot = 0
    for k in keys:
        tt = f.get_slice(k)
        shp = tt.get_shape()
        n = 1
        for d in shp: n *= d
        tot += n * 2
    print("shard tensors:", len(keys), "approx GB:", round(tot/1e9,2))
