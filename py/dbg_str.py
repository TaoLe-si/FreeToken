import json, os
SRC = r"E:\models\Qwen3.6-35B-A3B-MXFP4-MTP"
idx = json.load(open(os.path.join(SRC, "model.safetensors.index.json")))
wm = idx["weight_map"]
k = "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight"
cand = k + ".scales"
sim = [x for x in wm if x.endswith("switch_mlp.gate_proj.scales") and x.startswith("language")]
a, b = cand, sim[0]
print("len:", len(a), len(b))
print("equal:", a == b)
for i,(ca,cb) in enumerate(zip(a,b)):
    if ca != cb:
        print("first diff at", i, repr(ca), repr(cb), hex(ord(ca)), hex(ord(cb)))
        break
else:
    print("prefix equal; extra tail:", repr(a[len(b):]) if len(a)>len(b) else repr(b[len(a):]))
