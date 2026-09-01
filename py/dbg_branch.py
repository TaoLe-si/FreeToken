import json, os
SRC = r"E:\models\Qwen3.6-35B-A3B-MXFP4-MTP"
idx = json.load(open(os.path.join(SRC, "model.safetensors.index.json")))
wm = idx["weight_map"]
k = "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight"
print("key in wm:", k in wm)
print("scales in wm:", (k+".scales") in wm)
print("biases in wm:", (k+".biases") in wm)
# 找出实际以该词中缀的键
sim = [x for x in wm if "layers.0.mlp.switch_mlp.gate_proj" in x]
print("相关键:", sim)
