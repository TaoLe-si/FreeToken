import sys, re, os, json
sys.path.insert(0, r"E:\FreeToken\python")
from freetoken.models.config import detect_compressed_tensors_fp8_groups
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained(r"E:\models\Qwen3.8-27B-NVFP4")
q = cfg.quantization_config
print("groups keys:", list(q["config_groups"].keys()))
for name, g in q["config_groups"].items():
    print("  ", name, g.get("format"), "bits=", (g.get("weights") or {}).get("num_bits"))
    for t in (g.get("targets") or []):
        print("    T:", repr(t))
print()
a, m = detect_compressed_tensors_fp8_groups(cfg)
print("attn_fp8=", a, "mlp_layers=", sorted(m))