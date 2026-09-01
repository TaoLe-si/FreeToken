import re
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained(r"E:\models\Qwen3.8-27B-NVFP4")
q = cfg.quantization_config
for name, g in q["config_groups"].items():
    w = (g or {}).get("weights") or {}
    if int(w.get("num_bits", 0) or 0) != 8: continue
    for t in (g.get("targets") or []):
        if not isinstance(t, str) or not t.startswith("re:"): continue
        pat = t[3:]
        if ".mlp." not in pat: continue
        print("pat:", repr(pat))
        li = pat.find("layers")
        print("  li=", li)
        lp = pat.find("(", li + 6) if li >= 0 else -1
        rp = pat.find(")", lp + 1) if lp >= 0 else -1
        print("  lp=",lp,"rp=",rp,"inner=",pat[lp+1:rp] if 0<lp<rp else None)