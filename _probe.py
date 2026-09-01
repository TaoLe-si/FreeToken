import json, sys, re, os
os.environ["HF_HUB_OFFLINE"]="1"; os.environ["TRANSFORMERS_OFFLINE"]="1"
sys.path.insert(0, r"E:\FreeToken\python")
from freetoken.models.qwen3_5_moe.config import parse_config
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained(r"E:\models\Qwen3.8-27B-NVFP4")
c = parse_config(cfg)
print("attn_quant:", c.attn_quant)
print("dense_quant:", c.dense_quant)
print("lm_head_quant:", c.lm_head_quant)
print("layer_dense_quant_map:", c.layer_dense_quant_map)
print("L56 dense:", c.dense_quant_for_layer(56))
print("L0  dense:", c.dense_quant_for_layer(0))