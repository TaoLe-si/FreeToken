import sys
sys.path.insert(0, r"E:\FreeToken\python")
import importlib, freetoken.models.config as m
importlib.reload(m)
from freetoken.models.config import detect_compressed_tensors_fp8_groups
from freetoken.models.qwen3_5_moe.config import parse_config
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained(r"E:\models\Qwen3.8-27B-NVFP4")
a, mlp = detect_compressed_tensors_fp8_groups(cfg)
print("attn:", a, "mlp:", sorted(mlp))
c = parse_config(cfg)
print("attn_quant:", c.attn_quant, "dense_quant:", c.dense_quant)
print("layer_dense_quant_map:", c.layer_dense_quant_map)
print("L56:", c.dense_quant_for_layer(56), "L0:", c.dense_quant_for_layer(0))