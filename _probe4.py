import sys
sys.path.insert(0, r"E:\FreeToken\python")
import importlib, freetoken.models.config as m
importlib.reload(m)
from freetoken.models.config import detect_compressed_tensors_fp8_groups
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained(r"E:\models\Qwen3.8-27B-NVFP4")
a, mlp = detect_compressed_tensors_fp8_groups(cfg)
print("attn:", a, "mlp:", sorted(mlp))