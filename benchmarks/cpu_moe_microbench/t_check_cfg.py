
import safetensors.torch
state = safetensors.torch.load_file(r'E:\models\Qwen3.6-35B-A3B-MXFP4-MTP\model-00001-of-00023.safetensors')
# Look at config
import json
with open(r'E:\models\Qwen3.6-35B-A3B-MXFP4-MTP\config.json') as f:
    cfg = json.load(f)
print('hidden_size:', cfg.get('hidden_size'))
print('linear_attn:', cfg.get('linear_attn', '?'))
print('linear_num_key_heads/val_heads/qk_dim/v_dim:', cfg.get('linear_num_key_heads'), cfg.get('linear_num_value_heads'), cfg.get('linear_key_head_dim'), cfg.get('linear_value_head_dim'))
# Find all linear_attn params
for k, v in state.items():
    if 'layers.0.' in k and 'attn' in k:
        print(f'  {k}: {tuple(v.shape)}')
