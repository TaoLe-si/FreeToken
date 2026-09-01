
import safetensors.torch, os
state = safetensors.torch.load_file(r'E:\models\Qwen3.6-35B-A3B-MXFP4-MTP\model-00023-of-00023.safetensors')
# Find main model layer 0 o_proj (not mtp)
for k in state:
    if k.startswith('model.layers.0.self_attn'):
        print(k, state[k].shape, state[k].dtype)
    if k.startswith('model.layers.0.mlp'):
        print(k, state[k].shape, state[k].dtype)
        if 'expert' in k and '0' in k and 'scales' in k:
            break
