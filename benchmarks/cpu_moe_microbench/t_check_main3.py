
import safetensors.torch
state = safetensors.torch.load_file(r'E:\models\Qwen3.6-35B-A3B-MXFP4-MTP\model-00001-of-00023.safetensors')
for k in list(state.keys())[:30]:
    if 'layers.0.' in k and ('attn' in k or 'mlp' in k):
        v = state[k]
        if 'weight' in k and not 'norm' in k:
            print(f'{k} {tuple(v.shape)} {v.dtype}')
        if 'scales' in k:
            print(f'{k} {tuple(v.shape)} {v.dtype}')
