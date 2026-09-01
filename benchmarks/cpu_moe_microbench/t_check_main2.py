
import safetensors.torch, os
mp = r'E:\models\Qwen3.6-35B-A3B-MXFP4-MTP'
import glob
files = sorted(glob.glob(mp + '/model-*.safetensors'))
for p in files:
    state = safetensors.torch.load_file(p)
    for k in list(state.keys())[:5]:
        if k.startswith('model.layers.0.'):
            v = state[k]
            if 'attn' in k and 'weight' in k and not 'norm' in k:
                print(f'{os.path.basename(p)}: {k} {tuple(v.shape)} {v.dtype}')
            if 'o_proj' in k and 'scales' in k:
                print(f'{os.path.basename(p)}: {k} {tuple(v.shape)} {v.dtype}')
