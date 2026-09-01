
import safetensors.torch, os, glob
mp = r'E:\models\Qwen3.6-35B-A3B-MXFP4-MTP'
files = sorted(glob.glob(mp + '/model-*.safetensors'))
print('files:', [os.path.basename(f) for f in files])
for p in files:
    state = safetensors.torch.load_file(p)
    mtp = {k: v for k, v in state.items() if k.startswith('mtp')}
    if mtp:
        print(f'--- {os.path.basename(p)} ({len(mtp)} mtp tensors) ---')
        for k, v in mtp.items():
            if 'attn' in k or 'self_attn' in k or 'switch_mlp.gate_proj' in k:
                print(f'  {k}: {tuple(v.shape)} {v.dtype}')
