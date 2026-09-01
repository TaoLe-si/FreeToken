
import safetensors.torch
import os
# Check both files
for fname in ['model-00022-of-00023.safetensors', 'model-00021-of-00023.safetensors', 'model-00020-of-00023.safetensors']:
    p = r'E:\models\Qwen3.6-35B-A3B-MXFP4-MTP\' + fname
    if not os.path.exists(p): continue
    state = safetensors.torch.load_file(p)
    for k in state:
        if 'mtp' in k and 'attn' in k and 'weight' in k and 'o_proj' in k:
            print(fname, k, state[k].shape, state[k].dtype)
        if 'mtp' in k and 'q_proj' in k and 'weight' in k:
            print(fname, k, state[k].shape, state[k].dtype)
