
import safetensors.torch
state = safetensors.torch.load_file(r'E:\models\Qwen3.6-35B-A3B-MXFP4-MTP\model-00022-of-00023.safetensors')
for k in state:
    if 'mtp' in k and 'o_proj' in k:
        print(k, state[k].shape, state[k].dtype)
    if 'mtp' in k and 'q_proj' in k:
        print(k, state[k].shape, state[k].dtype)
    if 'mtp' in k and 'gate_proj' in k and 'scales' in k:
        print(k, state[k].shape, state[k].dtype)
    if 'mtp' in k and 'self_attn' in k and 'weight' in k:
        print(k, state[k].shape, state[k].dtype)
