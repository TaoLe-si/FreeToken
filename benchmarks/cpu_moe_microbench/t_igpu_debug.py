
import sys, time
sys.path.insert(0, r'E:\FreeToken\python')
sys.path.insert(0, r'E:\FreeToken\benchmarks\cpu_moe_microbench')
import torch
torch.set_grad_enabled(False)
import numpy as np
from freetoken.kernel.igpu_fc import IgpuFcClient

client = IgpuFcClient()
import safetensors.torch
state = safetensors.torch.load_file(r'E:\models\Qwen3.6-35B-A3B-MXFP4-MTP\model-00023-of-00023.safetensors')
q_w = state['mtp.layers.0.self_attn.q_proj.weight'].numpy()  # [8192, 256]
k_w = state['mtp.layers.0.self_attn.k_proj.weight'].numpy()  # [512, 256]
v_w = state['mtp.layers.0.self_attn.v_proj.weight'].numpy()  # [512, 256]
qkv_w = np.concatenate([q_w, k_w, v_w], axis=0)  # [9216, 256]
K_in = 2048
act = np.random.randn(K_in).astype(np.float32)
try:
    qkv_out = client.forward(qkv_w, act.view(np.int32))
    print(f'success: outv shape {qkv_out.shape}')
except Exception as e:
    print(f'error: {e}')
print('Server log:')
for l in client.stderr_lines[-10:]: print(l, end='')
