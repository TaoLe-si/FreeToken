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

# Concatenate vertically: q on top, k middle, v bottom
# This makes M=1 with M_out_total=8192+512+512=9216. Server is M=1.
# Output will be [9216] which we split as q[8192], k[512], v[512]
qkv_w = np.concatenate([q_w, k_w, v_w], axis=0)  # [9216, 256]
K_in = 2048
act = np.random.randn(K_in).astype(np.float32)
N = 50
t0 = time.time()
for _ in range(N):
    qkv_out = client.forward(qkv_w[:1], act.view(np.int32))
t1 = time.time()
print(f'1 batched qkv call (M=1, K=2048, out=9216): {(t1-t0)*1000/N:.3f}ms/iter')
# Split
q_out = qkv_out[:8192]
k_out = qkv_out[8192:8192+512]
v_out = qkv_out[8192+512:]
print(f'q_out shape={q_out.shape}, k_out shape={k_out.shape}, v_out shape={v_out.shape}')

# Now with output dim = 8192 vs old 5.8ms dGPU
# iGPU does 9216 K=2048 in 0.3-0.5ms based on earlier benches (MoE 256 expert K=512 was 0.456ms)
# So this should be similar or slightly slower (~0.5-0.6ms for 9216 output)
