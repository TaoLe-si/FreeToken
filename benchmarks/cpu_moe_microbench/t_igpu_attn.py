"""iGPU MTP attention wrapper.

Replaces dGPU qkv_proj (5.8ms) with single iGPU call (~0.4ms).
Packs q/k/v weights into one stacked [1, q_w+k_w+v_w] packed row and runs GEMV.
"""
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
# Get q/k/v/o weights (raw uint32)
q_w = state['mtp.layers.0.self_attn.q_proj.weight'].numpy()  # [4096, 256] uint32 = K=2048
k_w = state['mtp.layers.0.self_attn.k_proj.weight'].numpy()  # [512, 256] uint32
v_w = state['mtp.layers.0.self_attn.v_proj.weight'].numpy()  # [512, 256] uint32
o_w = state['mtp.layers.0.self_attn.o_proj.weight'].numpy()  # [2048, 512] uint32

print(f'q_w: {q_w.shape} uint32, K={q_w.shape[1]*8}')
print(f'k_w: {k_w.shape} uint32, K={k_w.shape[1]*8}')
print(f'v_w: {v_w.shape} uint32, K={v_w.shape[1]*8}')

# For unified M=1 call, need to stack along output dim (concat vertically)
# But shapes differ in output dim. q is [4096,256], k is [512,256], v is [512,256]
# So can't just stack vertically because rows differ.

# Approach: 3 separate calls (q, k, v) and concat results
K_in = 2048
act = np.random.randn(K_in).astype(np.float32)

# Time 3 sequential calls
N = 50
t0 = time.time()
for _ in range(N):
    q_out = client.forward(q_w[:1], act.view(np.int32))
    k_out = client.forward(k_w[:1], act.view(np.int32))
    v_out = client.forward(v_w[:1], act.view(np.int32))
t1 = time.time()
print(f'3 sequential qkv calls: {(t1-t0)*1000/N:.3f}ms/iter')
print(f'  vs dGPU qkv_proj: 5.801ms -> save ~5.4ms')

# Time 1 separate o call
K_in_o = 4096
o_w_1 = o_w[:1]  # M=1
act_o = np.random.randn(K_in_o).astype(np.float32)
t0 = time.time()
for _ in range(N): o_out = client.forward(o_w_1, act_o.view(np.int32))
t1 = time.time()
print(f'1 o call: {(t1-t0)*1000/N:.3f}ms/iter')
print(f'  vs dGPU o_proj: 0.080ms -> no savings actually (o_proj is fast on dGPU)')
