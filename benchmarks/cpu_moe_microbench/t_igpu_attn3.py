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

# qkv as M=9216, K=2048 (server needs M>1 which we proved works for 256 experts)
qkv_w = np.concatenate([q_w, k_w, v_w], axis=0)  # [9216, 256]
K_in = 2048
act = np.random.randn(K_in).astype(np.float32)
act_broadcast = np.broadcast_to(act, (qkv_w.shape[0], K_in)).copy()  # act replicated per row
# Or just send act as [K_in] once and use packed [M, K/8] -- act is shared per row
N = 50
t0 = time.time()
for _ in range(N):
    qkv_out = client.forward(qkv_w, act.view(np.int32))
t1 = time.time()
print(f'1 batched qkv call (M=9216, K=2048): {(t1-t0)*1000/N:.3f}ms/iter')
print(f'  outv shape: {qkv_out.shape}')
q_out = qkv_out[:8192]
k_out = qkv_out[8192:8192+512]
v_out = qkv_out[8192+512:]
print(f'q_out[:3]={q_out[:3]}, k_out[:3]={k_out[:3]}, v_out[:3]={v_out[:3]}')

# Compare to 3 separate
N = 50
t0 = time.time()
for _ in range(N):
    q_out2 = client.forward(q_w[:1], act.view(np.int32))
    k_out2 = client.forward(k_w[:1], act.view(np.int32))
    v_out2 = client.forward(v_w[:1], act.view(np.int32))
t1 = time.time()
print(f'3 sequential qkv calls: {(t1-t0)*1000/N:.3f}ms/iter')

# 256 expert MoE K=512 was 0.456ms. 9216*2048 = 18.9M flops vs 256*512 = 131K = 144x more
# But 1 dispatch vs 1 dispatch, just more rows. Should be ~0.5-1ms.
