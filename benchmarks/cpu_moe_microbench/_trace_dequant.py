
import sys, os
import numpy as np
import torch
import safetensors.torch
import json

with open(r'E:\models\Qwen3.6-35B-A3B-MXFP4-MTP\model.safetensors.index.json') as f:
    idx = json.load(f)
files_needed = set(f for k, f in idx['weight_map'].items() if k.startswith('mtp.layers.0.mlp.switch_mlp.down_proj'))
all_mtp_state = {}
for f in files_needed:
    all_mtp_state.update(safetensors.torch.load_file(os.path.join(r'E:\models\Qwen3.6-35B-A3B-MXFP4-MTP', f)))

sw_gate = all_mtp_state['mtp.layers.0.mlp.switch_mlp.gate_proj.weight']
sw_gate_s = all_mtp_state['mtp.layers.0.mlp.switch_mlp.gate_proj.scales']

sys.path.insert(0, r'E:\FreeToken\benchmarks\cpu_moe_microbench')
from t_mxfp4_dequant import dequant_mxfp4_expert_block

gate_dq = dequant_mxfp4_expert_block(sw_gate[7:8].float(), sw_gate_s[7:8].float(), torch.zeros_like(sw_gate_s[7:8]).float()).numpy()[0]
print(f'gate_dq[0, :10] = {gate_dq[0, :10]}')
print(f'gate_dq[0, 32:42] = {gate_dq[0, 32:42]}')
print(f'gate_dq[0, 64:74] = {gate_dq[0, 64:74]}')
print(f'\nSum of gate_dq[0, :32] = {gate_dq[0, :32].sum()}')
print(f'Sum of gate_dq[0, 32:64] = {gate_dq[0, 32:64].sum()}')

np.random.seed(42)
x = np.random.randn(2048).astype(np.float32) * 0.02
print(f'\nx[:5] = {x[:5]}')
print(f'x[32:37] = {x[32:37]}')
print(f'x[64:69] = {x[64:69]}')

# So gate_dq[0] = sum_b sum_k W[0, k] * scale[0, 0, b] for k in block b
# = sum_b scale_b * wsum_b
# 
# Then CPU gate output[0] = gate_dq[0] @ x = sum_k W[0, k] * x[k]
# = sum_b sum_k W[0, k] * x[k] for k in block b
# = sum_b per_block_sum (W * x) * scale_b  ... wait this is wrong
# 
# gate_dq[0, k] = W[0, k] * scale[0, 0, k//32] (scale repeats every 32 K-elements)
# 
# So sum_k gate_dq[0, k] * x[k] = sum_k W[0, k] * scale[0, 0, k//32] * x[k]
# = sum_b scale_b * sum_{k in block b} W[0, k] * x[k]
# = sum_b scale_b * wsum_dot_b
# 
# That's exactly what the shader computes. So output should match!

print(f'\nCPU gate output = {gate_dq @ x}')
print(f'Manual (sum_b scale_b * sum_k W*x) = {sum(gate_dq[0, :32] * x[:32])}, {sum(gate_dq[0, 32:64] * x[32:64])}, etc.')

# The dot product = sum over k of W[0,k] * scale[0,0,k//32] * x[k]
# = sum over blocks: scale_b * sum_k_in_block W*x
# 
# Let me check if the iGPU is computing this correctly.
