import sys, os
sys.path.insert(0, r'E:\FreeToken\python')
sys.path.insert(0, r'E:\FreeToken\benchmarks\cpu_moe_microbench')
os.chdir(r'E:\FreeToken\benchmarks\cpu_moe_microbench')
import safetensors.torch, numpy as np
kE2M1 = [0,1,2,3,4,6,8,12, 0,-1,-2,-3,-4,-6,-8,-12]
state = safetensors.torch.load_file(r'E:\models\Qwen3.6-35B-A3B-MXFP4-MTP\model-00022-of-00023.safetensors')
fc_w = state['mtp.fc.weight'].numpy()  # shape (1, K/2) uint8
# Each byte has 2 nibbles
row = fc_w[0]
total = 0
for byte in row:
    n_lo = byte & 0xF
    n_hi = (byte >> 4) & 0xF
    total += kE2M1[n_lo] + kE2M1[n_hi]
print(f'sum_nibbles = {total}')
for a in [0.01, 0.05, 0.10, 0.20]:
    print(f'  a={a}: a*sum_nibbles = {a*total}')
