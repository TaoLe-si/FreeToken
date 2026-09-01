import sys, os, struct, subprocess
import numpy as np
import safetensors.torch

# Test the OLDER d3d12 server (if it has a STATELESS-like command)
SERVER = r'E:\FreeToken\benchmarks\cpu_moe_microbench\t_mxfp4_gemv_d3d12.exe'
import os
if os.path.exists(SERVER):
    state = safetensors.torch.load_file(r'E:\models\Qwen3.6-35B-A3B-MXFP4-MTP\model-00022-of-00023.safetensors')
    fc_w = state['mtp.fc.weight']
    fc_s = state['mtp.fc.scales']
    M, K = 1, 4096
    szP = fc_w[0:1].numpy().size * 4
    szS = fc_s[0:1].float().numpy().size * 4
    szA = K * 4
    payload = struct.pack('<IIIIII', M, K, szP, szS, szA, M * 4) + fc_w[0:1].numpy().tobytes() + fc_s[0:1].float().numpy().tobytes() + (0.05 * np.ones(K, dtype=np.float32)).tobytes() + b'\x00' * (M * 4)  # bias = 0
    p = subprocess.Popen([SERVER], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=os.path.dirname(SERVER))
    out, err = p.communicate(payload, timeout=15)
    print(f'stderr: {err.decode(errors="replace")[:200]}')
    print(f'stout len: {len(out)}')
    print(f'first 20 bytes hex: {out[:20].hex()}')
