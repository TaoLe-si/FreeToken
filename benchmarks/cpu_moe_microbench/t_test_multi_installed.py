"""Test multi-GEMV shader via DXIL dispatch."""
import sys, os, struct, ctypes
import numpy as np

# Load the multi-GEMV dxil directly and call it via ctypes (no server needed).
# This requires D3D12 setup, which is complex. Let me just install the shader
# as the new t_mxfp4_gemv_sk.dxil and test via a custom Python wrapper using ctypes.

# Actually simpler: just install it as t_mxfp4_gemv_sk.dxil and test via v3 server
# with a new BATCH_ALL_MULTI command. But v3 server doesn't have that command.

# Let me just install the multi-GEMV shader as the new t_mxfp4_gemv_sk.dxil
# and see if it works as a single-GEMV (B=1).
import shutil
shutil.copy('E:/FreeToken/benchmarks/cpu_moe_microbench/_multi_gemv.dxil',
            'E:/FreeToken/benchmarks/cpu_moe_microbench/t_mxfp4_gemv_sk.dxil')
shutil.copy('E:/FreeToken/benchmarks/cpu_moe_microbench/t_mxfp4_gemv_sk.dxil',
            'E:/FreeToken/benchmarks/cpu_moe_microbench/t_mxfp4_gemv_sk.dxil.bak2')

import sys
sys.path.insert(0, 'E:/FreeToken/python')
import torch
import safetensors.torch
state = safetensors.torch.load_file('E:/models/Qwen3.6-35B-A3B-MXFP4-MTP/model-00022-of-00023.safetensors')
fc_w = state['mtp.fc.weight']

# Use v3 server to test (it uses this shader)
import subprocess
SERVER = 'E:/FreeToken/benchmarks/cpu_moe_microbench/t_mxfp4_gemv_v3_server.exe'
M, K = 1, 4096
szP = fc_w[0:1].numpy().size * 4
# For B=1 (single), just have scales = 128 floats
import numpy as np
fc_s = state['mtp.fc.scales'][0:1].float().numpy()
szS = fc_s.size * 4
szA = K * 4
szB = M * 4
payload = f'STATELESS {M} {K} {szP} {szS} {szA} {szB}\n'.encode()
payload += fc_w[0:1].numpy().tobytes() + fc_s.tobytes()
payload += np.full(K, 0.05, dtype=np.float32).tobytes() + np.zeros(M, dtype=np.float32).tobytes()
payload += b'QUIT\n'
p = subprocess.Popen([SERVER], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
out, err = p.communicate(payload, timeout=30)
print(f'stderr: {err.decode(errors="replace")[:300]}')
if len(out) >= 4:
    sz = struct.unpack('<I', out[:4])[0]
    v = np.frombuffer(out[4:4+sz], dtype=np.float32).copy()[0]
    print(f'FC output: {v} (expected 0.271919)')
