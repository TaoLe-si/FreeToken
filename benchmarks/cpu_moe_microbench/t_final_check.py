import sys, os, struct, subprocess
import numpy as np
import torch
import safetensors.torch

SERVER = r'E:\FreeToken\benchmarks\cpu_moe_microbench\t_mxfp4_gemv_v3_server.exe'
state = safetensors.torch.load_file(r'E:\models\Qwen3.6-35B-A3B-MXFP4-MTP\model-00022-of-00023.safetensors')
fc_w = state['mtp.fc.weight']
fc_s = state['mtp.fc.scales']
M, K = 1, 4096
szP = fc_w[0:1].numpy().size * 4
szS = fc_s[0:1].float().numpy().size * 4
szA = K * 4
szB = M * 4
payload = f'STATELESS {M} {K} {szP} {szS} {szA} {szB}\n'.encode()
payload += fc_w[0:1].numpy().tobytes() + fc_s[0:1].float().numpy().tobytes()
payload += (0.05 * np.ones(K, dtype=np.float32)).tobytes()
payload += np.zeros(M, dtype=np.float32).tobytes()
payload += b'QUIT\n'
p = subprocess.Popen([SERVER], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=os.path.dirname(SERVER))
out, err = p.communicate(payload, timeout=30)
print(f'stderr: {err.decode(errors="replace")[:200]}')
if len(out) >= 4:
    sz = struct.unpack('<I', out[:4])[0]
    if sz > 0 and sz < 100000:
        v = np.frombuffer(out[4:4+sz], dtype=np.float32).copy()[0]
        print(f'STATELESS FC a=0.05: {v} (expected 0.271919)')
        print(f'Match: {abs(v - 0.271919) < 1e-4}')
    else:
        print(f'Bad sz: {sz}')
