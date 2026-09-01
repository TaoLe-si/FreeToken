import sys, os, struct
sys.path.insert(0, r'E:\FreeToken\python')
sys.path.insert(0, r'E:\FreeToken\benchmarks\cpu_moe_microbench')
os.chdir(r'E:\FreeToken\benchmarks\cpu_moe_microbench')
import safetensors.torch, numpy as np

state = safetensors.torch.load_file(r'E:\models\Qwen3.6-35B-A3B-MXFP4-MTP\model-00022-of-00023.safetensors')
fc_w = state['mtp.fc.weight'].numpy()
M, K = 1, 4096
ns = K // 32
nb = K // 8
packed = fc_w[:1].tobytes()
szP, szA, szS, szB = len(packed), K*4, M*ns*4, M*ns*4
act = np.full(K, 0.05, dtype='float32')

payload = b'STATELESS ' + str(M).encode() + b' ' + str(K).encode() + b' ' + str(szP).encode() + b' ' + str(szA).encode() + b' ' + str(szS).encode() + b' ' + str(szB).encode() + b'\n'
payload += packed + act.tobytes() + bytes(szS) + bytes(szB) + b'QUIT\n'

import subprocess
p = subprocess.Popen([r'E:\FreeToken\benchmarks\cpu_moe_microbench\t_mxfp4_gemv_v2_server.exe'],
                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                     cwd=r'E:\FreeToken\benchmarks\cpu_moe_microbench')
out, err = p.communicate(payload, timeout=30)
print('STDOUT len:', len(out))
if len(out) >= 8:
    sz = struct.unpack('<I', out[:4])[0]
    val = struct.unpack('<f', out[4:8])[0]
    print(f'STATELESS: sz={sz} outv={val}')
print('STDERR:', err.decode(errors='replace')[:1500])
print('rc:', p.returncode)
