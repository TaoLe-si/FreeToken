import sys, os, time
sys.path.insert(0, r'E:\\FreeToken\\python')
sys.path.insert(0, r'E:\\FreeToken\\benchmarks\\cpu_moe_microbench')
os.chdir(r'E:\\FreeToken\\benchmarks\\cpu_moe_microbench')
import safetensors.torch, numpy as np
state = safetensors.torch.load_file(r'E:\\models\\Qwen3.6-35B-A3B-MXFP4-MTP\\model-00022-of-00023.safetensors')
fc_w = state['mtp.fc.weight'].numpy()
act = np.full(4096, 0.05, dtype='float32').view(np.int32)
M, K = 1, 4096
ns = K // 32
packed = fc_w[:1].tobytes()
szA, szS, szB = K * 4, M * ns * 4, M * ns * 4

import subprocess
p = subprocess.Popen([r'E:\\FreeToken\\benchmarks\\cpu_moe_microbench\\t_mxfp4_gemv_multi_server.exe'],
                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                     cwd=r'E:\\FreeToken\\benchmarks\\cpu_moe_microbench')
p.stdin.write(b'LOAD fc 1 4096\n')
p.stdin.write(packed)
p.stdin.flush()
time.sleep(2)
p.stdin.write(b'CALL fc\n')
p.stdin.write(act.tobytes())
p.stdin.write(bytes(szS))
p.stdin.write(bytes(szB))
p.stdin.flush()
time.sleep(3)
out = p.stdout.read(4)
print('STDOUT first 4:', out.hex() if out else '(empty)')
err = p.stderr.read(8192).decode(errors='replace')
print('STDERR:', err[:2000])
p.terminate()
p.wait(timeout=3)
print('rc:', p.returncode)
