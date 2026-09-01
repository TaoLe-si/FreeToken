import sys, os, time
sys.path.insert(0, r'E:\\FreeToken\\python')
sys.path.insert(0, r'E:\\FreeToken\\benchmarks\\cpu_moe_microbench')
os.chdir(r'E:\\FreeToken\\benchmarks\\cpu_moe_microbench')
import numpy as np
import safetensors.torch

state = safetensors.torch.load_file(r'E:\\models\\Qwen3.6-35B-A3B-MXFP4-MTP\\model-00022-of-00023.safetensors')
fc_w = state['mtp.fc.weight'].numpy()
act = np.full(4096, 0.05, dtype='float32').view(np.int32)
M, K = 1, 4096
ns = K // 32
packed = fc_w[:1].tobytes()
szA = K * 4
szS = M * ns * 4
szB = M * ns * 4
import subprocess
# Split into separate writes with flushes
p = subprocess.Popen([r'E:\\FreeToken\\benchmarks\\cpu_moe_microbench\\t_mxfp4_gemv_multi_server.exe'],
                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                     cwd=r'E:\\FreeToken\\benchmarks\\cpu_moe_microbench')
p.stdin.write(b'LOAD fc 1 4096\n')
p.stdin.write(packed)
p.stdin.flush()
time.sleep(0.1)
p.stdin.write(b'CALL fc\n')
p.stdin.write(act.tobytes())
p.stdin.write(bytes(szS))
p.stdin.write(bytes(szB))
p.stdin.flush()
time.sleep(0.3)
p.stdin.write(b'QUIT\n')
p.stdin.flush()
try:
    out, err = p.communicate(timeout=8)
    print('STDOUT len:', len(out), 'first 16:', out[:16].hex() if out else '(empty)')
    print('STDERR:', err.decode(errors='replace')[:3000])
    print('rc:', p.returncode)
except subprocess.TimeoutExpired:
    p.kill()
    print('TIMED OUT')
