import sys, os, time
sys.path.insert(0, r'E:\\\\FreeToken\\\\python')
sys.path.insert(0, r'E:\\\\FreeToken\\\\benchmarks\\\\cpu_moe_microbench')
os.chdir(r'E:\\\\FreeToken\\\\benchmarks\\\\cpu_moe_microbench')
import numpy as np
import safetensors.torch
state = safetensors.torch.load_file(r'E:\\\\models\\\\Qwen3.6-35B-A3B-MXFP4-MTP\\\\model-00022-of-00023.safetensors')
fc_w = state['mtp.fc.weight'].numpy()
M, K = 1, 4096
ns = K // 32
packed = fc_w[:1].tobytes()
act = np.full(K, 0.05, dtype='float32').view(np.int32)
szA, szS, szB = K * 4, M * ns * 4, M * ns * 4
import struct
hdr = struct.pack('<IIIIII', M, K, len(packed), szA, szS, szB)
payload = hdr + packed + act.tobytes() + bytes(szS) + bytes(szB)
import subprocess
p = subprocess.Popen([r'E:\\\\FreeToken\\\\benchmarks\\\\cpu_moe_microbench\\\\t_mxfp4_gemv_server.exe'],
                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                     cwd=r'E:\\\\FreeToken\\\\benchmarks\\\\cpu_moe_microbench')
try:
    out, err = p.communicate(payload, timeout=8)
    print('STDOUT len:', len(out), 'first 8:', out[:8].hex() if out else '(empty)')
    if len(out) >= 8:
        sz = struct.unpack('<I', out[:4])[0]
        val = struct.unpack('<f', out[4:8])[0]
        print(f'sz={sz} outv={val}')
    print('STDERR:', err.decode(errors='replace')[:1000])
    print('rc:', p.returncode)
except subprocess.TimeoutExpired:
    p.kill()
    print('TIMED OUT')
