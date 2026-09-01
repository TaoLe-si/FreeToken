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
szP = len(packed)
szA = K * 4
szS = M * ns * 4
szB = M * ns * 4

import subprocess
for a_val in [0.01, 0.05, 0.10, 0.20]:
    act = np.full(K, a_val, dtype='float32')
    hdr = struct.pack('<IIIIII', M, K, szP, szA, szS, szB)
    payload = hdr + packed + act.tobytes() + bytes(szS) + bytes(szB)
    
    p = subprocess.Popen([r'E:\FreeToken\benchmarks\cpu_moe_microbench\t_mxfp4_gemv_server.exe'],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         cwd=r'E:\FreeToken\benchmarks\cpu_moe_microbench')
    out, err = p.communicate(payload, timeout=10)
    sz = struct.unpack('<I', out[:4])[0]
    val = struct.unpack('<f', out[4:8])[0]
    print(f'  a={a_val:+.2f}: outv={val:+.6f}')
    p.wait()
