import sys, os, time, struct
sys.path.insert(0, r'E:\FreeToken\python')
sys.path.insert(0, r'E:\FreeToken\benchmarks\cpu_moe_microbench')
os.chdir(r'E:\FreeToken\benchmarks\cpu_moe_microbench')
import safetensors.torch, numpy as np

state = safetensors.torch.load_file(r'E:\models\Qwen3.6-35B-A3B-MXFP4-MTP\model-00022-of-00023.safetensors')
fc_w = state['mtp.fc.weight'].numpy()
M, K = 1, 4096
ns = K // 32
nb = K // 8
packed = fc_w[:1].tobytes()  # M rows, each nb*4 bytes
print('packed_size=', len(packed), 'expected M*nb*4=', M*nb*4)

# Build payload: LOAD + packed + CALL + act(0.05) + scales(zeros) + biases(zeros) + QUIT
szA, szS, szB = K * 4, M * ns * 4, M * ns * 4
act = np.full(K, 0.05, dtype='float32')
payload = b'LOAD fc ' + str(M).encode() + b' ' + str(K).encode() + b' ' + str(len(packed)).encode() + b'\n'
payload += packed
payload += b'CALL fc ' + str(szA).encode() + b' ' + str(szS).encode() + b' ' + str(szB).encode() + b'\n'
payload += act.tobytes() + bytes(szS) + bytes(szB)
payload += b'QUIT\n'

print('payload total size:', len(payload))

import subprocess
try:
    p = subprocess.Popen([r'E:\FreeToken\benchmarks\cpu_moe_microbench\t_mxfp4_gemv_v2_server.exe'],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         cwd=r'E:\FreeToken\benchmarks\cpu_moe_microbench')
    out, err = p.communicate(payload, timeout=10)
    print('STDOUT len:', len(out))
    if len(out) >= 8:
        # First response is from CALL: 4-byte len + M*4 floats
        sz = struct.unpack('<I', out[:4])[0]
        print('response sz =', sz)
        val = struct.unpack('<f', out[4:8])[0]
        print(f'outv[0] = {val:.6f}')
        # Expected: sum(nibble * act) for M=1, K=4096 with all act=0.05
        # For MXFP4 e2m1, expected = sum of all weight values * 0.05
        # But we also have scales and biases. The stateless server got 4.69 with no scales/biases
        # and only act. So 0.05 * sum(weights) = 4.69 -> sum(weights) = 93.8
    print('STDERR:', err.decode(errors='replace')[:3000])
    print('rc:', p.returncode)
except subprocess.TimeoutExpired:
    p.kill()
    print('TIMED OUT')
