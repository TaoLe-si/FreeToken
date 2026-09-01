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
szA, szS, szB = K * 4, M * ns * 4, M * ns * 4

# Build payload: LOAD once + 5 CALLs with different act values
payload = b'LOAD fc ' + str(M).encode() + b' ' + str(K).encode() + b' ' + str(len(packed)).encode() + b'\n'
payload += packed
acts = [0.01, 0.05, 0.1, 0.2, -0.05]
for a in acts:
    act = np.full(K, a, dtype='float32')
    payload += b'CALL fc ' + str(szA).encode() + b' ' + str(szS).encode() + b' ' + str(szB).encode() + b'\n'
    payload += act.tobytes() + bytes(szS) + bytes(szB)
payload += b'QUIT\n'

print('payload total size:', len(payload))

# Reference: stateless server result for each act
# From earlier: act=0.05 -> outv=4.6925 -> sum(weights) = 93.85
# For linear case (no scales/biases): outv[a] = a * sum(weights) - 0 (rowB=0)
sum_w = 4.6925 / 0.05
print(f'expected sum_w = {sum_w:.4f}')

import subprocess
p = subprocess.Popen([r'E:\FreeToken\benchmarks\cpu_moe_microbench\t_mxfp4_gemv_v2_server.exe'],
                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                     cwd=r'E:\FreeToken\benchmarks\cpu_moe_microbench')
out, err = p.communicate(payload, timeout=30)
# Parse: LOAD reply (text "OK fc 1 4096\n") + N CALL responses (8 bytes each)
pos = 0
nl = out.find(b'\n', pos)
load_reply = out[pos:nl+1].decode()
print(f'LOAD reply: {load_reply}')
pos = nl + 1
print('CALL results:')
for i, a in enumerate(acts):
    if pos + 8 > len(out):
        print(f'  CALL {i} (a={a}): MISSING response')
        break
    sz = struct.unpack('<I', out[pos:pos+4])[0]
    val = struct.unpack('<f', out[pos+4:pos+8])[0]
    expected = a * sum_w
    print(f'  CALL {i} (a={a:+.2f}): outv={val:+.6f} expected={expected:+.6f} diff={val-expected:+.6f}')
    pos += 8

print('STDERR last 2000:', err.decode(errors='replace')[-2000:])
print('rc:', p.returncode)
