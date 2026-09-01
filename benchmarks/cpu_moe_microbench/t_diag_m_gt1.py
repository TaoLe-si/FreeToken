"""Diagnose M>1 issue in v2 server.

Compare v1 vs v2 for M=1, M=2 with the same data.
"""
import sys, os, struct, subprocess
sys.path.insert(0, r'E:\FreeToken\python')
sys.path.insert(0, r'E:\FreeToken\benchmarks\cpu_moe_microbench')
os.chdir(r'E:\FreeToken\benchmarks\cpu_moe_microbench')
import numpy as np

V2 = r'E:\FreeToken\benchmarks\cpu_moe_microbench\t_mxfp4_gemv_v2_server.exe'
V1 = r'E:\FreeToken\benchmarks\cpu_moe_microbench\t_mxfp4_gemv_server.exe'

def run(server, payload):
    p = subprocess.Popen([server], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         cwd=os.path.dirname(server))
    return p.communicate(payload, timeout=60)

def random_packed(M, K, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2**32, size=(M, K // 8), dtype=np.uint32)

def random_act(K, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(-100, 100, size=(K,)).astype(np.float32)

K = 4096
act = random_act(K, seed=1)
packed1 = random_packed(1, K, seed=2)

# === M=1: v1 vs v2 ===
print('=== M=1 ===')
# v1 protocol
hdr = struct.pack('<IIIIII', 1, K, packed1.size*4, K*4, 0, 0)
v1_payload = hdr + packed1.tobytes() + act.tobytes() + b'\x00' * 0
out1, _ = run(V1, v1_payload)
v1_m1 = struct.unpack('<f', out1[4:8])[0]
print(f'  v1 M=1: {v1_m1:.4f}')

# v2 STATELESS
szP = packed1.size*4; szA = K*4; szS = 1*(K//32)*4; szB = 1*(K//32)*4
v2_payload = f'STATELESS 1 {K} {szP} {szA} {szS} {szB}\n'.encode() + packed1.tobytes() + act.tobytes() + b'\x00'*(szS+szB) + b'QUIT\n'
out2, _ = run(V2, v2_payload)
v2_m1 = struct.unpack('<f', out2[4:8])[0]
print(f'  v2 M=1: {v2_m1:.4f}')
print(f'  match: {abs(v1_m1 - v2_m1) < 1e-3}')

# === M=2 with same first row + new second row ===
print('\n=== M=2 ===')
packed2 = np.zeros((2, K // 8), dtype=np.uint32)
packed2[0] = packed1[0]  # same as M=1
packed2[1] = random_packed(1, K, seed=3)[0]  # different

# v1: STATELESS for M=2
hdr = struct.pack('<IIIIII', 2, K, packed2.size*4, K*4, 2*(K//32)*4, 2*(K//32)*4)
v1_payload = hdr + packed2.tobytes() + act.tobytes() + b'\x00'*(2*(K//32)*4)*2
out1, _ = run(V1, v1_payload)
v1_m2 = np.frombuffer(out1[4:4+8], dtype=np.float32).copy()
print(f'  v1 M=2: {v1_m2}')

# v2: STATELESS for M=2
szP = packed2.size*4; szA = K*4; szS = 2*(K//32)*4; szB = 2*(K//32)*4
v2_payload = f'STATELESS 2 {K} {szP} {szA} {szS} {szB}\n'.encode() + packed2.tobytes() + act.tobytes() + b'\x00'*(szS+szB) + b'QUIT\n'
out2, _ = run(V2, v2_payload)
v2_m2 = np.frombuffer(out2[4:4+8], dtype=np.float32).copy()
print(f'  v2 M=2: {v2_m2}')

# v1 M=1 of row 2 alone
hdr = struct.pack('<IIIIII', 1, K, (K//8)*4, K*4, 1*(K//32)*4, 1*(K//32)*4)
v1_payload = hdr + packed2[1:2].tobytes() + act.tobytes() + b'\x00'*(1*(K//32)*4)*2
out1, _ = run(V1, v1_payload)
v1_m1_row2 = struct.unpack('<f', out1[4:8])[0]
print(f'  v1 M=1 row2: {v1_m1_row2:.4f}')

print(f'\n  v1 M=2 row 0 = v1 M=1 row 0? {abs(v1_m2[0] - v1_m1) < 1e-3}')
print(f'  v1 M=2 row 1 = v1 M=1 row 1? {abs(v1_m2[1] - v1_m1_row2) < 1e-3}')
print(f'  v2 M=2 row 0 = v2 M=1?      {abs(v2_m2[0] - v2_m1) < 1e-3}')
print(f'  v2 M=2 row 1 = v1 M=1 row 1? {abs(v2_m2[1] - v1_m1_row2) < 1e-3}')
