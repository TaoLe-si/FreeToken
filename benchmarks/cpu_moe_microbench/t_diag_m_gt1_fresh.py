"""Each M=N in a fresh server process (no realloc within one process)."""
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
act = random_act(K, seed=999)

# M=1 row 0 alone in fresh v2
print('=== Fresh v2 server, M=1 ===')
packed_r0 = random_packed(1, K, seed=10)
szP = packed_r0.size*4; szA = K*4; szS = 1*(K//32)*4; szB = 1*(K//32)*4
v2_payload = f'STATELESS 1 {K} {szP} {szA} {szS} {szB}\n'.encode() + packed_r0.tobytes() + act.tobytes() + b'\x00'*(szS+szB) + b'QUIT\n'
out, _ = run(V2, v2_payload)
v2_m1_r0 = struct.unpack('<f', out[4:8])[0]
print(f'  v2 M=1 (row 0): {v2_m1_r0:.4f}')

# M=2 (rows 0, 1) in fresh v2
print('\n=== Fresh v2 server, M=2 ===')
packed_r1 = random_packed(1, K, seed=11)
packed_2 = np.zeros((2, K // 8), dtype=np.uint32)
packed_2[0] = packed_r0[0]
packed_2[1] = packed_r1[0]
szP = packed_2.size*4; szA = K*4; szS = 2*(K//32)*4; szB = 2*(K//32)*4
v2_payload = f'STATELESS 2 {K} {szP} {szA} {szS} {szB}\n'.encode() + packed_2.tobytes() + act.tobytes() + b'\x00'*(szS+szB) + b'QUIT\n'
out, _ = run(V2, v2_payload)
v2_m2 = np.frombuffer(out[4:4+8], dtype=np.float32).copy()
print(f'  v2 M=2: {v2_m2}')

# M=1 (just row 1) in fresh v2
print('\n=== Fresh v2 server, M=1 (just row 1) ===')
packed_only_r1 = random_packed(1, K, seed=11)
szP = packed_only_r1.size*4; szA = K*4; szS = 1*(K//32)*4; szB = 1*(K//32)*4
v2_payload = f'STATELESS 1 {K} {szP} {szA} {szS} {szB}\n'.encode() + packed_only_r1.tobytes() + act.tobytes() + b'\x00'*(szS+szB) + b'QUIT\n'
out, _ = run(V2, v2_payload)
v2_m1_r1 = struct.unpack('<f', out[4:8])[0]
print(f'  v2 M=1 (row 1 alone): {v2_m1_r1:.4f}')

print(f'\n  v2 M=2[0] == v2 M=1 (row 0)? {abs(v2_m2[0] - v2_m1_r0) < 1e-3}  diff={v2_m2[0] - v2_m1_r0:.2e}')
print(f'  v2 M=2[1] == v2 M=1 (row 1)? {abs(v2_m2[1] - v2_m1_r1) < 1e-3}  diff={v2_m2[1] - v2_m1_r1:.2e}')

# Same test in v1
print('\n=== Fresh v1 server, M=2 ===')
hdr = struct.pack('<IIIIII', 2, K, packed_2.size*4, K*4, 2*(K//32)*4, 2*(K//32)*4)
v1_payload = hdr + packed_2.tobytes() + act.tobytes() + b'\x00'*(2*(K//32)*4)*2
out, _ = run(V1, v1_payload)
v1_m2 = np.frombuffer(out[4:4+8], dtype=np.float32).copy()
print(f'  v1 M=2: {v1_m2}')

# v1 M=1 row 0
hdr = struct.pack('<IIIIII', 1, K, packed_r0.size*4, K*4, 1*(K//32)*4, 1*(K//32)*4)
v1_payload = hdr + packed_r0.tobytes() + act.tobytes() + b'\x00'*(1*(K//32)*4)*2
out, _ = run(V1, v1_payload)
v1_m1_r0 = struct.unpack('<f', out[4:8])[0]
print(f'  v1 M=1 (row 0): {v1_m1_r0:.4f}')

# v1 M=1 row 1
hdr = struct.pack('<IIIIII', 1, K, packed_only_r1.size*4, K*4, 1*(K//32)*4, 1*(K//32)*4)
v1_payload = hdr + packed_only_r1.tobytes() + act.tobytes() + b'\x00'*(1*(K//32)*4)*2
out, _ = run(V1, v1_payload)
v1_m1_r1 = struct.unpack('<f', out[4:8])[0]
print(f'  v1 M=1 (row 1): {v1_m1_r1:.4f}')

print(f'\n  v1 M=2[0] == v1 M=1 (row 0)? {abs(v1_m2[0] - v1_m1_r0) < 1e-3}  diff={v1_m2[0] - v1_m1_r0:.2e}')
print(f'  v1 M=2[1] == v1 M=1 (row 1)? {abs(v1_m2[1] - v1_m1_r1) < 1e-3}  diff={v1_m2[1] - v1_m1_r1:.2e}')
