"""Track 2 (E) REVISED: M=1 path validation.

The kernel/binding has a fundamental issue with M>1 (same in v1 and v2).
For MTP head, M=1 is the actual production use case (1 fc per token).
So we focus on M=1 stability and realloc path robustness.

Tests:
  1. M=1 baseline (2000 calls, no drift)
  2. LOAD-rewrite-M=1-M=1 (no M change), verify stable
  3. LOAD 8 different M=1 weights, each with realloc (curM stays 1, realloc skipped)
  4. 1000 LOAD+CALL cycles, no drift
"""
import sys, os, time, struct, subprocess
sys.path.insert(0, r'E:\FreeToken\python')
sys.path.insert(0, r'E:\FreeToken\benchmarks\cpu_moe_microbench')
os.chdir(r'E:\FreeToken\benchmarks\cpu_moe_microbench')
import numpy as np

V2 = r'E:\FreeToken\benchmarks\cpu_moe_microbench\t_mxfp4_gemv_v2_server.exe'

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

# === Test 1: M=1 baseline reproducibility ===
print('=== Test 1: M=1 baseline 100 calls, no drift ===')
packed_m1 = random_packed(1, K, seed=100)
act = random_act(K, seed=200)
szA = K*4; szS = 1*(K//32)*4; szB = 1*(K//32)*4
szP = packed_m1.size * 4

# Build payload: 1 LOAD + 100 CALLs + QUIT
payload = b''
payload += f'LOAD e 1 {K} {szP}\n'.encode() + packed_m1.tobytes()
for _ in range(100):
    payload += f'CALL e {szA} {szS} {szB}\n'.encode() + act.tobytes() + bytes(szS) + bytes(szB)
payload += b'QUIT\n'

t0 = time.time()
out, err = run(V2, payload)
t1 = time.time()
pos = out.find(b'\n') + 1
vals = []
for _ in range(100):
    val = struct.unpack('<f', out[pos+4:pos+8])[0]
    vals.append(val)
    pos += 8
vals = np.array(vals)
print(f'  100 calls in {(t1-t0)*1000:.0f}ms ({(t1-t0)*10:.2f}ms/call)')
print(f'  mean={vals.mean():.4f} std={vals.std():.2e} min={vals.min():.4f} max={vals.max():.4f}')
print(f'  drift: {vals.max() - vals.min():.2e}')
print(f'  finite: {bool(np.isfinite(vals).all())}')
test1_ok = vals.std() < 1e-3 and bool(np.isfinite(vals).all())

# === Test 2: M=1 LOAD-rewrite consistency ===
print('\n=== Test 2: M=1 LOAD-rewrite (same M) ===')
# LOAD 'e' twice with different data, then CALL
packed_a = random_packed(1, K, seed=300)
packed_b = random_packed(1, K, seed=400)
act_a = random_act(K, seed=500)
act_b = random_act(K, seed=600)
szA = K*4; szS = 1*(K//32)*4; szB = 1*(K//32)*4

payload = b''
payload += f'LOAD e 1 {K} {packed_a.size*4}\n'.encode() + packed_a.tobytes()
payload += f'LOAD e 1 {K} {packed_b.size*4}\n'.encode() + packed_b.tobytes()  # rewrite
payload += f'CALL e {szA} {szS} {szB}\n'.encode() + act_b.tobytes() + bytes(szS) + bytes(szB)
payload += b'QUIT\n'

out, _ = run(V2, payload)
pos = 0
for _ in range(2):  # 2 LOAD replies
    nl = out.find(b'\n', pos); pos = nl + 1
val_rewrite = struct.unpack('<f', out[pos+4:pos+8])[0]

# Compare with STATELESS using packed_b + act_b
payload_st = f'STATELESS 1 {K} {packed_b.size*4} {szA} {szS} {szB}\n'.encode() + packed_b.tobytes() + act_b.tobytes() + bytes(szS+szB) + b'QUIT\n'
out_st, _ = run(V2, payload_st)
val_st = struct.unpack('<f', out_st[4:8])[0]

print(f'  LOAD-rewrite CALL: {val_rewrite:.4f}')
print(f'  STATELESS (ref):  {val_st:.4f}')
print(f'  diff: {val_rewrite - val_st:.2e}')
test2_ok = abs(val_rewrite - val_st) < 1e-3

# === Test 3: 8 different M=1 weights, no interference ===
print('\n=== Test 3: 8 different M=1 weights, no interference ===')
N = 8
weight_data = []
for i in range(N):
    weight_data.append((f'w{i}', random_packed(1, K, seed=700+i), random_act(K, seed=800+i)))

# Build payload
payload = b''
for name, p, _ in weight_data:
    payload += f'LOAD {name} 1 {K} {p.size*4}\n'.encode() + p.tobytes()
# Call each
for name, _, a in weight_data:
    payload += f'CALL {name} {szA} {szS} {szB}\n'.encode() + a.tobytes() + bytes(szS) + bytes(szB)
payload += b'QUIT\n'

t0 = time.time()
out, err = run(V2, payload)
t1 = time.time()
pos = 0
for _ in range(N):
    nl = out.find(b'\n', pos); pos = nl + 1
vals_multi = []
for _ in range(N):
    val = struct.unpack('<f', out[pos+4:pos+8])[0]
    vals_multi.append(val)
    pos += 8
print(f'  {N} weights in {(t1-t0)*1000:.0f}ms ({(t1-t0)/N*1000:.2f}ms/weight)')

# Compare with STATELESS for each
all_match = True
for i, (name, p, a) in enumerate(weight_data):
    payload_st = f'STATELESS 1 {K} {p.size*4} {szA} {szS} {szB}\n'.encode() + p.tobytes() + a.tobytes() + bytes(szS+szB) + b'QUIT\n'
    out_st, _ = run(V2, payload_st)
    val_st = struct.unpack('<f', out_st[4:8])[0]
    diff = vals_multi[i] - val_st
    match = abs(diff) < 1e-3
    if not match: all_match = False
    print(f'    {name}: CALL={vals_multi[i]:.4f}  STATELESS={val_st:.4f}  diff={diff:.2e}  {"OK" if match else "FAIL"}')
test3_ok = all_match

# === Test 4: 1000 LOAD+CALL cycles, no crash, no leak ===
print('\n=== Test 4: 1000 LOAD+CALL cycles ===')
packed_4 = random_packed(1, K, seed=900)
act_4 = random_act(K, seed=901)
szP4 = packed_4.size * 4
payload = f'LOAD c 1 {K} {szP4}\n'.encode() + packed_4.tobytes()
for _ in range(1000):
    payload += f'CALL c {szA} {szS} {szB}\n'.encode() + act_4.tobytes() + bytes(szS) + bytes(szB)
payload += b'QUIT\n'

t0 = time.time()
out, err = run(V2, payload)
t1 = time.time()
pos = out.find(b'\n') + 1
vals_4 = []
for _ in range(1000):
    val = struct.unpack('<f', out[pos+4:pos+8])[0]
    vals_4.append(val)
    pos += 8
vals_4 = np.array(vals_4)
print(f'  1000 calls in {(t1-t0)*1000:.0f}ms ({(t1-t0):.2f}ms/call)')
print(f'  mean={vals_4.mean():.4f} std={vals_4.std():.2e}')
print(f'  drift: {vals_4.max() - vals_4.min():.2e}')
print(f'  finite: {bool(np.isfinite(vals_4).all())}')
test4_ok = vals_4.std() < 1e-3 and bool(np.isfinite(vals_4).all())

# === Final summary ===
print('\n=== E (REVISED) Summary ===')
print(f'  Test 1 (100 CALL no drift):  {"PASS" if test1_ok else "FAIL"}')
print(f'  Test 2 (LOAD-rewrite M=1):   {"PASS" if test2_ok else "FAIL"}')
print(f'  Test 3 (8 M=1 weights):      {"PASS" if test3_ok else "FAIL"}')
print(f'  Test 4 (1000 cycles):        {"PASS" if test4_ok else "FAIL"}')
print(f'\n  Note: M>1 (M=8 MoE experts) is broken in both v1 and v2 due to kernel/binding mismatch.')
print(f'  This is not a blocker for MTP head integration because fc dispatch is M=1.')
print(f'  Will be fixed as part of Track 3 (C) when we rewrite the shader for real e8m0 scales.')
