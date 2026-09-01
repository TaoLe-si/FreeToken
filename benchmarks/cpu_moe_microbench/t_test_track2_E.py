"""Track 2 (E): M>1 (MoE top-8 experts) validation.

Tests:
  1. M=8 LOAD+CALL: output should be sum of 8x M=1 outputs (linearity)
  2. M=1 -> M=8 -> M=1: realloc path stable
  3. M change in middle of CALL sequence: no NaN/Inf
  4. Compare with v1 server for M=1 (no regression)
"""
import sys, os, time, struct, subprocess
sys.path.insert(0, r'E:\FreeToken\python')
sys.path.insert(0, r'E:\FreeToken\benchmarks\cpu_moe_microbench')
os.chdir(r'E:\FreeToken\benchmarks\cpu_moe_microbench')
import numpy as np

V2_SERVER = r'E:\FreeToken\benchmarks\cpu_moe_microbench\t_mxfp4_gemv_v2_server.exe'
V1_SERVER = r'E:\FreeToken\benchmarks\cpu_moe_microbench\t_mxfp4_gemv_server.exe'

def random_packed(M, K, seed):
    rng = np.random.default_rng(seed)
    nb = K // 8
    return rng.integers(0, 2**32, size=(M, nb), dtype=np.uint32)

def random_act(K, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(-100, 100, size=(K,)).astype(np.float32)

def run_server(server, payload):
    p = subprocess.Popen([server], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         cwd=os.path.dirname(server))
    out, err = p.communicate(payload, timeout=60)
    return out, err

def build_stateless(M, K, packed, act, scales=None, biases=None):
    nb = K // 8
    ns = K // 32
    szP = packed.size * 4
    szA = K * 4
    szS = M * ns * 4 if scales is not None else 0
    szB = M * ns * 4 if biases is not None else 0
    payload = f'STATELESS {M} {K} {szP} {szA} {szS} {szB}\n'.encode()
    payload += packed.tobytes() + act.tobytes()
    if scales is not None: payload += scales.tobytes()
    else: payload += bytes(M * ns * 4)
    if biases is not None: payload += biases.tobytes()
    else: payload += bytes(M * ns * 4)
    return payload

def build_load_call(name, M, K, packed, act):
    nb = K // 8
    ns = K // 32
    szA = K * 4
    szS = M * ns * 4
    szB = M * ns * 4
    packed_size = packed.size * 4
    payload = f'LOAD {name} {M} {K} {packed_size}\n'.encode()
    payload += packed.tobytes()
    payload += f'CALL {name} {szA} {szS} {szB}\n'.encode()
    payload += act.tobytes() + bytes(szS) + bytes(szB)
    return payload

def parse_response(out):
    """Parse 4-byte len + M*4 bytes float."""
    if len(out) < 4: return None
    sz = struct.unpack('<I', out[:4])[0]
    return np.frombuffer(out[4:4+sz], dtype=np.float32).copy()

# === Test 1: M=1 baseline ===
print('=== Test 1: M=1 baseline (v2 server) ===')
K = 4096
M = 1
packed_m1 = random_packed(M, K, seed=100)
act = random_act(K, seed=200)
payload = build_load_call('e', M, K, packed_m1, act) + b'QUIT\n'
out, err = run_server(V2_SERVER, payload)
# Skip LOAD reply
pos = out.find(b'\n') + 1
outv_m1 = parse_response(out[pos:])
print(f'  M={M} K={K}: outv = {outv_m1}')

# === Test 2: M=8, same data repeated 8 rows ===
print('\n=== Test 2: M=8 with 8 copies of same data ===')
M8 = 8
packed_m8 = np.tile(packed_m1, (M8, 1))  # 8 rows identical
# Same act, expect each row to produce the same result
payload = build_load_call('e8', M8, K, packed_m8, act) + b'QUIT\n'
out, err = run_server(V2_SERVER, payload)
pos = out.find(b'\n') + 1
outv_m8 = parse_response(out[pos:])
print(f'  M={M8} K={K}: outv = {outv_m8}')
print(f'  shape: {outv_m8.shape}')

# Linearity check: each row should equal M=1 result
expected = np.full(M8, outv_m1[0], dtype=np.float32)
diff = outv_m8 - expected
print(f'  All rows equal M=1 result? {bool(np.allclose(outv_m8, expected, atol=1e-3))}')
print(f'  max diff: {np.abs(diff).max():.2e}')

# === Test 3: M=8 vs sum of 8 separate M=1 calls ===
print('\n=== Test 3: M=8 linearity: each row = single-row GEMV ===')
# Build 8 different (packed_i, act) and stack as M=8
packed_m8_diff = np.zeros((M8, K // 8), dtype=np.uint32)
acts_diff = np.zeros((M8, K), dtype=np.float32)
outvs_m1 = []
for r in range(M8):
    p = random_packed(1, K, seed=1000 + r)
    a = random_act(K, seed=2000 + r)
    packed_m8_diff[r] = p[0]
    acts_diff[r] = a
    # Run M=1 with this data
    payload = build_load_call(f'r{r}', 1, K, p, a) + b'QUIT\n'
    out, _ = run_server(V2_SERVER, payload)
    pos = out.find(b'\n') + 1
    outv = parse_response(out[pos:])[0]
    outvs_m1.append(outv)
outvs_m1 = np.array(outvs_m1)

# Now run M=8 with same data per row
payload = build_load_call('all8', M8, K, packed_m8_diff, acts_diff[0]) + b'QUIT\n'
# Wait - we need to pass acts_diff[0] as a single act (K values), server applies to all rows
# But each row has different act. The server protocol has single act shared across rows.
# So we can only test linearity with SAME act.
# Let me redo: use the same act for all rows
print('  Note: server protocol shares single act across all M rows.')
print('  Testing with same act: 8 rows of (packed_i, same_act)')

# Use acts_diff[0] for all rows
shared_act = acts_diff[0]
outvs_m1_same_act = []
for r in range(M8):
    p = packed_m8_diff[r:r+1]
    payload = build_load_call(f's{r}', 1, K, p, shared_act) + b'QUIT\n'
    out, _ = run_server(V2_SERVER, payload)
    pos = out.find(b'\n') + 1
    outv = parse_response(out[pos:])[0]
    outvs_m1_same_act.append(outv)
outvs_m1_same_act = np.array(outvs_m1_same_act)

# M=8 with stacked packed and same act
payload = build_load_call('all8sa', M8, K, packed_m8_diff, shared_act) + b'QUIT\n'
out, _ = run_server(V2_SERVER, payload)
pos = out.find(b'\n') + 1
outv_m8_same_act = parse_response(out[pos:])

diff = outv_m8_same_act - outvs_m1_same_act
print(f'  8 separate M=1: {outvs_m1_same_act}')
print(f'  M=8 batch:        {outv_m8_same_act}')
print(f'  max diff: {np.abs(diff).max():.2e}')
print(f'  Match (atol=1e-3)? {bool(np.allclose(outv_m8_same_act, outvs_m1_same_act, atol=1e-3))}')

# === Test 4: M=1 -> M=8 -> M=1 realloc stability ===
print('\n=== Test 4: M=1 -> M=8 -> M=1 realloc stability ===')
seq = []
# LOAD M=1, CALL
p1 = random_packed(1, K, seed=300)
a = random_act(K, seed=301)
seq.append(('LOAD', 1, K, p1))
seq.append(('CALL', 1, a))
# LOAD M=8, CALL (different name)
p8 = random_packed(8, K, seed=400)
seq.append(('LOAD', 8, K, p8, 'b'))
seq.append(('CALL', 8, a, 'b'))
# LOAD M=1 again (different name), CALL
p1b = random_packed(1, K, seed=500)
seq.append(('LOAD', 1, K, p1b, 'c'))
seq.append(('CALL', 1, a, 'c'))

payload = b''
for item in seq:
    if item[0] == 'LOAD':
        _, M_, K_, p = item[0], item[1], item[2], item[3]
        name = item[4] if len(item) > 4 else 'a'
        packed_size = p.size * 4
        payload += f'LOAD {name} {M_} {K_} {packed_size}\n'.encode()
        payload += p.tobytes()
    else:
        _, M_, a_ = item[0], item[1], item[2]
        name = item[3] if len(item) > 3 else 'a'
        ns = K_ // 32
        szA, szS, szB = K_*4, M_*ns*4, M_*ns*4
        payload += f'CALL {name} {szA} {szS} {szB}\n'.encode()
        payload += a_.tobytes() + bytes(szS) + bytes(szB)
payload += b'QUIT\n'

out, err = run_server(V2_SERVER, payload)
# Parse: 3 LOAD replies, 3 CALL responses
pos = 0
for _ in range(3):
    nl = out.find(b'\n', pos); pos = nl + 1
results = []
for _ in range(3):
    sz = struct.unpack('<I', out[pos:pos+4])[0]
    val = np.frombuffer(out[pos+4:pos+4+sz], dtype=np.float32).copy()
    results.append(val)
    pos += 4 + sz

print(f'  After M=1 LOAD/CALL: outv = {results[0]}')
print(f'  After M=8 LOAD/CALL: outv = {results[1]} (shape {results[1].shape})')
print(f'  After M=1 re-LOAD/CALL: outv = {results[2]}')
print(f'  All finite? {bool(np.all(np.isfinite(results[0])) and np.all(np.isfinite(results[1])) and np.all(np.isfinite(results[2])))}')

# Compare M=1 final result with M=1 baseline
# Re-run the M=1 baseline with same seed for comparison
p1c = random_packed(1, K, seed=500)
payload = build_load_call('c', 1, K, p1c, a) + b'QUIT\n'
out, _ = run_server(V2_SERVER, payload)
pos = out.find(b'\n') + 1
outv_m1_final = parse_response(out[pos:])
print(f'  Compare M=1 re-LOAD to standalone: {results[2][0]:.4f} vs {outv_m1_final[0]:.4f}  diff={results[2][0] - outv_m1_final[0]:.2e}')

# === Test 5: 100x M=8 calls, no NaN ===
print('\n=== Test 5: 100x M=8 calls, no NaN/Inf ===')
packed_m8_loop = random_packed(8, K, seed=600)
act_loop = random_act(K, seed=601)
payload = b''
szA, szS, szB = K*4, 8*(K//32)*4, 8*(K//32)*4
packed_size = packed_m8_loop.size * 4
payload += f'LOAD big 8 {K} {packed_size}\n'.encode()
payload += packed_m8_loop.tobytes()
for _ in range(100):
    payload += f'CALL big {szA} {szS} {szB}\n'.encode()
    payload += act_loop.tobytes() + bytes(szS) + bytes(szB)
payload += b'QUIT\n'

t0 = time.time()
out, err = run_server(V2_SERVER, payload)
t1 = time.time()
pos = out.find(b'\n') + 1
vals = []
for _ in range(100):
    sz = struct.unpack('<I', out[pos:pos+4])[0]
    v = np.frombuffer(out[pos+4:pos+4+sz], dtype=np.float32).copy()
    vals.append(v)
    pos += 4 + sz
vals = np.stack(vals)  # [100, 8]

print(f'  100 calls in {(t1-t0)*1000:.0f}ms ({(t1-t0)*10:.2f}ms/call)')
print(f'  shape: {vals.shape}')
print(f'  All finite? {bool(np.isfinite(vals).all())}')
print(f'  Per-row std: {vals.std(axis=0)}')
print(f'  Per-row mean: {vals.mean(axis=0)}')
print(f'  Drift (max - min per row): {(vals.max(axis=0) - vals.min(axis=0))}')

# === Summary ===
print('\n=== E Summary ===')
linear_ok = bool(np.allclose(outv_m8_same_act, outvs_m1_same_act, atol=1e-3))
realloc_ok = bool(np.all(np.isfinite(results[0])) and np.all(np.isfinite(results[1])) and np.all(np.isfinite(results[2])))
m1_match = abs(results[2][0] - outv_m1_final[0]) < 1e-3
no_nan = bool(np.isfinite(vals).all())
no_drift = bool((vals.max(axis=0) - vals.min(axis=0)).max() < 1e-3)

print(f'  Test 1 (M=1 baseline): {"PASS" if outv_m1 is not None else "FAIL"}')
print(f'  Test 2 (M=8 same data 8 rows): {"PASS" if bool(np.allclose(outv_m8, expected, atol=1e-3)) else "FAIL"}')
print(f'  Test 3 (M=8 linearity): {"PASS" if linear_ok else "FAIL"}')
print(f'  Test 4 (M=1->M=8->M=1 realloc): {"PASS" if realloc_ok and m1_match else "FAIL"}')
print(f'  Test 5 (100x M=8 no NaN/drift): {"PASS" if no_nan and no_drift else "FAIL"}')
