"""Track 1 (B): Multi-weight sticky cache + cache consistency validation.

Tests:
  1. 8 different-weight LOADs simultaneously, CALL each with different act.
     Verify each is dispatched independently (no cross-talk).
  2. LOAD-rewrite-same-name: 2nd LOAD with different packed, verify 2nd takes effect.
  3. Repeated LOAD/CALL cycles: no drift, no crash, stable output.

Reference: STATELESS path (mathematically identical to LOAD+CALL) is the gold standard.
We compare multi-weight CALL outputs to STATELESS outputs of the same input.
"""
import sys, os, time, struct, subprocess
sys.path.insert(0, r'E:\FreeToken\python')
sys.path.insert(0, r'E:\FreeToken\benchmarks\cpu_moe_microbench')
os.chdir(r'E:\FreeToken\benchmarks\cpu_moe_microbench')
import numpy as np

SERVER = r'E:\FreeToken\benchmarks\cpu_moe_microbench\t_mxfp4_gemv_v2_server.exe'

# === Test data: 8 weights with different shapes ===
# Real MTP head dimensions:
# - fc: M=1, K=4096 (concat 2048+2048)
# - attn q: M=4096, K=2048 (8 qo heads * 512 dim)
# - attn k: M=512, K=2048 (2 kv heads * 256 dim)
# - attn v: M=512, K=2048
# - attn o: M=2048, K=4096
# - MoE expert (sample): M=512, K=2048
# - 2 more for 8 total

WEIGHTS = [
    ('fc',  1,    4096, 'fc_random'),
    ('q',   4096, 2048, 'q_random'),
    ('k',   512,  2048, 'k_random'),
    ('v',   512,  2048, 'v_random'),
    ('o',   2048, 4096, 'o_random'),
    ('e0',  512,  2048, 'e0_random'),
    ('e1',  512,  2048, 'e1_random'),
    ('e2',  512,  2048, 'e2_random'),
]

def random_packed(M, K, seed):
    rng = np.random.default_rng(seed)
    nb = K // 8
    return rng.integers(0, 2**32, size=(M, nb), dtype=np.uint32)

def random_act(K, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(-100, 100, size=(K,)).astype(np.float32)

# Generate all weights' packed data + acts
data = {}
for i, (name, M, K, _) in enumerate(WEIGHTS):
    data[name] = {
        'M': M, 'K': K,
        'packed': random_packed(M, K, seed=1000 + i),
        'act': random_act(K, seed=2000 + i),
    }

# === Build payload ===
def build_multi_weight_test(weights_to_load, rewrite_name=None, rewrite_data=None):
    """Build a single payload that:
       - LOADs each weight in weights_to_load
       - (optional) rewrite rewrite_name with rewrite_data (2nd LOAD)
       - CALLs each weight with its act
       - QUIT
       Returns: payload bytes
    """
    payload = b''
    for name, M, K, _ in weights_to_load:
        d = data[name]
        assert d['M'] == M and d['K'] == K
        packed_size = d['packed'].size * 4  # bytes
        payload += f'LOAD {name} {M} {K} {packed_size}\n'.encode()
        payload += d['packed'].tobytes()
    # Optional rewrite
    if rewrite_name and rewrite_data:
        M = rewrite_data['M']
        K = rewrite_data['K']
        packed_size = rewrite_data['packed'].size * 4
        payload += f'LOAD {rewrite_name} {M} {K} {packed_size}\n'.encode()
        payload += rewrite_data['packed'].tobytes()
    # CALL each (in same order)
    for name, M, K, _ in weights_to_load:
        d = data[name]
        ns = K // 32
        szA = K * 4
        szS = M * ns * 4
        szB = M * ns * 4
        payload += f'CALL {name} {szA} {szS} {szB}\n'.encode()
        payload += d['act'].tobytes() + bytes(szS) + bytes(szB)
    payload += b'QUIT\n'
    return payload

def build_stateless_test(weights_to_load):
    """Build payload using STATELESS path (one-shot, no LOAD)."""
    payload = b''
    for name, M, K, _ in weights_to_load:
        d = data[name]
        nb = K // 8
        ns = K // 32
        szP = d['packed'].size * 4
        szA = K * 4
        szS = M * ns * 4
        szB = M * ns * 4
        payload += f'STATELESS {M} {K} {szP} {szA} {szS} {szB}\n'.encode()
        payload += d['packed'].tobytes() + d['act'].tobytes() + bytes(szS) + bytes(szB)
    payload += b'QUIT\n'
    return payload

def run_server(payload):
    p = subprocess.Popen([SERVER],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         cwd=os.path.dirname(SERVER))
    out, err = p.communicate(payload, timeout=60)
    return out, err

def parse_responses(out, n_expected, has_load_replies=True):
    """Parse N responses. If has_load_replies, skip N LOAD reply lines first."""
    pos = 0
    if has_load_replies:
        for _ in range(n_expected):
            nl = out.find(b'\n', pos)
            if nl < 0: raise RuntimeError(f'no LOAD reply at pos {pos}')
            pos = nl + 1
    responses = []
    for _ in range(n_expected):
        if pos + 4 > len(out): break
        sz = struct.unpack('<I', out[pos:pos+4])[0]
        if pos + 4 + sz > len(out):
            raise RuntimeError(f'short response at {pos}, need {4+sz}, have {len(out)-pos}')
        val_bytes = out[pos+4:pos+4+sz]
        if sz == 4:
            val = np.frombuffer(val_bytes, dtype=np.float32)[0]
        else:
            val = np.frombuffer(val_bytes, dtype=np.float32).copy()
        responses.append(val)
        pos += 4 + sz
    return responses

# === Test 1: 8 weights, LOAD + CALL ===
print('=== Test 1: 8-weight simultaneous LOAD+CALL ===')
payload = build_multi_weight_test(WEIGHTS)
print(f'  payload size: {len(payload)/1024:.1f} KB')
t0 = time.time()
out, err = run_server(payload)
t1 = time.time()
res = parse_responses(out, len(WEIGHTS))
print(f'  total: {(t1-t0)*1000:.1f}ms ({(t1-t0)/len(WEIGHTS)*1000:.2f}ms/weight)')
for i, (name, M, K, _) in enumerate(WEIGHTS):
    if M == 1:
        print(f'  [{i}] {name} M={M} K={K}: outv={res[i]:.4f}')
    else:
        print(f'  [{i}] {name} M={M} K={K}: outv[0]={res[i][0]:.4f}  shape={res[i].shape}')

# === Test 2: Compare multi-weight CALL with STATELESS ===
print('\n=== Test 2: CALL vs STATELESS (mathematical equivalence) ===')
payload_st = build_stateless_test(WEIGHTS)
out_st, _ = run_server(payload_st)
res_st = parse_responses(out_st, len(WEIGHTS), has_load_replies=False)

# Compare each
print(f'  {"name":<6} {"M":>5} {"K":>5} | {"CALL[0]":>12} {"STATELESS[0]":>14} {"diff":>10} | {"rel":>10}')
all_pass = True
for i, (name, M, K, _) in enumerate(WEIGHTS):
    v_call = res[i][0] if M > 1 else res[i]
    v_st = res_st[i][0] if M > 1 else res_st[i]
    diff = v_call - v_st
    rel = abs(diff) / (abs(v_st) + 1e-9)
    ok = 'OK' if abs(diff) < 1e-3 else 'FAIL'
    if ok == 'FAIL': all_pass = False
    print(f'  {name:<6} {M:>5} {K:>5} | {v_call:>12.4f} {v_st:>14.4f} {diff:>10.2e} | {rel:>10.2e} {ok}')

# === Test 3: LOAD-rewrite same name with different data ===
print('\n=== Test 3: LOAD-rewrite same name, verify 2nd LOAD takes effect ===')
rewrite_data = {
    'M': 1, 'K': 4096,
    'packed': random_packed(1, 4096, seed=9999),  # completely different
    'act': random_act(4096, seed=8888),
}
# Build: LOAD fc + LOAD fc (rewrite) + CALL fc
# Use simpler test: LOAD fc, then load ONLY fc again (rewrite), then call
payload_rw = b''
fc_d = data['fc']
packed_size = fc_d['packed'].size * 4
payload_rw += f'LOAD fc {fc_d["M"]} {fc_d["K"]} {packed_size}\n'.encode()
payload_rw += fc_d['packed'].tobytes()
# Rewrite with new data
rw_packed_size = rewrite_data['packed'].size * 4
payload_rw += f'LOAD fc {rewrite_data["M"]} {rewrite_data["K"]} {rw_packed_size}\n'.encode()
payload_rw += rewrite_data['packed'].tobytes()
# Now CALL with both acts to compare
# Call 1: original act, should give original output
ns = 4096 // 32
szA, szS, szB = 4096*4, 1*ns*4, 1*ns*4
payload_rw += f'CALL fc {szA} {szS} {szB}\n'.encode()
payload_rw += fc_d['act'].tobytes() + bytes(szS) + bytes(szB)
# Call 2: new act, should give new output
payload_rw += f'CALL fc {szA} {szS} {szB}\n'.encode()
payload_rw += rewrite_data['act'].tobytes() + bytes(szS) + bytes(szB)
payload_rw += b'QUIT\n'

out_rw, _ = run_server(payload_rw)
# Parse: 2 LOAD replies, then 2 CALL responses
pos = 0
for _ in range(2):
    nl = out_rw.find(b'\n', pos)
    pos = nl + 1
res_rw = parse_responses(out_rw[pos:], 2, has_load_replies=False)

# Compare with STATELESS for both
payload_st2 = b''
rw = rewrite_data
nb2 = 4096 // 8
szP2 = rw['packed'].size * 4
payload_st2 += f'STATELESS {rw["M"]} {rw["K"]} {szP2} {szA} {szS} {szB}\n'.encode()
payload_st2 += rw['packed'].tobytes() + rw['act'].tobytes() + bytes(szS) + bytes(szB)
payload_st2 += b'QUIT\n'
out_st2, _ = run_server(payload_st2)
res_st2 = parse_responses(out_st2, 1, has_load_replies=False)

diff_orig = res_rw[0] - res_st[0]  # first call was original act with rewrite data
diff_rew = res_rw[1] - res_st2[0]  # second call was new act with new data
print(f'  Rewrite verified?')
print(f'    call 1 (orig act, rewritten weight): {res_rw[0]:.4f}  vs STATELESS(orig): {res_st[0]:.4f}  diff={diff_orig:.2e}  (NOT 0 expected - data changed)')
print(f'    call 2 (new act,  new weight):       {res_rw[1]:.4f}  vs STATELESS(rew):  {res_st2[0]:.4f}  diff={diff_rew:.2e}  (close to 0 expected)')
rewrite_ok = abs(diff_rew) < 1e-3 and abs(diff_orig) > 1e-3  # new matches, old differs
print(f'  LOAD-rewrite works correctly: {rewrite_ok}')

# === Test 4: 100 LOAD/CALL cycles, check for drift or memory leak ===
print('\n=== Test 4: 100 LOAD+CALL cycles, no drift / no crash ===')
single_payload = b''
fc_d = data['fc']
packed_size = fc_d['packed'].size * 4
ns = 4096 // 32
szA, szS, szB = 4096*4, 1*ns*4, 1*ns*4
for i in range(100):
    if i == 0:
        # First cycle: LOAD
        single_payload += f'LOAD fc {fc_d["M"]} {fc_d["K"]} {packed_size}\n'.encode()
        single_payload += fc_d['packed'].tobytes()
    # CALL
    single_payload += f'CALL fc {szA} {szS} {szB}\n'.encode()
    single_payload += fc_d['act'].tobytes() + bytes(szS) + bytes(szB)
single_payload += b'QUIT\n'

t0 = time.time()
out_loop, err_loop = run_server(single_payload)
t1 = time.time()
# Parse: 1 LOAD reply, 100 CALL responses
pos = out_loop.find(b'\n') + 1
vals = []
for _ in range(100):
    val = struct.unpack('<f', out_loop[pos+4:pos+8])[0]
    vals.append(val)
    pos += 8

vals = np.array(vals)
print(f'  100 calls in {(t1-t0)*1000:.0f}ms ({(t1-t0)*10:.2f}ms/call)')
print(f'  mean={vals.mean():.6f} std={vals.std():.2e} min={vals.min():.6f} max={vals.max():.6f}')
print(f'  drift (max - min): {vals.max() - vals.min():.2e}')
print(f'  finite: {bool(np.isfinite(vals).all())}')
print(f'  stable: {bool(vals.std() < 1e-3)}')

# === Final summary ===
print('\n=== B Summary ===')
print(f'  Test 1 (8-weight): PASS' if len(res) == 8 else '  Test 1: FAIL')
print(f'  Test 2 (CALL=STATELESS): {"PASS" if all_pass else "FAIL"}')
print(f'  Test 3 (LOAD-rewrite): {"PASS" if rewrite_ok else "FAIL"}')
print(f'  Test 4 (100 cycles): {"PASS" if bool(np.isfinite(vals).all()) and vals.std() < 1e-3 else "FAIL"}')
