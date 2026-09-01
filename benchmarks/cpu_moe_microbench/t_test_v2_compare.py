import sys, os, struct, subprocess
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
szP, szA, szS, szB = len(packed), K*4, M*ns*4, M*ns*4

def run_v1_stateless(act_val):
    act = np.full(K, act_val, dtype='float32')
    hdr = struct.pack('<IIIIII', M, K, szP, szA, szS, szB)
    payload = hdr + packed + act.tobytes() + bytes(szS) + bytes(szB)
    p = subprocess.Popen([r'E:\FreeToken\benchmarks\cpu_moe_microbench\t_mxfp4_gemv_server.exe'],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         cwd=r'E:\FreeToken\benchmarks\cpu_moe_microbench')
    out, _ = p.communicate(payload, timeout=10)
    val = struct.unpack('<f', out[4:8])[0]
    p.wait()
    return val

def run_v2_stateless(act_val):
    act = np.full(K, act_val, dtype='float32')
    payload = b'STATELESS ' + str(M).encode() + b' ' + str(K).encode() + b' ' + str(szP).encode() + b' ' + str(szA).encode() + b' ' + str(szS).encode() + b' ' + str(szB).encode() + b'\n'
    payload += packed + act.tobytes() + bytes(szS) + bytes(szB) + b'QUIT\n'
    p = subprocess.Popen([r'E:\FreeToken\benchmarks\cpu_moe_microbench\t_mxfp4_gemv_v2_server.exe'],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         cwd=r'E:\FreeToken\benchmarks\cpu_moe_microbench')
    out, _ = p.communicate(payload, timeout=10)
    # Skip 'OK' for STATELESS, just the binary
    val = struct.unpack('<f', out[4:8])[0]
    p.wait()
    return val

def run_v2_load_call(act_vals):
    # Build: LOAD + N CALLs + QUIT
    payload = b'LOAD fc ' + str(M).encode() + b' ' + str(K).encode() + b' ' + str(len(packed)).encode() + b'\n' + packed
    for a in act_vals:
        act = np.full(K, a, dtype='float32')
        payload += b'CALL fc ' + str(szA).encode() + b' ' + str(szS).encode() + b' ' + str(szB).encode() + b'\n'
        payload += act.tobytes() + bytes(szS) + bytes(szB)
    payload += b'QUIT\n'
    p = subprocess.Popen([r'E:\FreeToken\benchmarks\cpu_moe_microbench\t_mxfp4_gemv_v2_server.exe'],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         cwd=r'E:\FreeToken\benchmarks\cpu_moe_microbench')
    out, _ = p.communicate(payload, timeout=30)
    p.wait()
    # Parse: LOAD reply 'OK fc 1 4096\n' (12 bytes), then N responses
    pos = out.find(b'\n') + 1
    results = []
    for _ in act_vals:
        if pos + 8 > len(out): break
        val = struct.unpack('<f', out[pos+4:pos+8])[0]
        results.append(val)
        pos += 8
    return results

# Test data
test_vals = [0.01, 0.05, 0.10, 0.20, -0.05, 0.0, 1.0]
print('=== Compare v1 vs v2 STATELESS for individual act values ===')
print(f'{"act":>8} | {"v1":>12} | {"v2 stat":>12} | {"diff":>10}')
for a in test_vals:
    v1 = run_v1_stateless(a)
    v2s = run_v2_stateless(a)
    print(f'{a:>8.3f} | {v1:>12.6f} | {v2s:>12.6f} | {v1-v2s:>10.2e}')

print('\n=== LOAD + multiple CALLs (only one server start) ===')
results = run_v2_load_call(test_vals)
for i, a in enumerate(test_vals):
    print(f'  CALL {a:>6.3f}: {results[i]:>12.6f}')

print('\n=== Verify CALLs match STATELESS (each value independently) ===')
for i, a in enumerate(test_vals):
    v2s = run_v2_stateless(a)
    diff = results[i] - v2s
    match = 'OK' if abs(diff) < 1e-4 else 'MISMATCH'
    print(f'  act={a:>6.3f}: LOAD+CALL={results[i]:.6f} vs STATELESS={v2s:.6f}  diff={diff:.2e}  {match}')
