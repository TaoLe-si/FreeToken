import sys, os, time, struct, subprocess
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
act = np.full(K, 0.05, dtype='float32')

# === v1 STATELESS (each call = new process) ===
print('=== v1 STATELESS (new process per call) ===')
hdr = struct.pack('<IIIIII', M, K, szP, szA, szS, szB)
payload_v1 = hdr + packed + act.tobytes() + bytes(szS) + bytes(szB)
N = 20
t0 = time.time()
for _ in range(N):
    p = subprocess.Popen([r'E:\FreeToken\benchmarks\cpu_moe_microbench\t_mxfp4_gemv_server.exe'],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         cwd=r'E:\FreeToken\benchmarks\cpu_moe_microbench')
    out, _ = p.communicate(payload_v1, timeout=10)
    val = struct.unpack('<f', out[4:8])[0]
    p.wait()
t1 = time.time()
print(f'  {N} calls in {(t1-t0)*1000:.1f}ms = {(t1-t0)/N*1000:.1f}ms/call (incl. process startup)')

# === v2 LOAD + N CALLs (single process) ===
print('\n=== v2 LOAD + N CALLs (single process) ===')
N2 = 100
payload = b'LOAD fc ' + str(M).encode() + b' ' + str(K).encode() + b' ' + str(len(packed)).encode() + b'\n' + packed
for _ in range(N2):
    payload += b'CALL fc ' + str(szA).encode() + b' ' + str(szS).encode() + b' ' + str(szB).encode() + b'\n'
    payload += act.tobytes() + bytes(szS) + bytes(szB)
payload += b'QUIT\n'

t0 = time.time()
p = subprocess.Popen([r'E:\FreeToken\benchmarks\cpu_moe_microbench\t_mxfp4_gemv_v2_server.exe'],
                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                     cwd=r'E:\FreeToken\benchmarks\cpu_moe_microbench')
out, _ = p.communicate(payload, timeout=60)
t1 = time.time()
p.wait()

# Parse responses
pos = out.find(b'\n') + 1
results = []
for _ in range(N2):
    val = struct.unpack('<f', out[pos+4:pos+8])[0]
    results.append(val)
    pos += 8

print(f'  {N2} CALLs in {(t1-t0)*1000:.1f}ms = {(t1-t0)/N2*1000:.3f}ms/call (sticky weight)')
print(f'  First 5: {results[:5]}')
print(f'  Last 5:  {results[-5:]}')

# Subtract the LOAD time (first call was ~0.667ms, subsequent ~0.2ms)
# Just look at the variability
arr = np.array(results)
print(f'  mean={arr.mean():.6f} std={arr.std():.2e} min={arr.min():.6f} max={arr.max():.6f}')
