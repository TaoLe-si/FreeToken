
import os, struct, time, subprocess, numpy as np
os.chdir(r'E:\FreeToken\benchmarks\cpu_moe_microbench')
import safetensors.torch

# Use REAL t_mtp_fc_with_act.bin data (the same fc_clean verified)
data = open('t_mtp_fc_with_act.bin', 'rb').read()
fm, fk, fnb, fns = struct.unpack('<IIII', data[:16])
off = 16
fcW = np.frombuffer(data[off:off+fm*fnb*4], dtype=np.uint32); off += fm*fnb*4
fcB = np.frombuffer(data[off:off+fm*fns*4], dtype=np.float32); off += fm*fns*4
fcS = np.frombuffer(data[off:off+fm*fns*4], dtype=np.float32); off += fm*fns*4
act = np.frombuffer(data[off:off+fk*4], dtype=np.float32); off += fk*4
print('fcW[0] first uint =', hex(fcW[0]), 'fcS[0] =', fcS[0], 'fcB[0] =', fcB[0], 'act[0] =', act[0])
# Build request identical to validate6
M, K = fm, fk
sz_p = M * fnb * 4
sz_s = M * fns * 4
sz_b = M * fns * 4
hdr = struct.pack('<IIIII', M, K, sz_p, sz_s, sz_b)
act_int = act.view(np.int32)
req = hdr + fcW.tobytes() + fcS.tobytes() + fcB.tobytes() + act_int.tobytes()
print('req size', len(req))

p = subprocess.Popen(['t_mxfp4_gemv_server.exe'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
import threading
lines = []
def drain():
    while True:
        l = p.stderr.readline()
        if not l: break
        lines.append(l.decode(errors='replace'))
threading.Thread(target=drain, daemon=True).start()
time.sleep(2.0)
t0 = time.time()
p.stdin.write(req); p.stdin.flush()
rl = p.stdout.read(4)
sz = struct.unpack('<I', rl)[0]
outv = p.stdout.read(sz)
t1 = time.time()
print(f'latency: {(t1-t0)*1000:.3f}ms, outv[:4]: {np.frombuffer(outv, dtype=np.float32)[:4]}')

# Second request with the same data (should be fast now)
t0 = time.time()
p.stdin.write(req); p.stdin.flush()
rl = p.stdout.read(4)
sz = struct.unpack('<I', rl)[0]
outv = p.stdout.read(sz)
t1 = time.time()
print(f'2nd latency: {(t1-t0)*1000:.3f}ms, outv[:4]: {np.frombuffer(outv, dtype=np.float32)[:4]}')

# Now try with random act
np.random.seed(42)
act_r = (np.random.randn(K) * 0.1).astype(np.float32)
act_r_int = act_r.view(np.int32)
req2 = hdr + fcW.tobytes() + fcS.tobytes() + fcB.tobytes() + act_r_int.tobytes()
t0 = time.time()
p.stdin.write(req2); p.stdin.flush()
rl = p.stdout.read(4)
sz = struct.unpack('<I', rl)[0]
outv = p.stdout.read(sz)
t1 = time.time()
print(f'random-act latency: {(t1-t0)*1000:.3f}ms, outv[:4]: {np.frombuffer(outv, dtype=np.float32)[:4]}')
print('CPU expect (approx):', float((act_r).sum()) * 0.0 + 0.0)  # gbl=1, rowB=0

p.terminate(); p.wait(timeout=5)
print('--- server stderr ---')
print(''.join(lines))
