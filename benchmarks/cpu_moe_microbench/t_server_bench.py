
import os, struct, time, subprocess, numpy as np
os.chdir(r'E:\FreeToken\benchmarks\cpu_moe_microbench')
data = open('t_mtp_fc_with_act.bin', 'rb').read()
fm, fk, fnb, fns = struct.unpack('<IIII', data[:16])
off = 16
fcW = np.frombuffer(data[off:off+fm*fnb*4], dtype=np.uint32); off += fm*fnb*4
fcB = np.frombuffer(data[off:off+fm*fns*4], dtype=np.float32); off += fm*fns*4
fcS = np.frombuffer(data[off:off+fm*fns*4], dtype=np.float32); off += fm*fns*4
act = np.frombuffer(data[off:off+fk*4], dtype=np.float32); off += fk*4
M, K = fm, fk
sz_p = M * fnb * 4
sz_s = M * fns * 4
sz_b = M * fns * 4
hdr = struct.pack('<IIIII', M, K, sz_p, sz_s, sz_b)
act_int = act.view(np.int32)
base = hdr + fcW.tobytes() + fcS.tobytes() + fcB.tobytes() + act_int.tobytes()

# run server
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

N = 50
ts = []
for i in range(N):
    t0 = time.time()
    p.stdin.write(base)
    p.stdin.flush()
    rl = p.stdout.read(4)
    sz = struct.unpack('<I', rl)[0]
    outv = p.stdout.read(sz)
    t1 = time.time()
    ts.append((t1 - t0) * 1000)
ts = np.array(ts)
print(f'latency ms: min={ts[5:].min():.3f} median={np.median(ts[5:]):.3f} mean={ts[5:].mean():.3f} max={ts.max():.3f}')
print(f'first 5: {ts[:5]}')
print(f'last 5: {ts[-5:]}')
print(f'outv[0] first={struct.unpack("<f", outv[:4])[0]:.4f}')
p.terminate(); p.wait(timeout=5)
print('--- server stderr (last 20 lines) ---')
print('\n'.join(lines[-20:]))
