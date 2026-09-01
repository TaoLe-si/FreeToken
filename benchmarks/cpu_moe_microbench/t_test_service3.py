
import subprocess, struct, numpy as np, time

K_E2M1X2 = np.array([0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12], dtype=np.int32)

def ref(packed, scl, act, asb, gbl, M, K):
    NB = K // 16
    out = np.zeros(M, dtype=np.float32)
    for r in range(M):
        acc = 0.0
        for b in range(NB):
            pk = packed[r*NB*8 + b*8 : r*NB*8 + b*8 + 8]
            wsum = 0
            for j in range(8):
                byte = int(pk[j])
                wsum += int(K_E2M1X2[byte & 0xF]) * int(act[b*16 + j]) + int(K_E2M1X2[byte >> 4]) * int(act[b*16 + 8 + j])
            acc += float(wsum) * 0.01 * float(scl[r*NB + b] & 0xFF) + float(asb[b])
        out[r] = acc * 0.25 * float(gbl[r])
    return out

def payload_for(M, K, packed, scl, act, asb, gbl):
    return struct.pack("<II", M, K) + packed.tobytes() + scl.tobytes() + act.tobytes() + asb.tobytes() + gbl.tobytes()

def call(p, payload, M):
    p.stdin.write(payload); p.stdin.flush()
    raw = p.stdout.read(M * 4)
    return np.frombuffer(raw, dtype=np.float32)

M, K = 2048, 4096
NB = K // 16
rng = np.random.default_rng(42)
packed = rng.integers(0, 256, M * NB * 8, dtype=np.uint8)
scl = rng.integers(0, 128, M * NB, dtype=np.uint32)
act = rng.integers(-127, 128, NB * 16, dtype=np.int32)
asb = (0.01 + 0.05 * rng.integers(0, 100, NB) / 100.0).astype(np.float32)
gbl = (0.5 + 0.5 * rng.integers(0, 100, M) / 100.0).astype(np.float32)
exp = ref(packed, scl, act, asb, gbl, M, K)

p = subprocess.Popen([r"t_d3d12_service.exe"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
call(p, payload_for(M, K, packed, scl, act, asb, gbl), M)
t0 = time.perf_counter()
for i in range(5):
    out = call(p, payload_for(M, K, packed, scl, act, asb, gbl), M)
dt = (time.perf_counter() - t0) / 5
bad = int(np.sum(np.abs(out - exp) > 1e-3 * (np.abs(exp) + 1)))
wbytes = M * NB * 8 + M * NB * 4 + NB * 16 * 4 + NB * 4 + M * 4
print(f"in-proc call: {dt*1000:.2f} ms -> {wbytes/dt/1e9:.1f} GB/s, bad={bad}/{M}")
print(f"out[0..3]={out[:4]} exp={exp[:4]}")
p.stdin.close(); p.wait()
print("SERVICE:", "OK" if bad == 0 else "FAIL")
