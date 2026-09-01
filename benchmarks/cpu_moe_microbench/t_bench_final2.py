
import os, struct, time, subprocess, numpy as np
os.chdir(r'E:\FreeToken\benchmarks\cpu_moe_microbench')
import safetensors.torch
real_data = open('t_mtp_fc_with_act.bin', 'rb').read()
fm, fk, fnb, fns = struct.unpack('<IIII', real_data[:16])
off = 16
fcW_real = np.frombuffer(real_data[off:off+fm*fnb*4], dtype=np.uint32); off += fm*fnb*4
fcB_real = np.frombuffer(real_data[off:off+fm*fns*4], dtype=np.float32); off += fm*fns*4
fcS_real = np.frombuffer(real_data[off:off+fm*fns*4], dtype=np.float32); off += fm*fns*4
act_real = np.frombuffer(real_data[off:off+fk*4], dtype=np.float32); off += fk*4

def make_req(M, K, act_data=None):
    nb = K // 8
    ns = K // 32
    if M == 1:
        packed = fcW_real[:nb].tobytes()
    else:
        packed = fcW_real[:M*nb].tobytes()  # extra rows = same as row 0 (will need M rows of data — for now M=8 still uses fcW_real first M rows)
    scales = bytes([0] * (M * ns * 4))
    biases = bytes([0] * (M * ns * 4))
    if act_data is None:
        act_data = act_real[:K]
    act = act_data.view(np.int32).tobytes()
    return struct.pack('<IIIIII', M, K, len(packed), len(act), len(scales), len(biases)) + packed + act + scales + biases

p = subprocess.Popen(['t_mxfp4_gemv_server.exe'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
import threading
lines = []
def drain():
    while True:
        l = p.stderr.readline()
        if not l: break
        lines.append(l.decode(errors='replace'))
threading.Thread(target=drain, daemon=True).start()
time.sleep(2.0)

# Warmup M=1 K=4096
warmup = make_req(1, 4096)
p.stdin.write(warmup); p.stdin.flush()
time.sleep(0.5)
rl = p.stdout.read(4)
sz = struct.unpack('<I', rl)[0]
outv = p.stdout.read(sz)
print(f'warmup: outv[0]={struct.unpack("<f", outv[:4])[0] if len(outv)>=4 else "n/a"}')

shapes = [
    ('fc M=1 K=4096', 1, 4096),
    ('attn q M=1 K=4096', 1, 4096),
    ('attn k/v M=1 K=512', 1, 512),
    ('attn o M=1 K=2048', 1, 2048),
    ('MoE gate M=1 K=2048', 1, 2048),
    ('MoE down M=1 K=512', 1, 512),
]
N = 10
print()
print(f'{"shape":<30} {"size_KB":<10} {"lat_p50_ms":<12} {"outv[0]":<12}')
for name, M, K in shapes:
    lats = []
    last_v = None
    for i in range(N):
        r = make_req(M, K)
        t0 = time.time()
        p.stdin.write(r); p.stdin.flush()
        rl = p.stdout.read(4)
        sz = struct.unpack('<I', rl)[0] if len(rl) == 4 else 0
        outv = p.stdout.read(sz)
        t1 = time.time()
        lats.append((t1-t0)*1000)
        if len(outv) >= 4:
            last_v = struct.unpack('<f', outv[:4])[0]
    print(f'{name:<30} {len(r)/1024:<10.1f} {np.median(lats):<12.3f} {last_v:<12.4f}')

p.terminate(); p.wait(timeout=5)
print()
print('=== last 6 server-reported dispatch times ===')
dispatches = [l for l in lines if 'Dispatch' in l]
for l in dispatches[-6:]: print(l.strip())
