
import os, struct, time, subprocess, numpy as np
os.chdir(r'E:\FreeToken\benchmarks\cpu_moe_microbench')
real_data = open('t_mtp_fc_with_act.bin', 'rb').read()
fm, fk, fnb, fns = struct.unpack('<IIII', real_data[:16])
off = 16
fcW_real = np.frombuffer(real_data[off:off+fm*fnb*4], dtype=np.uint32); off += fm*fnb*4
act_real = np.frombuffer(real_data[off:off+fk*4], dtype=np.float32); off += fk*4

def make_req(M, K):
    nb = K // 8
    ns = K // 32
    if M == 1:
        packed = fcW_real[:nb].tobytes()
    else:
        # Repeat row 0 for M rows (so M*nb uints total)
        rows = [fcW_real[:nb].tobytes()] * M
        packed = b''.join(rows)
    scales = bytes([0] * (M * ns * 4))
    biases = bytes([0] * (M * ns * 4))
    act = act_real[:K].view(np.int32).tobytes()
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

# Warmup
warmup = make_req(1, 4096)
p.stdin.write(warmup); p.stdin.flush()
time.sleep(0.5)
rl = p.stdout.read(4); _ = p.stdout.read(struct.unpack('<I', rl)[0])

shapes = [
    ('fc M=1 K=4096', 1, 4096),
    ('fc M=8 K=4096', 8, 4096),
    ('attn q M=1 K=4096', 1, 4096),
    ('attn k/v M=1 K=512', 1, 512),
    ('attn o M=1 K=2048', 1, 2048),
    ('MoE gate M=1 K=2048', 1, 2048),
    ('MoE down M=1 K=512', 1, 512),
    ('8 experts M=8 K=2048', 8, 2048),
    ('8 experts M=8 K=512', 8, 512),
]
N = 10
print(f'{"shape":<30} {"size_KB":<10} {"lat_p50_ms":<12} {"GPU_disp_ms":<12} {"outv[0]":<12}')
for name, M, K in shapes:
    lats = []
    last_v = None
    for i in range(N):
        r = make_req(M, K)
        t0 = time.time()
        p.stdin.write(r); p.stdin.flush()
        rl = p.stdout.read(4)
        sz = struct.unpack('<I', rl)[0]
        outv = p.stdout.read(sz)
        t1 = time.time()
        lats.append((t1-t0)*1000)
        if len(outv) >= 4:
            last_v = struct.unpack('<f', outv[:4])[0]
    dispatches = [l for l in lines if 'Dispatch' in l]
    try:
        s = dispatches[-1].split(': ')[1].replace('ms', '').strip()
        last_disp = float(s)
    except:
        last_disp = 0.0
    print(f'{name:<30} {len(r)/1024:<10.1f} {np.median(lats):<12.3f} {last_disp:<12.3f} {last_v:<12.4f}')

p.terminate(); p.wait(timeout=5)
