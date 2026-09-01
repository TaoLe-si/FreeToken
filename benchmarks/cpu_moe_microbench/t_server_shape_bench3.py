
import os, struct, time, subprocess, numpy as np
os.chdir(r'E:\FreeToken\benchmarks\cpu_moe_microbench')
import safetensors.torch
data = safetensors.torch.load_file(r'E:\models\Qwen3.6-35B-A3B-MXFP4-MTP\model-00022-of-00023.safetensors')
fcW = data['mtp.fc.weight']  # [2048, 512] uint32

def make_req(M, K, act_arr=None, seed=42):
    nb = K // 8
    ns = K // 32
    assert K % 32 == 0
    rows = []
    for r in range(M):
        ri = r % fcW.shape[0]
        rows.append(fcW[ri, :nb].numpy().tobytes())
    packed = b''.join(rows)
    sz_p = len(packed)
    scales = bytes([0] * (M * ns * 4))
    biases = bytes([0] * (M * ns * 4))
    if act_arr is None:
        np.random.seed(seed)
        act_arr = (np.random.randn(K) * 0.1).astype(np.float32)
    act_int = act_arr.view(np.int32).tobytes()
    hdr = struct.pack('<IIIII', M, K, sz_p, M * ns * 4, M * ns * 4)
    return hdr + packed + scales + biases + act_int

# Warmup request
warmup = make_req(1, 4096)
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
p.stdin.write(warmup); p.stdin.flush()
rl = p.stdout.read(4); _ = p.stdout.read(struct.unpack('<I', rl)[0])  # warmup
time.sleep(0.5)

shapes = [
    ('fc M=1 K=4096', 1, 4096),
    ('attn q M=1 K=4096', 1, 4096),
    ('attn k/v M=1 K=512', 1, 512),
    ('attn o M=1 K=2048', 1, 2048),
    ('MoE gate/up M=1 K=2048', 1, 2048),
    ('MoE down M=1 K=512', 1, 512),
    ('8 experts batch M=8 K=2048', 8, 2048),
    ('8 experts batch M=8 K=512', 8, 512),
]

# Steady state: 5 iters per shape, take last
N = 5
print(f'{"shape":<35} {"size_KB":<10} {"lat_p50_ms":<12} {"outv[0]":<12}')
for name, M, K in shapes:
    lats = []
    last_outv = None
    for i in range(N):
        req = make_req(M, K, seed=i)
        t0 = time.time()
        p.stdin.write(req); p.stdin.flush()
        rl = p.stdout.read(4)
        sz = struct.unpack('<I', rl)[0]
        outv = p.stdout.read(sz)
        t1 = time.time()
        lats.append((t1 - t0) * 1000)
        if sz >= 4:
            last_outv = np.frombuffer(outv, dtype=np.float32)[0]
    size_kb = len(req) / 1024
    print(f'{name:<35} {size_kb:<10.1f} {np.median(lats):<12.3f} {last_outv:<12.4f}')

p.terminate(); p.wait(timeout=5)
print()
print('=== last 8 server-reported dispatch times ===')
dispatches = [l for l in lines if 'Dispatch' in l]
for l in dispatches[-8:]: print(l.strip())
