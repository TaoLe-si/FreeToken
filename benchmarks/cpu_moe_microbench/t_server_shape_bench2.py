
import os, struct, time, subprocess, numpy as np
os.chdir(r'E:\FreeToken\benchmarks\cpu_moe_microbench')
import safetensors.torch
data = safetensors.torch.load_file(r'E:\models\Qwen3.6-35B-A3B-MXFP4-MTP\model-00022-of-00023.safetensors')
fcW = data['mtp.fc.weight']  # [2048, 512] uint32 (K=4096, 512 uints/row)

def make_req(M, K, seed=42):
    nb = K // 8
    ns = K // 32
    if (K % 32) != 0:
        raise ValueError('K must be multiple of 32')
    rows = []
    for r in range(M):
        ri = r % fcW.shape[0]
        # take first nb uints from fcW row ri
        chunk = fcW[ri, :nb].numpy().tobytes()
        rows.append(chunk)
    packed = b''.join(rows)
    sz_p = len(packed)
    # scales/biases: M*ns*4 bytes (unused by shader, just need correct size)
    scales = bytes([0] * (M * ns * 4))
    biases = bytes([0] * (M * ns * 4))
    # act: K*4 int32 (float32 bit pattern)
    np.random.seed(seed)
    act = (np.random.randn(K) * 0.1).astype(np.float32).view(np.int32).tobytes()
    hdr = struct.pack('<IIIII', M, K, sz_p, M * ns * 4, M * ns * 4)
    return hdr + packed + scales + biases + act

shapes = [
    ('fc M=1 K=4096', 1, 4096),
    ('attn q M=1 K=4096', 1, 4096),
    ('attn k/v M=1 K=512', 1, 512),
    ('attn o M=1 K=2048', 1, 2048),
    ('MoE gate/up M=1 K=2048', 1, 2048),
    ('MoE down M=1 K=512', 1, 512),
    ('8 experts batch M=8 K=2048', 8, 2048),
    ('8 experts batch M=8 K=512', 8, 512),
    ('MoE top-8 concat M=1 K=2048', 1, 2048),
]

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

print(f'{"shape":<35} {"size_KB":<10} {"latency_ms":<12} {"outv[0]":<12}')
results = []
for name, M, K in shapes:
    req = make_req(M, K)
    size_kb = len(req) / 1024
    t0 = time.time()
    p.stdin.write(req); p.stdin.flush()
    rl = p.stdout.read(4)
    sz = struct.unpack('<I', rl)[0]
    outv = p.stdout.read(sz)
    t1 = time.time()
    lat = (t1 - t0) * 1000
    v0 = struct.unpack('<f', outv[:4])[0] if sz >= 4 else float('nan')
    print(f'{name:<35} {size_kb:<10.1f} {lat:<12.3f} {v0:<12.4f}')
    results.append((name, M, K, lat, v0))

p.terminate(); p.wait(timeout=5)
print()
print('=== server-reported GPU dispatch times (steady state) ===')
dispatches = [l for l in lines if 'Dispatch' in l]
for l in dispatches[-len(shapes):]: print(l.strip())
