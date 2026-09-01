
import os, struct, time, subprocess, numpy as np
os.chdir(r'E:\FreeToken\benchmarks\cpu_moe_microbench')

# Load real mtp attn q/k/v/o and MoE weights for shape-realistic tests
import safetensors.torch
data = safetensors.torch.load_file(r'E:\models\Qwen3.6-35B-A3B-MXFP4-MTP\model-00022-of-00023.safetensors')
# Use fcW as synthetic packed for various sizes (fcW is row-major uint32 = 1 row of K=4096)
fcW = data['mtp.fc.weight']  # [2048, 512] uint32 (K=4096, out=2048)

def make_req(M, K, gbl=1.0, rowB=0.0):
    """Make a server request. fcW layout: each row has K/8 uint32 packed."""
    nb = K // 8
    ns = K // 32
    # We need M rows: take first M rows of fcW (or repeat row 0 for simplicity)
    rows = []
    for r in range(M):
        if r < fcW.shape[0]:
            rows.append(fcW[r].numpy().tobytes())
        else:
            rows.append(fcW[r % fcW.shape[0]].numpy().tobytes())
    packed = b''.join(rows)
    sz_p = len(packed)
    # scales: M*ns*4 bytes (float per block, fc_clean protocol)
    sz_s = M * ns * 4
    scales = b'\x00' * sz_s
    sz_b = M * ns * 4
    biases = b'\x00' * sz_b
    # act: K*4 bytes (int32 bit pattern = float32)
    np.random.seed(42)
    act = (np.random.randn(K) * 0.1).astype(np.float32).view(np.int32).tobytes()
    hdr = struct.pack('<IIIII', M, K, sz_p, sz_s, sz_b)
    return hdr + packed + scales + biases + act

# Test shapes covering MTP head use cases
shapes = [
    ('fc M=1 K=4096', 1, 4096),
    ('attn q M=1 K=4096', 1, 4096),
    ('attn k/v M=1 K=512', 1, 512),
    ('attn o M=1 K=2048', 1, 2048),
    ('MoE gate M=1 K=2048', 1, 2048),
    ('MoE up M=1 K=2048', 1, 2048),
    ('MoE down M=1 K=512', 1, 512),
    ('MoE 8 experts batch M=8 K=2048', 8, 2048),
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

print(f'{"shape":<35} {"size_KB":<8} {"latency_ms":<12}')
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
    print(f'{name:<35} {size_kb:<8.1f} {lat:<12.3f}')

p.terminate(); p.wait(timeout=5)
# Last few server-stated times
print('--- last 8 server-reported dispatch times ---')
dispatches = [l for l in lines if 'Dispatch' in l]
for l in dispatches[-8:]: print(l.strip())
