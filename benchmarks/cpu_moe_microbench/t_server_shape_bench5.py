
import os, struct, time, subprocess, numpy as np
os.chdir(r'E:\FreeToken\benchmarks\cpu_moe_microbench')
import safetensors.torch
data = safetensors.torch.load_file(r'E:\models\Qwen3.6-35B-A3B-MXFP4-MTP\model-00022-of-00023.safetensors')
fcW = data['mtp.fc.weight']

def make_req(M, K, seed=42):
    nb = K // 8
    ns = K // 32
    assert K % 32 == 0
    rows = [fcW[r % fcW.shape[0], :nb].numpy().tobytes() for r in range(M)]
    packed = b''.join(rows)
    scales = bytes([0] * (M * ns * 4))
    biases = bytes([0] * (M * ns * 4))
    np.random.seed(seed)
    act = (np.random.randn(K) * 0.1).astype(np.float32)
    act_int = act.view(np.int32).tobytes()
    hdr = struct.pack('<IIIII', M, K, len(packed), M * ns * 4, M * ns * 4)
    return hdr + packed + scales + biases + act_int

def read_exact(fd, n):
    out = b''
    while len(out) < n:
        chunk = os.read(fd, n - len(out))
        if not chunk: return out
        out += chunk
    return out

# Use low-level os.pipe + CreateProcess via subprocess but with direct fd access
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

stdin_fd = p.stdin.fileno()
stdout_fd = p.stdout.fileno()
# Reopen stdin/stdout as binary to avoid text mode translation
import msvcrt
msvcrt.setmode(stdin_fd, os.O_BINARY)
msvcrt.setmode(stdout_fd, os.O_BINARY)

# Warmup with real act
real_data = open('t_mtp_fc_with_act.bin', 'rb').read()
fm, fk, fnb, fns = struct.unpack('<IIII', real_data[:16])
off = 16
fcW_real = np.frombuffer(real_data[off:off+fm*fnb*4], dtype=np.uint32); off += fm*fnb*4
fcB_real = np.frombuffer(real_data[off:off+fm*fns*4], dtype=np.float32); off += fm*fns*4
fcS_real = np.frombuffer(real_data[off:off+fm*fns*4], dtype=np.float32); off += fm*fns*4
act_real = np.frombuffer(real_data[off:off+fk*4], dtype=np.float32)
warmup = struct.pack('<IIIII', fm, fk, fm*fnb*4, fm*fns*4, fm*fns*4) + fcW_real.tobytes() + fcS_real.tobytes() + fcB_real.tobytes() + act_real.view(np.int32).tobytes()
os.write(stdin_fd, warmup)
rl = read_exact(stdout_fd, 4); outv = read_exact(stdout_fd, struct.unpack('<I', rl)[0])
print('warmup outv[0]:', np.frombuffer(outv, dtype=np.float32)[0])
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
N = 5
print()
print(f'{"shape":<35} {"size_KB":<10} {"lat_p50_ms":<12} {"outv[0]":<12}')
for name, M, K in shapes:
    lats = []
    last_outv = None
    for i in range(N):
        req = make_req(M, K, seed=i+100)
        t0 = time.time()
        os.write(stdin_fd, req)
        rl = read_exact(stdout_fd, 4)
        if len(rl) < 4:
            print(f'  short read for {name} iter {i}: got {len(rl)} bytes')
            continue
        sz = struct.unpack('<I', rl)[0]
        outv = read_exact(stdout_fd, sz)
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
