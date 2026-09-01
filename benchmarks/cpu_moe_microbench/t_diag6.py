
import os, struct, time, subprocess
import numpy as np
import safetensors.torch
os.chdir(r'E:\FreeToken\benchmarks\cpu_moe_microbench')
data = safetensors.torch.load_file(r'E:\models\Qwen3.6-35B-A3B-MXFP4-MTP\model-00022-of-00023.safetensors')
fcW = data['mtp.fc.weight']

def make_req(M, K, seed=42):
    nb = K // 8
    ns = K // 32
    rows = [fcW[r % fcW.shape[0], :nb].numpy().tobytes() for r in range(M)]
    packed = b''.join(rows)
    scales = bytes([0] * (M * ns * 4))
    biases = bytes([0] * (M * ns * 4))
    np.random.seed(seed)
    act = (np.random.randn(K) * 0.1).astype(np.float32)
    act_int = act.view(np.int32).tobytes()
    hdr = struct.pack('<IIIII', M, K, len(packed), M * ns * 4, M * ns * 4)
    return hdr + packed + scales + biases + act_int

# Use stream.read like t_diag.py
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
for i in range(3):
    req = make_req(1, 4096, seed=i+1)
    t0 = time.time()
    p.stdin.write(req)
    p.stdin.flush()
    time.sleep(0.3)
    rl = p.stdout.read(4)
    sz = struct.unpack('<I', rl)[0] if len(rl)==4 else 0
    outv = p.stdout.read(sz)
    t1 = time.time()
    print(f'iter {i+1}: outv[0]={struct.unpack("<f", outv[:4])[0] if len(outv)>=4 else "n/a"}, lat={(t1-t0)*1000:.1f}ms')
p.terminate(); p.wait(timeout=5)
print('--- stderr ---')
print(''.join(lines))
