
import os, struct, time, subprocess, numpy as np
os.chdir(r'E:\FreeToken\benchmarks\cpu_moe_microbench')
real_data = open('t_mtp_fc_with_act.bin', 'rb').read()
fm, fk, fnb, fns = struct.unpack('<IIII', real_data[:16])
off = 16
fcW_real = np.frombuffer(real_data[off:off+fm*fnb*4], dtype=np.uint32); off += fm*fnb*4
act_real = np.frombuffer(real_data[off:off+fk*4], dtype=np.float32); off += fk*4

# Make M independent rows of K=4096 by reading M*nbPerRow fresh from raw file
# t_mtp_fc_with_act.bin has M=1 only. For M=8 we need to load 8 rows from safetensors
import safetensors.torch
state = safetensors.torch.load_file(r'E:\models\Qwen3.6-35B-A3B-MXFP4-MTP\model-00022-of-00023.safetensors')
fcW_safetensors = state['mtp.fc.weight']  # [2048, 512] uint32

def make_req(M, K):
    nb = K // 8
    ns = K // 32
    # Take M distinct rows from safetensors (or fall back to row 0)
    rows = []
    for r in range(M):
        if r < fcW_safetensors.shape[0]:
            rows.append(fcW_safetensors[r, :nb].numpy().tobytes())
        else:
            rows.append(fcW_safetensors[0, :nb].numpy().tobytes())
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

# M=8 first directly
r = make_req(8, 4096)
print(f'req size {len(r)}')
p.stdin.write(r); p.stdin.flush()
time.sleep(1.5)
rl = p.stdout.read(4)
if len(rl) == 4:
    sz = struct.unpack('<I', rl)[0]
    outv = p.stdout.read(sz)
    print(f'M=8: sz={sz}, outv={np.frombuffer(outv, dtype=np.float32)[:8] if len(outv)>=32 else "n/a"}')
else:
    print(f'M=8: got {len(rl)} bytes')

p.terminate(); p.wait(timeout=5)
print('--- stderr ---')
print(''.join(lines))
