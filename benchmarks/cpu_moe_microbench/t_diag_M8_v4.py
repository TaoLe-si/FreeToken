
import os, struct, time, subprocess, numpy as np
import safetensors.torch
os.chdir(r'E:\FreeToken\benchmarks\cpu_moe_microbench')
state = safetensors.torch.load_file(r'E:\models\Qwen3.6-35B-A3B-MXFP4-MTP\model-00022-of-00023.safetensors')
fcW = state['mtp.fc.weight']
real_data = open('t_mtp_fc_with_act.bin', 'rb').read()
fm, fk, fnb, fns = struct.unpack('<IIII', real_data[:16])
off = 16
fcW_real = np.frombuffer(real_data[off:off+fm*fnb*4], dtype=np.uint32); off += fm*fnb*4
act_real = np.frombuffer(real_data[off:off+fk*4], dtype=np.float32); off += fk*4

def make_req(M, K, act_arr=None):
    nb = K // 8
    ns = K // 32
    if M == 1:
        packed = fcW_real[:nb].tobytes()
    else:
        rows = [fcW[r % fcW.shape[0], :nb].numpy().tobytes() for r in range(M)]
        packed = b''.join(rows)
    scales = bytes([0] * (M * ns * 4))
    biases = bytes([0] * (M * ns * 4))
    if act_arr is None:
        act_arr = act_real
    act = act_arr[:K].view(np.int32).tobytes() if act_arr.size >= K else act_arr.view(np.int32).tobytes()
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

# Test 1: M=1
r = make_req(1, 4096)
p.stdin.write(r); p.stdin.flush()
time.sleep(0.5)
rl = p.stdout.read(4); sz = struct.unpack('<I', rl)[0]
outv = p.stdout.read(sz)
print(f'M=1: outv[0]={struct.unpack("<f", outv[:4])[0]:.4f}')

# Test 2: M=8 with real act
r = make_req(8, 4096)
p.stdin.write(r); p.stdin.flush()
time.sleep(1.0)
rl = p.stdout.read(4)
if len(rl) == 4:
    sz = struct.unpack('<I', rl)[0]
    outv = p.stdout.read(sz)
    if len(outv) >= 32:
        vs = np.frombuffer(outv, dtype=np.float32)
        print(f'M=8: outv={vs[:8]}')
    else:
        print(f'M=8: got {len(outv)} bytes')
else:
    print(f'M=8: got {len(rl)} bytes')

p.terminate(); p.wait(timeout=5)
print('--- stderr ---')
print(''.join(lines))
