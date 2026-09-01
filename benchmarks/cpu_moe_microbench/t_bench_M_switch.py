
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
        rows = [fcW_real[:nb].tobytes()] * M
        packed = b''.join(rows)
    scales = bytes([0] * (M * ns * 4))
    biases = bytes([0] * (M * ns * 4))
    act = act_real[:K].view(np.int32).tobytes()
    return struct.pack('<IIIIII', M, K, len(packed), len(act), len(scales), len(biases)) + packed + act + scales + biases

# Start fresh server for M=8 only
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

# M=8 K=4096 first
r1 = make_req(8, 4096)
p.stdin.write(r1); p.stdin.flush()
time.sleep(1.0)
rl = p.stdout.read(4)
sz = struct.unpack('<I', rl)[0] if len(rl)==4 else 0
outv = p.stdout.read(sz) if sz > 0 else b''
print(f'M=8 K=4096: sz={sz}, outv={np.frombuffer(outv, dtype=np.float32)[:8] if len(outv)>=32 else "n/a"}')

p.terminate(); p.wait(timeout=5)
print('--- stderr ---')
print(''.join(lines))
