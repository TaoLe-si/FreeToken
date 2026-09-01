
import os, struct, time, subprocess, numpy as np
os.chdir(r'E:\FreeToken\benchmarks\cpu_moe_microbench')
real_data = open('t_mtp_fc_with_act.bin', 'rb').read()
fm, fk, fnb, fns = struct.unpack('<IIII', real_data[:16])
off = 16
fcW_real = np.frombuffer(real_data[off:off+fm*fnb*4], dtype=np.uint32); off += fm*fnb*4
act_real = np.frombuffer(real_data[off:off+fk*4], dtype=np.float32); off += fk*4

def make_req(M, K, packed_data=None, act_data=None):
    nb = K // 8
    ns = K // 32
    if packed_data is None:
        packed = fcW_real[:nb].tobytes()  # always 1 row of nb uints
    else:
        packed = packed_data
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

# Sequence: M=1 K=4096, M=1 K=512, M=1 K=4096 again
for K in [4096, 512, 4096, 512, 4096]:
    r = make_req(1, K)
    p.stdin.write(r); p.stdin.flush()
    time.sleep(0.5)
    rl = p.stdout.read(4)
    sz = struct.unpack('<I', rl)[0] if len(rl)==4 else 0
    outv = p.stdout.read(sz) if sz > 0 else b''
    v0 = struct.unpack('<f', outv[:4])[0] if len(outv) >= 4 else 'n/a'
    print(f'M=1 K={K}: outv[0]={v0}')

p.terminate(); p.wait(timeout=5)
print('--- stderr ---')
print(''.join(lines))
