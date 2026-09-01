
import os, struct, time, subprocess
import numpy as np
import safetensors.torch
os.chdir(r'E:\FreeToken\benchmarks\cpu_moe_microbench')
real_data = open('t_mtp_fc_with_act.bin', 'rb').read()
fm, fk, fnb, fns = struct.unpack('<IIII', real_data[:16])
off = 16
fcW_real = np.frombuffer(real_data[off:off+fm*fnb*4], dtype=np.uint32); off += fm*fnb*4
fcB_real = np.frombuffer(real_data[off:off+fm*fns*4], dtype=np.float32); off += fm*fns*4
fcS_real = np.frombuffer(real_data[off:off+fm*fns*4], dtype=np.float32); off += fm*fns*4
act_real = np.frombuffer(real_data[off:off+fk*4], dtype=np.float32); off += fk*4

def req(packed, act_bytes, scales, biases):
    return struct.pack('<IIIIII', 1, fk, len(packed), len(act_bytes), len(scales), len(biases)) + packed + act_bytes + scales + biases

# All 0xff scales, all 0 biases
v1 = req(fcW_real[:512].tobytes(), act_real.view(np.int32).tobytes(), bytes([0xff]*512), bytes([0]*512))
# All 0 scales
v2 = req(fcW_real[:512].tobytes(), act_real.view(np.int32).tobytes(), bytes([0]*512), bytes([0]*512))
# Real S, 0 B
v3 = req(fcW_real[:512].tobytes(), act_real.view(np.int32).tobytes(), fcS_real.tobytes(), bytes([0]*512))
# 0 S, real B
v4 = req(fcW_real[:512].tobytes(), act_real.view(np.int32).tobytes(), bytes([0]*512), fcB_real.tobytes())
# Real S, real B (diag14 A)
v5 = req(fcW_real[:512].tobytes(), act_real.view(np.int32).tobytes(), fcS_real.tobytes(), fcB_real.tobytes())
# 128 S (only 128 bytes)
v6 = req(fcW_real[:512].tobytes(), act_real.view(np.int32).tobytes(), bytes([0]*128), bytes([0]*128))

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
for name, r in [('v1: 0xff S, 0 B', v1), ('v2: 0 S, 0 B', v2), ('v3: real S, 0 B', v3), ('v4: 0 S, real B', v4), ('v5: real S, real B (A)', v5), ('v6: 128B S, 128B B', v6)]:
    p.stdin.write(r); p.stdin.flush()
    time.sleep(0.5)
    rl = p.stdout.read(4)
    sz = struct.unpack('<I', rl)[0] if len(rl)==4 else 0
    outv = p.stdout.read(sz)
    v0 = struct.unpack('<f', outv[:4])[0] if len(outv)>=4 else 'n/a'
    print(f'{name:<30} outv[0]={v0}')
p.terminate(); p.wait(timeout=5)
print('stderr:', ''.join(lines))
