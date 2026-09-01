
import os, struct, time, subprocess
import numpy as np
import safetensors.torch
os.chdir(r'E:\FreeToken\benchmarks\cpu_moe_microbench')
data = safetensors.torch.load_file(r'E:\models\Qwen3.6-35B-A3B-MXFP4-MTP\model-00022-of-00023.safetensors')
fcW = data['mtp.fc.weight']
real_data = open('t_mtp_fc_with_act.bin', 'rb').read()
fm, fk, fnb, fns = struct.unpack('<IIII', real_data[:16])
off = 16
fcW_real = np.frombuffer(real_data[off:off+fm*fnb*4], dtype=np.uint32); off += fm*fnb*4
fcB_real = np.frombuffer(real_data[off:off+fm*fns*4], dtype=np.float32); off += fm*fns*4
fcS_real = np.frombuffer(real_data[off:off+fm*fns*4], dtype=np.float32); off += fm*fns*4
act_real = np.frombuffer(real_data[off:off+fk*4], dtype=np.float32); off += fk*4

def req(packed, act_bytes, scales, biases):
    return struct.pack('<IIIIII', 1, fk, len(packed), len(act_bytes), len(scales), len(biases)) + packed + act_bytes + scales + biases

# Test variants
variants = [
    ('A: real A', fcW_real.tobytes(), act_real.view(np.int32).tobytes(), fcS_real.tobytes(), fcB_real.tobytes()),
    ('ones A', fcW_real.tobytes(), (np.ones(fk, dtype=np.float32)).view(np.int32).tobytes(), fcS_real.tobytes(), fcB_real.tobytes()),
    ('zeros A', fcW_real.tobytes(), (np.zeros(fk, dtype=np.float32)).view(np.int32).tobytes(), fcS_real.tobytes(), fcB_real.tobytes()),
    ('0x3f800000 A (1.0)', fcW_real.tobytes(), bytes([0, 0, 0x80, 0x3f]*fk), fcS_real.tobytes(), fcB_real.tobytes()),
    ('rand A', fcW_real.tobytes(), (np.random.randn(fk)*0.1).astype(np.float32).view(np.int32).tobytes(), fcS_real.tobytes(), fcB_real.tobytes()),
]
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
for name, p_, a, s, b in variants:
    r = req(p_, a, s, b)
    p.stdin.write(r); p.stdin.flush()
    time.sleep(0.5)
    rl = p.stdout.read(4)
    sz = struct.unpack('<I', rl)[0] if len(rl)==4 else 0
    outv = p.stdout.read(sz)
    v0 = struct.unpack('<f', outv[:4])[0] if len(outv)>=4 else 'n/a'
    print(f'{name:<35} outv[0]={v0}')
p.terminate(); p.wait(timeout=5)
print('--- stderr ---')
print(''.join(lines))
