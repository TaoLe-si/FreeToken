
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

# Variants of packed/scales/biases/act
def req(packed, scales, biases, act_int_bytes):
    return struct.pack('<IIIII', 1, fk, len(packed), len(scales), len(biases)) + packed + scales + biases + act_int_bytes

variants = [
    ('A: real P, real S, real B, real A', fcW_real.tobytes(), fcS_real.tobytes(), fcB_real.tobytes(), act_real.view(np.int32).tobytes()),
    ('B: real P, real S, real B, rand A', fcW_real.tobytes(), fcS_real.tobytes(), fcB_real.tobytes(), (np.random.randn(fk)*0.1).astype(np.float32).view(np.int32).tobytes()),
    ('C: real P, zero S, real B, real A', fcW_real.tobytes(), bytes([0]*512), fcB_real.tobytes(), act_real.view(np.int32).tobytes()),
    ('D: real P, real S, zero B, real A', fcW_real.tobytes(), fcS_real.tobytes(), bytes([0]*512), act_real.view(np.int32).tobytes()),
    ('E: real P, zero S, zero B, real A', fcW_real.tobytes(), bytes([0]*512), bytes([0]*512), act_real.view(np.int32).tobytes()),
    ('F: real P, zero S, zero B, rand A', fcW_real.tobytes(), bytes([0]*512), bytes([0]*512), (np.random.randn(fk)*0.1).astype(np.float32).view(np.int32).tobytes()),
    ('G: SAFETENSORS fcW[0], zero S/B, real A', fcW[0].numpy().tobytes(), bytes([0]*512), bytes([0]*512), act_real.view(np.int32).tobytes()),
    ('H: SAFETENSORS fcW[0], zero S/B, rand A', fcW[0].numpy().tobytes(), bytes([0]*512), bytes([0]*512), (np.random.randn(fk)*0.1).astype(np.float32).view(np.int32).tobytes()),
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
for name, p_, s, b, a in variants:
    r = req(p_, s, b, a)
    p.stdin.write(r); p.stdin.flush()
    time.sleep(0.3)
    rl = p.stdout.read(4)
    sz = struct.unpack('<I', rl)[0] if len(rl)==4 else 0
    outv = p.stdout.read(sz)
    v0 = struct.unpack('<f', outv[:4])[0] if len(outv)>=4 else 'n/a'
    print(f'{name:<55} outv[0]={v0}')
p.terminate(); p.wait(timeout=5)
print('--- stderr ---')
print(''.join(lines))
