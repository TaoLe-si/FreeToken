
import os, struct, numpy as np
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

# Build "A" req and "E" req separately
def req_new(packed, act_bytes, scales, biases):
    return struct.pack('<IIIIII', 1, fk, len(packed), len(act_bytes), len(scales), len(biases)) + packed + act_bytes + scales + biases

req_A = req_new(fcW_real.tobytes(), act_real.view(np.int32).tobytes(), fcS_real.tobytes(), fcB_real.tobytes())
req_E = req_new(fcW[0].numpy().tobytes(), act_real.view(np.int32).tobytes(), bytes([0]*512), bytes([0]*512))
# Compare byte-by-byte
print('packed equal:', fcW_real.tobytes() == fcW[0].numpy().tobytes())
print('act equal:', act_real.view(np.int32).tobytes() == act_real.view(np.int32).tobytes())
# Send both
import subprocess, time
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

for name, r in [('A', req_A), ('E', req_E)]:
    p.stdin.write(r); p.stdin.flush()
    time.sleep(0.5)
    rl = p.stdout.read(4)
    sz = struct.unpack('<I', rl)[0] if len(rl)==4 else 0
    outv = p.stdout.read(sz)
    v0 = struct.unpack('<f', outv[:4])[0] if len(outv)>=4 else 'n/a'
    print(f'{name}: outv[0]={v0}')
p.terminate(); p.wait(timeout=5)
print('stderr:', ''.join(lines))
