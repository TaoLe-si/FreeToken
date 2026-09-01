
import os, struct, time, subprocess, numpy as np
os.chdir(r'E:\FreeToken\benchmarks\cpu_moe_microbench')
data = open('t_mtp_fc_with_act.bin', 'rb').read()
fm, fk, fnb, fns = struct.unpack('<IIII', data[:16])
print('header', fm, fk, fnb, fns)
off = 16
fcW = np.frombuffer(data[off:off+fm*fnb*4], dtype=np.uint32); off += fm*fnb*4
fcB = np.frombuffer(data[off:off+fm*fns*4], dtype=np.float32); off += fm*fns*4
fcS = np.frombuffer(data[off:off+fm*fns*4], dtype=np.float32); off += fm*fns*4
act = np.frombuffer(data[off:off+fk*4], dtype=np.float32); off += fk*4
# Build request: M, K, szPacked, szScales, szBiases (5 uints) + packed + scales + biases + act
M, K = fm, fk
sz_p = M * fnb * 4
sz_s = M * fns * 4
sz_b = M * fns * 4
hdr = struct.pack('<IIIII', M, K, sz_p, sz_s, sz_b)
# act as int32 bit pattern
act_int = act.view(np.int32)
req = hdr + fcW.tobytes() + fcS.tobytes() + fcB.tobytes() + act_int.tobytes()
print('request size', len(req))
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
p.stdin.write(req); p.stdin.flush()
rl = p.stdout.read(4)
print('resp len bytes:', len(rl))
if len(rl) == 4:
    sz = struct.unpack('<I', rl)[0]
    outv = np.frombuffer(p.stdout.read(sz), dtype=np.float32)
    print('server outv[:4]:', outv[:4])
p.terminate(); p.wait(timeout=5)
print('--- server stderr ---')
print(''.join(lines))
