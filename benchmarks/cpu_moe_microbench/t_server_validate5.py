
import os, struct, time, subprocess, numpy as np
os.chdir(r'E:\FreeToken\benchmarks\cpu_moe_microbench')
# Read t_mtp_fc_with_act.bin: header [fm,fk,fnb,fns] then fcW(fm*fnb*4) fcB(fm*fns*4) fcS(fm*fns*4) act(fk*4)
data = open('t_mtp_fc_with_act.bin', 'rb').read()
fm, fk, fnb, fns = struct.unpack('<IIII', data[:16])
print('header', fm, fk, fnb, fns)
off = 16
fcW = np.frombuffer(data[off:off+fm*fnb*4], dtype=np.uint32); off += fm*fnb*4
fcB = np.frombuffer(data[off:off+fm*fns*4], dtype=np.float32); off += fm*fns*4
fcS = np.frombuffer(data[off:off+fm*fns*4], dtype=np.float32); off += fm*fns*4
act = np.frombuffer(data[off:off+fk*4], dtype=np.float32); off += fk*4
print('fcW[0]', fcW[:5], 'fcB[0]', fcB[0], 'fcS[0]', fcS[0], 'act[:5]', act[:5])
# Build request for server (new protocol: M, K, szPacked, szAct, then packed, act)
M, K = fm, fk
sz_p = M * fnb * 4
sz_a = K * 4
hdr = struct.pack('<IIII', M, K, sz_p, sz_a)
# packed: fcW as bytes
packed_bytes = fcW.tobytes()
# act: must be int32 (same bit pattern as float32)
act_int = act.view(np.int32)
req = hdr + packed_bytes + act_int.tobytes()
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
    # CPU reference (nvfp4): out = sum(act * dequant(packed)) * gbl + rowB, with gbl=1, rowB=0
    # But what gbl/rowB should be? fc_clean loaded fcGbl=1, rowB=0 by default
    # For mtp.fc, gbl/rowB should be ... let me just compute simple nvfp4 dequant
    LUT = np.array([0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12], dtype=np.int32)
    fcW2d = fcW.reshape(M, fnb)
    refs = np.zeros(M, dtype=np.float32)
    for r in range(M):
        acc = 0
        for b in range(fns):  # micro-blocks
            # 4 uints per block
            bi = b * 4
            nibs = []
            for j in range(4):
                u = int(fcW2d[r, bi + j])
                nibs.extend([(u >> (4*k)) & 0xF for k in range(8)])
            vals = LUT[nibs]
            ai = b * 32
            acc += int(np.dot(vals.astype(np.int64), act[ai:ai+32].astype(np.int64)))
        refs[r] = float(acc)  # gbl=1, rowB=0
    print('CPU ref outv[:4]:', refs[:4])
    print('diff:', outv[:4] - refs[:4])
p.terminate(); p.wait(timeout=5)
print('--- server stderr ---')
print(''.join(lines))
