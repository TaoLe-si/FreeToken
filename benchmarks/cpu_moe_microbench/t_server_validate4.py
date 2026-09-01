
import os, struct, time, subprocess, torch, safetensors.torch, numpy as np
os.chdir(r'E:\FreeToken\benchmarks\cpu_moe_microbench')
state = safetensors.torch.load_file(r'E:\models\Qwen3.6-35B-A3B-MXFP4-MTP\model-00022-of-00023.safetensors')
fc_w = state['mtp.fc.weight'][0:1].contiguous()   # [1, 512] uint32
print('fc_w', tuple(fc_w.shape), fc_w.dtype)
np.random.seed(0)
K = 4096
act_fp32 = (np.random.randn(K) * 0.1).astype(np.float32)
# nvfp4 server uses int (uint) for act. send as int32 same bit pattern
act_int32 = act_fp32.view(np.int32)

# CPU reference: for each block of 32, decode 4 uint32 (= 32 nibbles), sum(nibble_table[k] * act[i])
LUT = np.array([0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12], dtype=np.int32)
w = fc_w[0].view(torch.uint8).numpy().copy()  # 512 uint32 = 2048 bytes
wu = w.view(np.uint32).reshape(512)
ref = 0.0
for b in range(128):  # 128 micro-blocks of 32
    base = b * 32
    s = 0
    for j in range(8):  # 8 uints = 32 nibbles
        u = wu[base // 4 + j]
        nib = np.array([(u >> (4*k)) & 0xF for k in range(8)], dtype=np.int32)
        vals = LUT[nib]
        ai = base + j*8
        a = act_int32[ai:ai+8]
        s += int(np.dot(vals, a))
    ref += s
# gbl = 1.0, rowB = 0.0 (server default)
ref = ref * 1.0 + 0.0
print('CPU ref outv:', ref)

# build request: M=1, K=4096, szPacked=2048, szAct=16384
sz_p = fc_w.numel() * 4
sz_a = K * 4
hdr = struct.pack('<IIII', 1, K, sz_p, sz_a)
req = hdr + fc_w.view(torch.uint8).contiguous().numpy().tobytes() + act_int32.tobytes()
print('request bytes:', len(req))
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
p.stdin.write(req)
p.stdin.flush()
rl = p.stdout.read(4)
print('resp len bytes:', len(rl))
if len(rl) == 4:
    sz = struct.unpack('<I', rl)[0]
    outv = np.frombuffer(p.stdout.read(sz), dtype=np.float32)
    print('server outv:', outv)
    print('diff vs CPU:', float(outv[0] - ref))
p.terminate()
p.wait(timeout=5)
print('--- server stderr ---')
print(''.join(lines))
