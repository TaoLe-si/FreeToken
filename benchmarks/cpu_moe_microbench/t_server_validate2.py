import os, struct, time, subprocess, torch, safetensors.torch, numpy as np
os.chdir(r'E:\FreeToken\benchmarks\cpu_moe_microbench')
from t_mxfp4_dequant import dequant_mxfp4_weight_v2
state = safetensors.torch.load_file(r'E:\models\Qwen3.6-35B-A3B-MXFP4-MTP\model-00022-of-00023.safetensors')
fc_w = state['mtp.fc.weight'][0:1].contiguous()   # [1, 512] uint32
fc_s = state['mtp.fc.scales'][0:1].contiguous()   # [1, 128] fp16
fc_b = state['mtp.fc.biases'][0:1].contiguous()   # [1, 128] fp16
print('fc_w', tuple(fc_w.shape), fc_w.dtype)
print('fc_s', tuple(fc_s.shape), fc_s.dtype)
print('fc_b', tuple(fc_b.shape), fc_b.dtype)
np.random.seed(0)
K = 4096
act = (np.random.randn(K) * 0.1).astype(np.float32)
# CPU reference
wq = dequant_mxfp4_weight_v2(fc_w, fc_s, fc_b)
print('wq shape:', tuple(wq.shape), wq.dtype)
ref = float((wq[0] * torch.from_numpy(act)).sum().item())
print('CPU ref outv:', ref)
# build request
sz_p = fc_w.numel() * 4
sz_s = fc_s.numel() * 2
sz_b = fc_b.numel() * 2
hdr = struct.pack('<IIIII', 1, K, sz_p, sz_s, sz_b)
req = hdr + fc_w.view(torch.uint8).contiguous().numpy().tobytes() + fc_s.view(torch.uint8).contiguous().numpy().tobytes() + fc_b.view(torch.uint8).contiguous().numpy().tobytes() + act.tobytes()
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
    print('diff:', float(outv[0] - ref))
p.terminate()
p.wait(timeout=5)
print('--- stderr ---')
print(''.join(lines))
