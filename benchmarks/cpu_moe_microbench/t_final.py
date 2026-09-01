import sys, os, struct, subprocess
import numpy as np
import safetensors.torch

SERVER = r'E:\FreeToken\benchmarks\cpu_moe_microbench\t_mxfp4_gemv_v3_server.exe'
state = safetensors.torch.load_file(r'E:\models\Qwen3.6-35B-A3B-MXFP4-MTP\model-00022-of-00023.safetensors')
fc_w = state['mtp.fc.weight']
fc_s = state['mtp.fc.scales']

M, K = 1, 4096
szP = fc_w[0:1].numpy().size * 4
szS = fc_s[0:1].float().numpy().size * 4
szA = K * 4
szB = M * 4
payload = f'STATELESS {M} {K} {szP} {szS} {szA} {szB}\n'.encode()
payload += fc_w[0:1].numpy().tobytes()
payload += fc_s[0:1].float().numpy().tobytes()
payload += (0.05 * np.ones(K, dtype=np.float32)).tobytes()
payload += np.zeros(M, dtype=np.float32).tobytes()  # bias
payload += b'QUIT\n'

print(f'Payload sizes:')
print(f'  packed: {szP} bytes ({len(fc_w[0:1].numpy().tobytes())})')
print(f'  scales: {szS} bytes ({len(fc_s[0:1].float().numpy().tobytes())})')
print(f'  act: {szA} bytes ({len((0.05 * np.ones(K, dtype=np.float32)).tobytes())})')
print(f'  bias: {szB} bytes')
print(f'  total header: {len(payload.split(chr(10))[0])} + body: {len(payload) - len(payload.split(chr(10))[0]) - 1}')
print(f'  expected szO: {M * (K // 32) * 4} bytes')

p = subprocess.Popen([SERVER], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=os.path.dirname(SERVER))
out, err = p.communicate(payload, timeout=30)
print(f'\nOutput: {len(out)} bytes')
if len(out) >= 4:
    sz = struct.unpack('<I', out[:4])[0]
    print(f'sz: {sz}')
    if sz > 0 and sz < 100000:
        v = np.frombuffer(out[4:4+sz], dtype=np.float32).copy()[0]
        print(f'value: {v}')
