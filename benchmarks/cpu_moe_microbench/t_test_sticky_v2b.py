import sys, os, struct
sys.path.insert(0, r'E:\FreeToken\python')
sys.path.insert(0, r'E:\FreeToken\benchmarks\cpu_moe_microbench')
os.chdir(r'E:\FreeToken\benchmarks\cpu_moe_microbench')
import safetensors.torch, numpy as np

state = safetensors.torch.load_file(r'E:\models\Qwen3.6-35B-A3B-MXFP4-MTP\model-00022-of-00023.safetensors')
fc_w = state['mtp.fc.weight'].numpy()
M, K = 1, 4096
ns = K // 32
nb = K // 8
packed = fc_w[:1].tobytes()
szA, szS, szB = K * 4, M * ns * 4, M * ns * 4
act = np.full(K, 0.05, dtype='float32')
payload = b'LOAD fc ' + str(M).encode() + b' ' + str(K).encode() + b' ' + str(len(packed)).encode() + b'\n'
payload += packed
payload += b'CALL fc ' + str(szA).encode() + b' ' + str(szS).encode() + b' ' + str(szB).encode() + b'\n'
payload += act.tobytes() + bytes(szS) + bytes(szB)
payload += b'QUIT\n'

import subprocess
p = subprocess.Popen([r'E:\FreeToken\benchmarks\cpu_moe_microbench\t_mxfp4_gemv_v2_server.exe'],
                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                     cwd=r'E:\FreeToken\benchmarks\cpu_moe_microbench')
out, err = p.communicate(payload, timeout=10)
print('STDOUT len:', len(out))
print('STDOUT bytes:', out.hex())
# The first part should be the LOAD reply "OK fc 1 4096\n" = 12 bytes
# Then CALL reply 4-byte len + M*4 bytes = 4 + 4 = 8 bytes
# Look for "OK " marker
ok_idx = out.find(b'OK ')
print(f'OK marker at offset {ok_idx}')
if ok_idx >= 0:
    # Find the newline after OK
    nl = out.find(b'\n', ok_idx)
    print(f'OK reply: {out[ok_idx:nl+1].decode()}')
    call_start = nl + 1
    print(f'CALL response (8 bytes): {out[call_start:call_start+8].hex()}')
    sz = struct.unpack('<I', out[call_start:call_start+4])[0]
    val = struct.unpack('<f', out[call_start+4:call_start+8])[0]
    print(f'  sz={sz} outv={val}')
print('STDERR:', err.decode(errors='replace')[:1500])
