import sys, os
sys.path.insert(0, r'E:\\FreeToken\\python')
sys.path.insert(0, r'E:\\FreeToken\\benchmarks\\cpu_moe_microbench')
os.chdir(r'E:\\FreeToken\\benchmarks\\cpu_moe_microbench')
import safetensors.torch, numpy as np
state = safetensors.torch.load_file(r'E:\\models\\Qwen3.6-35B-A3B-MXFP4-MTP\\model-00022-of-00023.safetensors')
fc_w = state['mtp.fc.weight'].numpy()
M, K = 1, 4096
ns = K // 32
packed = fc_w[:1].tobytes()
szA, szS, szB = K * 4, M * ns * 4, M * ns * 4

# Test with different act values to see if shader is reading act
for label, act_val in [("zeros", np.zeros(K, dtype='float32')), ("ones", np.ones(K, dtype='float32')), ("tenth", np.full(K, 0.1, dtype='float32')), ("random", np.random.randn(K).astype('float32'))]:
    act = act_val.view(np.int32)
    payload = b'LOAD fc 1 4096\n' + packed + b'CALL fc\n' + act.tobytes() + bytes(szS) + bytes(szB) + b'QUIT\n'
    import subprocess
    p = subprocess.Popen([r'E:\\FreeToken\\benchmarks\\cpu_moe_microbench\\t_mxfp4_gemv_multi_server.exe'],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         cwd=r'E:\\FreeToken\\benchmarks\\cpu_moe_microbench')
    try:
        out, err = p.communicate(payload, timeout=10)
        outv = out[13:17]  # 4-byte len + M*4 bytes
        import struct
        if len(out) >= 17:
            val = struct.unpack('<f', outv)[0]
            print(f'{label}: outv_bytes={outv.hex()}, value={val}')
        else:
            print(f'{label}: short output ({len(out)} bytes)')
        print('STDERR:', err.decode(errors='replace')[:500])
    except subprocess.TimeoutExpired:
        p.kill()
        print(f'{label}: TIMEOUT')
