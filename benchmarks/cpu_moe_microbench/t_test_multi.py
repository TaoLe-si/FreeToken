"""Test MULTI_GEMV command."""
import sys, os, struct, subprocess
import numpy as np

SERVER = r'E:\FreeToken\benchmarks\cpu_moe_microbench\t_mxfp4_gemv_v3_server.exe'
B, K = 4, 64
np.random.seed(42)
# B items, each M=1 K=K
packed_list = [np.random.randint(0, 2**32, size=(K // 8,), dtype=np.uint32) for _ in range(B)]
scales_list = [np.random.uniform(0.001, 0.01, size=(K // 32,)).astype(np.float32) for _ in range(B)]
act_list = [np.random.randn(K).astype(np.float32) for _ in range(B)]
bias_list = [np.zeros(1, dtype=np.float32) for _ in range(B)]
gbl_list = [np.ones(1, dtype=np.float32) for _ in range(B)]

szPPer = K // 8 * 4
szSPer = K // 32 * 4
szAPer = K * 4
szBPer = 1 * 4
gblPer = 1 * 4

# MULTI_GEMV command
cmd = f'MULTI_GEMV {B} {K} {szPPer} {szSPer} {szAPer} {szBPer} {gblPer}\n'
body = b''
for i in range(B):
    body += packed_list[i].tobytes() + scales_list[i].tobytes() + act_list[i].tobytes() + bias_list[i].tobytes() + gbl_list[i].tobytes()
cmd_bytes = cmd.encode() + body + b'QUIT\n'

p = subprocess.Popen([SERVER], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=os.path.dirname(SERVER))
out, err = p.communicate(cmd_bytes, timeout=30)
print(f'stderr: {err.decode(errors="replace")[:500]}')
sz = struct.unpack('<I', out[:4])[0]
print(f'sz = {sz} (expected {B*4}={B*4})')
if sz == B * 4:
    out_arr = np.frombuffer(out[4:4+sz], dtype=np.float32).copy()
    print(f'MULTI_GEMV output: {out_arr}')

# Manual
kE2M1 = np.array([0,1,2,3,4,6,8,12, 0,-1,-2,-3,-4,-6,-8,-12], dtype=np.int32)
ns = K // 32
manual = np.zeros(B, dtype=np.float32)
for b in range(B):
    for mb in range(ns):
        bs_sum = 0.0
        for j in range(4):
            w = int(packed_list[b][mb*4 + j])
            for byte_idx in range(4):
                byte = (w >> (byte_idx * 8)) & 0xFF
                nibble_lo = byte & 0xF
                bs_sum += kE2M1[nibble_lo] * act_list[b][mb*32 + j*8 + byte_idx*2]
                nibble_hi = (byte >> 4) & 0xF
                bs_sum += kE2M1[nibble_hi] * act_list[b][mb*32 + j*8 + byte_idx*2 + 1]
        manual[b] += bs_sum * scales_list[b][mb]
print(f'Manual: {manual}')
print(f'Match: {bool(np.allclose(out_arr, manual, atol=1e-3))}')
