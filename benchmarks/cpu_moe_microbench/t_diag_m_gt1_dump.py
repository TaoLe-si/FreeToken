"""Run v1 with M=2 and dump full stderr + add a dispatch print."""
import sys, os, struct, subprocess
sys.path.insert(0, r'E:\FreeToken\python')
sys.path.insert(0, r'E:\FreeToken\benchmarks\cpu_moe_microbench')
os.chdir(r'E:\FreeToken\benchmarks\cpu_moe_microbench')
import numpy as np

V1 = r'E:\FreeToken\benchmarks\cpu_moe_microbench\t_mxfp4_gemv_server.exe'
K = 4096
rng = np.random.default_rng(10)
packed1 = rng.integers(0, 2**32, size=(1, K // 8), dtype=np.uint32)
packed1b = rng.integers(0, 2**32, size=(1, K // 8), dtype=np.uint32)
packed2 = np.zeros((2, K // 8), dtype=np.uint32)
packed2[0] = packed1[0]
packed2[1] = packed1b[0]
act = (rng.integers(-100, 100, size=(K,)).astype(np.float32))

# Run v1 M=2
hdr = struct.pack('<IIIIII', 2, K, packed2.size*4, K*4, 2*(K//32)*4, 2*(K//32)*4)
payload = hdr + packed2.tobytes() + act.tobytes() + b'\x00'*(2*(K//32)*4)*2
p = subprocess.Popen([V1], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                     cwd=os.path.dirname(V1))
out, err = p.communicate(payload, timeout=30)
print('=== v1 M=2 stderr ===')
print(err.decode(errors='replace'))
print('=== output ===')
sz = struct.unpack('<I', out[:4])[0]
print(f'sz={sz} raw={out[4:4+sz].hex()}')
vals = np.frombuffer(out[4:4+sz], dtype=np.float32)
print(f'vals: {vals}')

# Run v1 M=1 with same row 1 alone
print('\n=== v1 M=1 (just row 1) stderr ===')
hdr = struct.pack('<IIIIII', 1, K, packed1b.size*4, K*4, 1*(K//32)*4, 1*(K//32)*4)
payload = hdr + packed1b.tobytes() + act.tobytes() + b'\x00'*(1*(K//32)*4)*2
p = subprocess.Popen([V1], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                     cwd=os.path.dirname(V1))
out, err = p.communicate(payload, timeout=30)
print(err.decode(errors='replace'))
sz = struct.unpack('<I', out[:4])[0]
print(f'sz={sz} raw={out[4:4+sz].hex()}')
print(f'vals: {np.frombuffer(out[4:4+sz], dtype=np.float32)}')
