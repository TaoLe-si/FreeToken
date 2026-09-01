"""Track 3 (C) Part 1: Verify what the current kernel actually computes.

FA shader math with v1/v2 binding interpretation:
  - t0 (packed): [M, K/8] uint
  - t1 (scl): M*K/32 bytes, but in v1/v2 we put K*4 bytes of act
  - t2 (act): M*ns*4 bytes, but in v1/v2 we put M*ns*4 zeros (or scales)
  - t3 (bias): K*4 bytes, but in v1/v2 we put K*4 bytes of act
  - t4 (gbl): M*4 bytes, 1.0
  Formula: outv[row] = (sum_b sum_k wsum_b * bs_b) * gbl[row] + bias[row]*gbl[row]
  where wsum_b = sum_{k in 32-block} kE2M1[w] * act[k]  (act is read from t2)
  and bs_b = exp2(sb - 127) where sb is byte from scl (read from t1)

When t1 has act bytes, sb = act[0]&0xFF for b=0. With random act ints -100..100, byte 0
is 0x00-0xFF. So bs varies wildly.

The current outputs are NOT physically meaningful. They are some hash of the data.
The "stable" baseline is whatever happens to come out.
"""
import sys, os, struct, subprocess
sys.path.insert(0, r'E:\FreeToken\python')
sys.path.insert(0, r'E:\FreeToken\benchmarks\cpu_moe_microbench')
os.chdir(r'E:\FreeToken\benchmarks\cpu_moe_microbench')
import numpy as np

V2 = r'E:\FreeToken\benchmarks\cpu_moe_microbench\t_mxfp4_gemv_v2_server.exe'
V1 = r'E:\FreeToken\benchmarks\cpu_moe_microbench\t_mxfp4_gemv_server.exe'

kE2M1 = np.array([0,1,2,3,4,6,8,12, 0,-1,-2,-3,-4,-6,-8,-12], dtype=np.int32)

def fa_gemv_with_v2_bindings(packed, M, K, t1_data, t2_data, t3_data, gbl):
    """Reference that matches the FA shader with v1/v2 binding pattern:
    t0=packed, t1=t1_data (scl slot but contains act bytes in our protocol),
    t2=t2_data (act slot, contains zeros in our protocol),
    t3=t3_data (bias slot, contains act bytes in our protocol),
    t4=gbl
    """
    nb = K // 8  # FA shader uses uint packed (4 bytes = 4 nibbles = 4 K-elements)
    ns = K // 32
    outv = np.zeros(M, dtype=np.float32)
    for row in range(M):
        acc = 0.0
        # FA shader: for b in 0..K/32-1 (with striding by 256 threads)
        for b in range(ns):
            # sb extraction: read scl[row*ns + b/4] as uint, then extract byte
            sIdx = row * ns + (b >> 2)
            byteIdx = b & 3
            sPack = t1_data[sIdx] if sIdx < len(t1_data) else 0
            if byteIdx == 0: sb = sPack & 0xFF
            elif byteIdx == 1: sb = (sPack >> 8) & 0xFF
            elif byteIdx == 2: sb = (sPack >> 16) & 0xFF
            else: sb = (sPack >> 24) & 0xFF
            bs = 0.0 if sb == 0 else np.exp2((float(sb) - 127.0))
            # FA shader: 4 uints per micro-block (4*8 = 32 K-elements)
            pBase = row * nb + b * 4
            if pBase + 3 >= len(packed): continue
            w0, w1, w2, w3 = packed[pBase], packed[pBase+1], packed[pBase+2], packed[pBase+3]
            abase = b * 32
            wsum = 0.0
            for j, w in enumerate([w0, w1, w2, w3]):
                for k in range(8):
                    nibble = (w >> (4*k)) & 0xF
                    w_e = kE2M1[nibble]
                    a_idx = abase + j*8 + k
                    a = float(t2_data[a_idx]) if a_idx < len(t2_data) else 0.0
                    wsum += float(w_e) * a
            acc += wsum * bs
        # bias[row] from t3_data (v1/v2 protocol: K*4 bytes of act)
        bias_idx = row
        bias = float(t3_data[bias_idx]) if bias_idx < len(t3_data) else 0.0
        outv[row] = (acc + bias) * float(gbl[row])
    return outv

# Test with deterministic data
M = 1
K = 4096
ns = K // 32
nb = K // 8

rng = np.random.default_rng(42)
packed = rng.integers(0, 2**32, size=(M, nb), dtype=np.uint32)  # K/8 uints per row
act = rng.integers(-100, 100, size=(K,)).astype(np.float32)
# t1 = act bytes (M*K/4 = K*4 bytes for M=1) - this is the bug
t1_data = act.view(np.uint32).copy()  # K uints
# t2 = zeros (M*ns*4 = K/32*4 bytes for M=1)
t2_data = np.zeros(K, dtype=np.uint32)  # K uints of zeros
# t3 = act bytes (K*4 bytes)
t3_data = act.view(np.uint32).copy()  # K uints
gbl = np.ones(M, dtype=np.float32)

# Compute reference
ref = fa_gemv_with_v2_bindings(packed, M, K, t1_data, t2_data, t3_data, gbl)
print(f'Reference (Python sim of v1/v2 binding+FA shader): {ref}')

# Run v1 server
hdr = struct.pack('<IIIIII', M, K, packed.size*4, K*4, M*ns*4, M*ns*4)
payload = hdr + packed.tobytes() + act.tobytes() + b'\x00' * (M*ns*4)*2
p = subprocess.Popen([V1], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                     cwd=os.path.dirname(V1))
out, _ = p.communicate(payload, timeout=30)
v1_out = struct.unpack('<f', out[4:8])[0]
print(f'v1 server output: {v1_out}')

# Run v2 server STATELESS
szP = packed.size*4; szA = K*4; szS = M*ns*4; szB = M*ns*4
v2_payload = f'STATELESS {M} {K} {szP} {szA} {szS} {szB}\n'.encode() + packed.tobytes() + act.tobytes() + b'\x00'*(szS+szB) + b'QUIT\n'
p = subprocess.Popen([V2], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                     cwd=os.path.dirname(V2))
out, _ = p.communicate(v2_payload, timeout=30)
v2_out = struct.unpack('<f', out[4:8])[0]
print(f'v2 server output: {v2_out}')

print(f'\n  Reference == v1? {abs(ref[0] - v1_out) < 1}')
print(f'  Reference == v2? {abs(ref[0] - v2_out) < 1}')
print(f'  v1 == v2? {abs(v1_out - v2_out) < 1}')
print(f'\n  Numerical understanding: outputs are arbitrary hash of data, not real GEMV.')
