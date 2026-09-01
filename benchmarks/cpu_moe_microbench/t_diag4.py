
import os, struct, numpy as np
import safetensors.torch
os.chdir(r'E:\FreeToken\benchmarks\cpu_moe_microbench')
data = safetensors.torch.load_file(r'E:\models\Qwen3.6-35B-A3B-MXFP4-MTP\model-00022-of-00023.safetensors')
fcW = data['mtp.fc.weight']
real_data = open('t_mtp_fc_with_act.bin', 'rb').read()
fm, fk, fnb, fns = struct.unpack('<IIII', real_data[:16])
off = 16
fcW_real = np.frombuffer(real_data[off:off+fm*fnb*4], dtype=np.uint32); off += fm*fnb*4
fcB_real = np.frombuffer(real_data[off:off+fm*fns*4], dtype=np.float32); off += fm*fns*4
fcS_real = np.frombuffer(real_data[off:off+fm*fns*4], dtype=np.float32); off += fm*fns*4
act_real = np.frombuffer(real_data[off:off+fk*4], dtype=np.float32); off += fk*4

def make_req(M, K, seed=42):
    nb = K // 8
    ns = K // 32
    rows = [fcW[r % fcW.shape[0], :nb].numpy().tobytes() for r in range(M)]
    packed = b''.join(rows)
    scales = bytes([0] * (M * ns * 4))
    biases = bytes([0] * (M * ns * 4))
    np.random.seed(seed)
    act = (np.random.randn(K) * 0.1).astype(np.float32)
    act_int = act.view(np.int32).tobytes()
    hdr = struct.pack('<IIIII', M, K, len(packed), M * ns * 4, M * ns * 4)
    return hdr + packed + scales + biases + act_int

req_make = make_req(1, 4096, seed=42)
req_real = struct.pack('<IIIII', fm, fk, fm*fnb*4, fm*fns*4, fm*fns*4) + fcW_real.tobytes() + fcS_real.tobytes() + fcB_real.tobytes() + act_real.view(np.int32).tobytes()
print('req_make len:', len(req_make))
print('req_real len:', len(req_real))
print('headers match:', req_make[:20] == req_real[:20])
# Compare packed bytes
mp = req_make[20:20+fm*fnb*4]
rp = req_real[20:20+fm*fnb*4]
print('packed equal:', mp == rp)
# Scales/biases
ms = req_make[20+fm*fnb*4:20+fm*fnb*4+fm*fns*4]
rs = req_real[20+fm*fnb*4:20+fm*fnb*4+fm*fns*4]
print('scales len make:', len(ms), 'real:', len(rs), 'make is all zero:', ms == bytes([0])*len(ms))
print('real scales first 16:', rs[:16].hex())
mb = req_make[20+fm*fnb*4+fm*fns*4:20+fm*fnb*4+fm*fns*4*2]
rb = req_real[20+fm*fnb*4+fm*fns*4:20+fm*fnb*4+fm*fns*4*2]
print('biases len make:', len(mb), 'real:', len(rb), 'make is all zero:', mb == bytes([0])*len(mb))
print('real biases first 16:', rb[:16].hex())
# Act
ma = req_make[20+fm*fnb*4+fm*fns*4*2:]
ra = req_real[20+fm*fnb*4+fm*fns*4*2:]
print('act len make:', len(ma), 'real:', len(ra))
print('act make first 16:', ma[:16].hex())
print('act real first 16:', ra[:16].hex())
