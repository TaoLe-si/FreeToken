
import os, struct, numpy as np
import safetensors.torch
os.chdir(r'E:\FreeToken\benchmarks\cpu_moe_microbench')
data = safetensors.torch.load_file(r'E:\models\Qwen3.6-35B-A3B-MXFP4-MTP\model-00022-of-00023.safetensors')
fcW = data['mtp.fc.weight']
nb, ns = 512, 128
rows = [fcW[0, :nb].numpy().tobytes()]
packed = b''.join(rows)
scales = bytes([0] * (1 * ns * 4))
biases = bytes([0] * (1 * ns * 4))
np.random.seed(42)
act = (np.random.randn(4096) * 0.1).astype(np.float32)
print('act[:3] =', act[:3])
print('act.view(int32)[:3] =', act.view(np.int32)[:3])
# Act bytes
act_int = act.view(np.int32).tobytes()
print('act_int first 16 hex:', act_int[:16].hex())
# What does fc_clean act look like?
real_data = open('t_mtp_fc_with_act.bin', 'rb').read()
fm, fk, fnb, fns = struct.unpack('<IIII', real_data[:16])
off = 16 + fm*fnb*4 + fm*fns*4*2
act_real = np.frombuffer(real_data[off:off+fk*4], dtype=np.float32)
print('act_real[:3] =', act_real[:3])
print('act_real.view(int32)[:3] =', act_real.view(np.int32)[:3])
# Are they byte-equal?
print('act all same?', np.array_equal(act, act_real))
