
import os, struct, numpy as np
import safetensors.torch
os.chdir(r'E:\FreeToken\benchmarks\cpu_moe_microbench')
# Load both sources
data = safetensors.torch.load_file(r'E:\models\Qwen3.6-35B-A3B-MXFP4-MTP\model-00022-of-00023.safetensors')
fcW_safetensors = data['mtp.fc.weight']  # [2048, 512] uint32
real_data = open('t_mtp_fc_with_act.bin', 'rb').read()
fm, fk, fnb, fns = struct.unpack('<IIII', real_data[:16])
off = 16
fcW_real = np.frombuffer(real_data[off:off+fm*fnb*4], dtype=np.uint32); off += fm*fnb*4
print('fcW_safetensors[0,:5]:', fcW_safetensors[0, :5].tolist())
print('fcW_real[:5]:', fcW_real[:5].tolist())
print('Equal?', np.array_equal(fcW_safetensors[0].numpy(), fcW_real))
# Check hex
print('safetensors[0] first 16 bytes:', fcW_safetensors[0, :4].numpy().tobytes().hex())
print('real[0] first 16 bytes:        ', fcW_real[:4].tobytes().hex())
