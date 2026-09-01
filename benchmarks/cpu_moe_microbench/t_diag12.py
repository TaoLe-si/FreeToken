
import os, struct, numpy as np
import safetensors.torch
os.chdir(r'E:\FreeToken\benchmarks\cpu_moe_microbench')
data = safetensors.torch.load_file(r'E:\models\Qwen3.6-35B-A3B-MXFP4-MTP\model-00022-of-00023.safetensors')
fcW = data['mtp.fc.weight']  # [2048, 512] uint32
fcS = data['mtp.fc.scales']  # [2048, 128] fp16
fcB = data['mtp.fc.biases']  # [2048, 128] fp16
print('fcW shape', fcW.shape, 'numel', fcW.numel(), 'bytes', fcW.numel()*4)
print('fcS shape', fcS.shape, 'numel', fcS.numel(), 'bytes', fcS.numel()*2)
print('fcB shape', fcB.shape, 'numel', fcB.numel(), 'bytes', fcB.numel()*2)
# K=4096, ns=K/32=128, M=1
# M*ns*4 = 1*128*4 = 512
# fcS row 0 has 128 fp16 = 256 bytes
# So fcS_real (256 bytes) does NOT match M*ns*4 (512 bytes)!
# Server expects szScales = M*ns*4 = 512
# A uses fcS_real (256 bytes) — but sends 256 bytes, server expects 512!
# Server reads 256 bytes then tries to read biases — but biases 256 bytes
# Total read = 256+256 = 512 bytes for S+B combined, but server parsed 1024 bytes (512+512)
# So server will read 512 extra bytes from act! corrupting act data!
print('MISMATCH! fcS row 0 has 256 bytes but server expects 512 (M*ns*4 with float32)')
print('Or: server uses scales as float32 (4 bytes), fcS uses fp16 (2 bytes)')
