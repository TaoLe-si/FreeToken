
import os, struct, torch, safetensors.torch, numpy as np
os.chdir(r'E:\FreeToken\benchmarks\cpu_moe_microbench')
state = safetensors.torch.load_file(r'E:\models\Qwen3.6-35B-A3B-MXFP4-MTP\model-00022-of-00023.safetensors')
fc_w = state['mtp.fc.weight'][0:1].contiguous()
fc_s = state['mtp.fc.scales'][0:1].contiguous()
fc_b = state['mtp.fc.biases'][0:1].contiguous()
act = torch.randn(4096, dtype=torch.float32) * 0.1
M, K = 1, 4096
sz_p = fc_w.numel() * 4
sz_s = fc_s.numel() * 2
sz_b = fc_b.numel() * 2
hdr = struct.pack('<IIIII', M, K, sz_p, sz_s, sz_b)
data = hdr + fc_w.view(torch.uint8).contiguous().numpy().tobytes() + fc_s.view(torch.uint8).contiguous().numpy().tobytes() + fc_b.view(torch.uint8).contiguous().numpy().tobytes() + act.numpy().tobytes()
with open('t_test_input.bin', 'wb') as f:
    f.write(data)
print(f'wrote t_test_input.bin: {len(data)} bytes, hdr={M},{K},{sz_p},{sz_s},{sz_b}')
print(f'fc_w shape={list(fc_w.shape)} numel={fc_w.numel()}')
