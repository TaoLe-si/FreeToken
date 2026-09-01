
import os, struct, torch, safetensors.torch, numpy as np
os.chdir(r'E:\FreeToken\benchmarks\cpu_moe_microbench')
state = safetensors.torch.load_file(r'E:\models\Qwen3.6-35B-A3B-MXFP4-MTP\model-00022-of-00023.safetensors')
fc_w = state['mtp.fc.weight'][0:1].contiguous()
fc_s = state['mtp.fc.scales'][0:1].contiguous()
fc_b = state['mtp.fc.biases'][0:1].contiguous()
np.random.seed(0)
act = (np.random.randn(4096) * 0.1).astype(np.float32)
M, K = 1, 4096
sz_p = fc_w.numel() * 4
sz_a = K * 4
sz_s = fc_s.numel() * 2
sz_b = fc_b.numel() * 2
# NEW: header 6 uints M, K, szPacked, szAct, szScales, szBiases
hdr = struct.pack('<IIIIII', M, K, sz_p, sz_a, sz_s, sz_b)
req = hdr + fc_w.view(torch.uint8).contiguous().numpy().tobytes() + act.view(np.int32).tobytes() + fc_s.view(torch.uint8).contiguous().numpy().tobytes() + fc_b.view(torch.uint8).contiguous().numpy().tobytes()
with open('t_test_input.bin', 'wb') as f:
    f.write(req)
print(f'wrote t_test_input.bin: {len(req)} bytes')
print(f'header: M={M} K={K} szP={sz_p} szA={sz_a} szS={sz_s} szB={sz_b}')
