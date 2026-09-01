import sys
sys.path.insert(0, r'E:/FreeToken/python')
import os, time
import numpy as np
import torch

os.environ['PATH'] = r'C:\Windows\system32;C:\Windows'

print('='*70)
print('INTEGRATION TEST: All MTP iGPU paths through HIP on AMD Radeon 780M')
print('='*70)

# === 1. Route kernel via HIP ===
print()
print('[1/3] MoE Route via HIP/ROCm:')
from freetoken.kernel.igpu_route import make_igpu_route_client
torch.manual_seed(42)
E, H = 256, 2048
router_w = torch.randn(E, H, dtype=torch.float32) * 0.01
client = make_igpu_route_client(prefer_hip=True, E=E, H=H)
print(f'  Client type: {type(client).__name__}')
client.load(router_w)
hidden = torch.randn(H, dtype=torch.float32) * 0.1
for _ in range(2): client.forward(hidden)
t0 = time.time()
for _ in range(5): client.forward(hidden)
ms = (time.time()-t0)/5*1000
print(f'  Forward latency: {ms:.1f} ms')
client.close()

# === 2. FC via HIP ===
print()
print('[2/3] FC via HIP/ROCm:')
from freetoken.kernel.igpu_fc import make_igpu_fc_sticky
M, K = 8, 4096
np.random.seed(42)
packed_u32 = np.zeros((M, K//8), dtype=np.uint32)
for m in range(M):
    for u in range(K//8):
        v = 0
        for i in range(8):
            v |= (np.random.randint(0, 16) & 0xF) << (i*4)
        packed_u32[m, u] = v
scales = (np.random.rand(M, K//32) * 0.5 - 0.25).astype(np.float32)
biases = (np.random.rand(M, K//32) * 0.05).astype(np.float32)
fc = make_igpu_fc_sticky(packed_u32, K, scales_f32=scales, biases_f32=biases)
print(f'  Backend: {type(fc).__name__}')
act = np.random.randn(K).astype(np.float32) * 0.1
for _ in range(3): fc(act)
t0 = time.time()
for _ in range(10): fc(act)
ms = (time.time()-t0)/10*1000
print(f'  fc_call latency: {ms:.1f} ms')
logs = fc.get_log(20)
hip_count = sum(1 for l in logs if 'hip' in l.lower() or 'HIP' in l)
print(f'  HIP log lines: {hip_count}/{len(logs)}')
fc.close()

# === 3. MtpHead full forward ===
print()
print('[3/3] MtpHead full forward:')
if torch.cuda.is_available():
    from freetoken.models.qwen3_5_moe.mtp import Qwen3_5MtpHead
    class Cfg:
        hidden_size = 2048
        vocab_size = 248320
        num_qo_heads = 16
        num_kv_heads = 2
        head_dim = 128
        partial_rotary_factor = 0.5
        rope_base = 10000.0
        num_experts = 4
        num_experts_per_tok = 2
        moe_intermediate = 64
        shared_expert_intermediate = 64
        norm_topk_prob = True
    cfg = Cfg()
    device = 'cuda'
    dtype = torch.bfloat16
    torch.manual_seed(42)
    head = Qwen3_5MtpHead(cfg, torch.nn.Embedding(cfg.vocab_size, cfg.hidden_size),
                          torch.nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False),
                          igpu_fc=None, dtype=torch.float32).to(device)
    head = head.to(dtype)
    head.embed_table = head.embed_table.to(dtype)
    head.lm_head = head.lm_head.to(dtype)
    head.attn.qkv_proj = head.attn.qkv_proj.to(dtype)
    head.attn.o_proj = head.attn.o_proj.to(dtype)
    fc_t = torch.nn.Linear(2*cfg.hidden_size, cfg.hidden_size, bias=False).to(device, dtype)
    fc_t.weight.data = torch.randn(cfg.hidden_size, 2*cfg.hidden_size, dtype=dtype, device=device) * 0.02
    class T:
        def __init__(s, fc): s.fc = fc
        def __call__(s, x): return s.fc(x.to(torch.bfloat16))
    head.igpu_fc = T(fc_t)
    head.dtype = dtype
    prev_token_id = torch.tensor([42], device=device)
    prev_hidden = torch.randn(1, cfg.hidden_size, dtype=dtype, device=device) * 0.1
    with torch.no_grad():
        logits, state = head.forward_with_state(prev_token_id, prev_hidden, 5)
    print(f'  logits shape: {logits.shape}')
    print(f'  state shape: {state.shape}')
    print(f'  logits no NaN: {not torch.isnan(logits).any()}')
    print(f'  state no NaN: {not torch.isnan(state).any()}')
else:
    print('  CUDA not available, skipping')

print()
print('='*70)
print('ALL INTEGRATION TESTS COMPLETE')
print('='*70)