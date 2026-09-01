import torch
import torch.nn.functional as F

# Numerical alignment for Phase 2.3 MoE HLSL kernel.
# Python port of HLSL MoE algorithm vs PyTorch reference.
# Diff should be < 1e-4 (algorithmic correctness, GPU side has bf16 noise).

def moe_route_py(hidden, router_w):
    logits = router_w @ hidden
    sorted_lv, sorted_idx = torch.sort(logits, descending=True)
    top8_idx = sorted_idx[:8]
    top8_lv = sorted_lv[:8]
    maxv = top8_lv.max()
    esum = torch.exp(top8_lv - maxv).sum()
    top8_w = torch.exp(top8_lv - maxv) / esum
    return top8_idx, top8_w

def moe_expert_8x_py(hidden, expert_gate, expert_up, expert_down, top8_idx):
    H = hidden.shape[0]
    expert_out = torch.zeros(8, H, dtype=torch.float32)
    for grp in range(8):
        e = top8_idx[grp].item()
        gate_act = expert_gate[e] @ hidden
        up_act = expert_up[e] @ hidden
        su = F.silu(gate_act) * up_act
        expert_out[grp] = expert_down[e] @ su
    return expert_out

def moe_shared_py(hidden, sgate, sup, sdown, sgw):
    H = hidden.shape[0]
    gate_act = sgate @ hidden
    up_act = sup @ hidden
    su = F.silu(gate_act) * up_act
    sh_out = sdown @ su
    g = torch.sigmoid(sgw @ hidden)
    return sh_out * g

def moe_combine_py(expert_out, top8_w, shared_out):
    out = shared_out.clone()
    for e in range(8):
        out += top8_w[e] * expert_out[e]
    return out

def moe_forward_py(hidden, weights):
    top8_idx, top8_w = moe_route_py(hidden, weights['router_w'])
    expert_out = moe_expert_8x_py(hidden, weights['expert_gate'], weights['expert_up'], weights['expert_down'], top8_idx)
    shared_out = moe_shared_py(hidden, weights['sgate'], weights['sup'], weights['sdown'], weights['sgw'])
    return moe_combine_py(expert_out, top8_w, shared_out)

def moe_reference_torch(hidden, weights):
    logits = weights['router_w'] @ hidden
    top8_lv, top8_idx = torch.topk(logits, 8)
    top8_w = F.softmax(top8_lv, dim=0)
    expert_out = torch.zeros(8, weights['expert_gate'].shape[2], dtype=torch.float32)
    for grp in range(8):
        e = top8_idx[grp].item()
        gate_act = weights['expert_gate'][e] @ hidden
        up_act = weights['expert_up'][e] @ hidden
        su = F.silu(gate_act) * up_act
        expert_out[grp] = weights['expert_down'][e] @ su
    sgate = weights['sgate'] @ hidden
    sup = weights['sup'] @ hidden
    su = F.silu(sgate) * sup
    sh_out = weights['sdown'] @ su
    g = torch.sigmoid(weights['sgw'] @ hidden)
    sh_out = sh_out * g
    out = sh_out.clone()
    for e in range(8):
        out += top8_w[e] * expert_out[e]
    return out, top8_idx, top8_w

if __name__ == '__main__':
    torch.manual_seed(42)
    E, I, H = 256, 512, 2048
    weights = {
        'expert_gate': torch.randn(E, I, H) * 0.01,
        'expert_up':   torch.randn(E, I, H) * 0.01,
        'expert_down': torch.randn(E, H, I) * 0.01,
        'sgate':       torch.randn(I, H) * 0.01,
        'sup':         torch.randn(I, H) * 0.01,
        'sdown':       torch.randn(H, I) * 0.01,
        'sgw':         torch.randn(H) * 0.01,
        'router_w':    torch.randn(E, H) * 0.01,
    }
    hidden = torch.randn(H) * 0.1
    out_py = moe_forward_py(hidden, weights)
    out_ref, _, _ = moe_reference_torch(hidden, weights)
    diff = (out_py - out_ref).abs().max().item()
    print(f'MoE kernel output diff: {diff:.6f}')
    if diff < 1e-4:
        print('PASS: HLSL MoE port matches PyTorch reference')
    else:
        print(f'FAIL: diff {diff} > 1e-4')