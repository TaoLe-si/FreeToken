import torch
import torch.nn.functional as F

# Numerical alignment for Phase 2.4 Attn HLSL kernel.
# Python port of HLSL attn algorithm vs PyTorch reference.
# Diff should be < 1e-4 (algorithmic correctness, GPU side has bf16 noise).

H = 2048
n_q = 16
n_kv = 2
head_dim = 128
n_q_per_kv = n_q // n_kv
QKV_OUT = (n_q + n_kv * 2) * head_dim

def rms_norm(x, w):
    return x / torch.sqrt((x ** 2).mean(-1, keepdim=True) + 1e-6) * w

def attn_qkv_py(hidden, qkv_w, q_norm_w, k_norm_w):
    qkv = qkv_w @ hidden
    Q = qkv[:n_q * head_dim].reshape(n_q, head_dim)
    K = qkv[n_q * head_dim : n_q * head_dim + n_kv * head_dim].reshape(n_kv, head_dim)
    V = qkv[n_q * head_dim + n_kv * head_dim :].reshape(n_kv, head_dim)
    Q = rms_norm(Q, q_norm_w)
    K = rms_norm(K, k_norm_w)
    return Q, K, V

def rope_py(Q, K, position, rope_inv_freq):
    cos = torch.cos(position * rope_inv_freq)
    sin = torch.sin(position * rope_inv_freq)
    Q_out = Q.clone()
    K_out = K.clone()
    for h in range(Q.shape[0]):
        for d in range(0, head_dim, 2):
            q0 = Q[h, d]
            q1 = Q[h, d+1]
            Q_out[h, d]   = q0 * cos[d//2] - q1 * sin[d//2]
            Q_out[h, d+1] = q1 * cos[d//2] + q0 * sin[d//2]
    for h in range(K.shape[0]):
        for d in range(0, head_dim, 2):
            k0 = K[h, d]
            k1 = K[h, d+1]
            K_out[h, d]   = k0 * cos[d//2] - k1 * sin[d//2]
            K_out[h, d+1] = k1 * cos[d//2] + k0 * sin[d//2]
    return Q_out, K_out

def attn_qkt_py(Q, K, V, kv_cache_K, kv_cache_V, gate_w):
    new_kv_K = torch.cat([kv_cache_K, K.unsqueeze(0)], dim=0)
    new_kv_V = torch.cat([kv_cache_V, V.unsqueeze(0)], dim=0)
    seq_len = kv_cache_K.shape[0]
    scores = torch.zeros(n_q, seq_len + 1, dtype=torch.float32)
    for h in range(n_q):
        kv_h = h // n_q_per_kv
        for t in range(seq_len + 1):
            scores[h, t] = (Q[h] * new_kv_K[t, kv_h]).sum()
    scores = scores / (head_dim ** 0.5)
    probs = F.softmax(scores, dim=-1)
    out = torch.zeros(n_q, head_dim, dtype=torch.float32)
    for h in range(n_q):
        kv_h = h // n_q_per_kv
        for t in range(seq_len + 1):
            out[h] += probs[h, t] * new_kv_V[t, kv_h]
    g = torch.sigmoid(gate_w @ torch.cat([Q.flatten(), K.flatten(), V.flatten()]))
    attn_out = (out * g).flatten()
    return attn_out, new_kv_K, new_kv_V

def attn_forward_py(hidden, kv_cache_K, kv_cache_V, weights, position):
    Q, K, V = attn_qkv_py(hidden, weights['qkv_w'], weights['q_norm_w'], weights['k_norm_w'])
    Q, K = rope_py(Q, K, position, weights['rope_inv_freq'])
    attn_out, new_K, new_V = attn_qkt_py(Q, K, V, kv_cache_K, kv_cache_V, weights['gate_w'])
    o = weights['o_w'] @ attn_out
    return o, new_K, new_V

def attn_reference_torch(hidden, kv_cache_K, kv_cache_V, weights, position):
    QKV = weights['qkv_w'] @ hidden
    Q = QKV[:n_q * head_dim].reshape(n_q, head_dim)
    K = QKV[n_q * head_dim : n_q * head_dim + n_kv * head_dim].reshape(n_kv, head_dim)
    V = QKV[n_q * head_dim + n_kv * head_dim :].reshape(n_kv, head_dim)
    Q = rms_norm(Q, weights['q_norm_w'])
    K = rms_norm(K, weights['k_norm_w'])
    cos = torch.cos(position * weights['rope_inv_freq'])
    sin = torch.sin(position * weights['rope_inv_freq'])
    Q_rot = torch.zeros_like(Q)
    for h in range(n_q):
        for d in range(0, head_dim, 2):
            Q_rot[h, d]   = Q[h, d] * cos[d//2] - Q[h, d+1] * sin[d//2]
            Q_rot[h, d+1] = Q[h, d+1] * cos[d//2] + Q[h, d] * sin[d//2]
    Q = Q_rot
    K_rot = torch.zeros_like(K)
    for h in range(n_kv):
        for d in range(0, head_dim, 2):
            K_rot[h, d]   = K[h, d] * cos[d//2] - K[h, d+1] * sin[d//2]
            K_rot[h, d+1] = K[h, d+1] * cos[d//2] + K[h, d] * sin[d//2]
    K = K_rot
    new_K = torch.cat([kv_cache_K, K.unsqueeze(0)], dim=0)
    new_V = torch.cat([kv_cache_V, V.unsqueeze(0)], dim=0)
    K_rep = new_K.repeat_interleave(n_q_per_kv, dim=1)
    V_rep = new_V.repeat_interleave(n_q_per_kv, dim=1)
    K_rep_t = K_rep.permute(1, 0, 2)
    V_rep_t = V_rep.permute(1, 0, 2)
    scores = (Q.unsqueeze(1) * K_rep_t).sum(-1) / (head_dim ** 0.5)
    probs = F.softmax(scores, dim=-1)
    out = (probs.unsqueeze(-1) * V_rep_t).sum(1)
    g = torch.sigmoid(weights['gate_w'] @ torch.cat([Q.flatten(), K.flatten(), V.flatten()]))
    attn_out = (out * g).flatten()
    o = weights['o_w'] @ attn_out
    return o, new_K, new_V

if __name__ == '__main__':
    torch.manual_seed(42)
    weights = {
        'qkv_w':       torch.randn(QKV_OUT, H) * 0.02,
        'o_w':         torch.randn(H, n_q * head_dim) * 0.02,
        'q_norm_w':    (torch.ones(n_q, head_dim) + torch.randn(n_q, head_dim) * 0.01),
        'k_norm_w':    (torch.ones(n_kv, head_dim) + torch.randn(n_kv, head_dim) * 0.01),
        'gate_w':      torch.randn(n_q * head_dim + n_kv * head_dim + n_kv * head_dim) * 0.01,
        'rope_inv_freq': torch.exp(-torch.arange(0, head_dim, 2).float() * 5.0 / head_dim),
    }
    seq_len = 4
    kv_cache_K = torch.randn(seq_len, n_kv, head_dim) * 0.1
    kv_cache_V = torch.randn(seq_len, n_kv, head_dim) * 0.1
    hidden = torch.randn(H) * 0.1
    position = 5
    out_py, new_K_py, new_V_py = attn_forward_py(hidden, kv_cache_K, kv_cache_V, weights, position)
    out_ref, new_K_ref, new_V_ref = attn_reference_torch(hidden, kv_cache_K, kv_cache_V, weights, position)
    diff = (out_py - out_ref).abs().max().item()
    k_diff = (new_K_py - new_K_ref).abs().max().item()
    v_diff = (new_V_py - new_V_ref).abs().max().item()
    print(f'Attn output diff: {diff:.6f}')
    print(f'new_K diff: {k_diff:.6f}')
    print(f'new_V diff: {v_diff:.6f}')
    if diff < 1e-4 and k_diff < 1e-4 and v_diff < 1e-4:
        print('PASS: HLSL attn port matches PyTorch reference')
    else:
        print(f'FAIL: diff > 1e-4')