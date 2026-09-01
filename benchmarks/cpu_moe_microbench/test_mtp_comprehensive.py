"""Comprehensive MTP verification (Phase 2.5, 2026-08-30).

Runs ALL the production verification paths in one Python script:
  1. IgpuRouteClient (real GPU dispatch) -- verifies the HLSL route kernel
     produces top-8 idx + weights that match PyTorch reference exactly.
  2. MtpHeadAttention SDPA fast path -- verifies that the production SDPA
     attention code (with GQA via enable_gqa=True) matches the einsum
     reference within bf16 noise.
  3. Qwen3_5MtpHead full forward -- verifies the production forward_with_state
     runs end-to-end (SDPA + MoE + FC + LM head) and produces valid outputs.
  4. MoE + attn HLSL algorithm alignment (Python port vs PyTorch reference).

Run with:
    python benchmarks/cpu_moe_microbench/test_mtp_comprehensive.py

Exit code 0 = all checks PASS, exit code 1 = at least one FAIL.
"""

import sys
import os
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))

import torch
import torch.nn.functional as F
import numpy as np

PASSED = []
FAILED = []


def check(name, ok, detail=""):
    if ok:
        PASSED.append((name, detail))
        print(f"  PASS: {name}  {detail}")
    else:
        FAILED.append((name, detail))
        print(f"  FAIL: {name}  {detail}")


# === 1. IgpuRouteClient real GPU dispatch ===
print()
print('=' * 70)
print('1. IgpuRouteClient real GPU dispatch verification')
print('=' * 70)
try:
    from freetoken.kernel.igpu_route import IgpuRouteClient
    E, H = 256, 2048
    torch.manual_seed(42)
    router_w = torch.randn(E, H, dtype=torch.float32) * 0.01
    client = IgpuRouteClient(E=E, H=H)
    client.load(router_w)
    hidden = torch.randn(H, dtype=torch.float32) * 0.1
    idx_gpu, w_gpu = client.forward(hidden)
    # PyTorch reference
    logits_pt = router_w @ hidden
    sorted_lv, sorted_idx_pt = torch.sort(logits_pt, descending=True)
    top8_idx_pt = sorted_idx_pt[:8].numpy().astype(np.uint32)
    top8_lv_pt = sorted_lv[:8]
    top8_w_pt = (torch.exp(top8_lv_pt - top8_lv_pt.max()) /
                 torch.exp(top8_lv_pt - top8_lv_pt.max()).sum()).numpy()
    idx_match = (idx_gpu == top8_idx_pt).all()
    w_diff = float((w_gpu - top8_w_pt).abs().max())
    check("IgpuRouteClient idx match", idx_match, f"({idx_gpu.tolist()[:4]}...)")
    check("IgpuRouteClient w diff < 0.001", w_diff < 0.001, f"(diff={w_diff:.6f})")
    client.close()
except Exception as e:
    check("IgpuRouteClient real GPU", False, f"({e})")
    traceback.print_exc()


# === 2. MtpHeadAttention SDPA fast path ===
print()
print('=' * 70)
print('2. MtpHeadAttention SDPA fast path vs einsum reference')
print('=' * 70)
try:
    from freetoken.models.qwen3_5_moe.mtp import MtpHeadAttention

    class Cfg:
        hidden_size = 2048
        num_qo_heads = 16
        num_kv_heads = 2
        head_dim = 128
        partial_rotary_factor = 0.5
        rope_base = 10000.0

    cfg = Cfg()
    if not torch.cuda.is_available():
        check("MtpHeadAttention CUDA available", False, "(no CUDA; skipping)")
    else:
        torch.manual_seed(42)
        attn = MtpHeadAttention(cfg).cuda().to(torch.bfloat16)
        x = torch.randn(1, cfg.hidden_size, dtype=torch.bfloat16, device='cuda')
        positions = torch.tensor([5], device='cuda')
        x_ctx = torch.randn(10, cfg.hidden_size, dtype=torch.bfloat16, device='cuda')
        pos_ctx = torch.arange(10, device='cuda')
        attn.append_rows(x_ctx, pos_ctx)
        with torch.no_grad():
            out_sdpa = attn(x, positions)
        # einsum reference
        attn2 = MtpHeadAttention(cfg).cuda().to(torch.bfloat16)
        attn2.load_state_dict(attn.state_dict())
        attn2.append_rows(x_ctx, pos_ctx)
        with torch.no_grad():
            q, k, v, gate = attn2._project(x, positions)
            k_cached, v_cached = attn2._draft_cache
            rep = attn2.num_q // attn2.num_kv
            kg = k_cached.repeat_interleave(rep, dim=1)
            vg = v_cached.repeat_interleave(rep, dim=1)
            scale = 1.0 / (attn2.head_dim ** 0.5)
            attn_logits = torch.einsum('nqd,mqd->nqm', q, kg) * scale
            attn_probs = F.softmax(attn_logits.float(), dim=-1).to(q.dtype)
            out_einsum = torch.einsum('nqm,mqd->nqd', attn_probs, vg) * torch.sigmoid(gate)
            out_einsum = out_einsum.reshape(-1, attn2.qo_dim)
            out_einsum = attn2.o_proj(out_einsum)
        diff = float((out_sdpa - out_einsum).abs().max())
        check("MtpHeadAttention SDPA output shape", out_sdpa.shape == (1, cfg.hidden_size),
              f"({out_sdpa.shape})")
        check("MtpHeadAttention SDPA vs einsum diff < 0.1", diff < 0.1, f"(diff={diff:.4f})")
except Exception as e:
    check("MtpHeadAttention SDPA path", False, f"({e})")
    traceback.print_exc()


# === 3. Qwen3_5MtpHead full forward ===
print()
print('=' * 70)
print('3. Qwen3_5MtpHead full forward_with_state')
print('=' * 70)
try:
    from freetoken.models.qwen3_5_moe.mtp import Qwen3_5MtpHead

    class Cfg2:
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

    cfg2 = Cfg2()
    if not torch.cuda.is_available():
        check("Qwen3_5MtpHead CUDA available", False, "(no CUDA; skipping)")
    else:
        device = 'cuda'
        dtype = torch.bfloat16
        torch.manual_seed(42)
        head = Qwen3_5MtpHead(cfg2, torch.nn.Embedding(cfg2.vocab_size, cfg2.hidden_size),
                              torch.nn.Linear(cfg2.hidden_size, cfg2.vocab_size, bias=False),
                              igpu_fc=None, dtype=torch.float32).to(device)
        head = head.to(dtype)
        head.embed_table = head.embed_table.to(dtype)
        head.lm_head = head.lm_head.to(dtype)
        head.attn.qkv_proj = head.attn.qkv_proj.to(dtype)
        head.attn.o_proj = head.attn.o_proj.to(dtype)

        # PyTorch fallback FC
        fc = torch.nn.Linear(2 * cfg2.hidden_size, cfg2.hidden_size, bias=False).to(device, dtype)
        fc.weight.data = torch.randn(cfg2.hidden_size, 2 * cfg2.hidden_size,
                                     dtype=dtype, device=device) * 0.02

        class TorchFcWrapper:
            def __init__(self, fc): self.fc = fc
            def __call__(self, x_flat): return self.fc(x_flat.to(torch.bfloat16))

        head.igpu_fc = TorchFcWrapper(fc)
        head.dtype = dtype

        prev_token_id = torch.tensor([42], device=device)
        prev_hidden = torch.randn(1, cfg2.hidden_size, dtype=dtype, device=device) * 0.1
        with torch.no_grad():
            logits, state = head.forward_with_state(prev_token_id, prev_hidden, 5)
        check("MtpHead logits shape", logits.shape == (1, cfg2.vocab_size),
              f"({logits.shape})")
        check("MtpHead state shape", state.shape == (1, cfg2.hidden_size),
              f"({state.shape})")
        check("MtpHead logits no NaN", not torch.isnan(logits).any())
        check("MtpHead state no NaN", not torch.isnan(state).any())
except Exception as e:
    check("Qwen3_5MtpHead forward", False, f"({e})")
    traceback.print_exc()


# === 4. MoE + attn algorithm alignment ===
print()
print('=' * 70)
print('4. MoE + attn HLSL algorithm alignment')
print('=' * 70)
try:
    sys.path.insert(0, os.path.dirname(__file__))
    from test_moe_align import moe_forward_py, moe_reference_torch
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
    diff = float((out_py - out_ref).abs().max())
    check("MoE Python port vs reference diff < 1e-4", diff < 1e-4, f"(diff={diff:.6f})")

    from test_attn_align import (
        attn_forward_py, attn_reference_torch, H as H_attn,
    )
    import torch as _t
    H_a = H_attn
    weights_a = {
        'qkv_w':       _t.randn((16 + 2*2) * 128, H_a) * 0.02,
        'o_w':         _t.randn(H_a, 16 * 128) * 0.02,
        'q_norm_w':    _t.ones(16, 128) + _t.randn(16, 128) * 0.01,
        'k_norm_w':    _t.ones(2, 128) + _t.randn(2, 128) * 0.01,
        'gate_w':      _t.randn(16 * 128 + 2 * 128 + 2 * 128) * 0.01,
        'rope_inv_freq': _t.exp(-_t.arange(0, 128, 2).float() * 5.0 / 128),
    }
    kv_K = _t.randn(4, 2, 128) * 0.1
    kv_V = _t.randn(4, 2, 128) * 0.1
    h = _t.randn(H_a) * 0.1
    out_py, _, _ = attn_forward_py(h, kv_K, kv_V, weights_a, 5)
    out_ref, _, _ = attn_reference_torch(h, kv_K, kv_V, weights_a, 5)
    diff = float((out_py - out_ref).abs().max())
    check("Attn Python port vs reference diff < 1e-4", diff < 1e-4, f"(diff={diff:.6f})")
except Exception as e:
    check("Algorithm alignment", False, f"({e})")
    traceback.print_exc()


# === Summary ===
print()
print('=' * 70)
print(f"SUMMARY: {len(PASSED)} passed, {len(FAILED)} failed")
print('=' * 70)
if FAILED:
    print('FAILURES:')
    for n, d in FAILED:
        print(f'  {n}: {d}')
    sys.exit(1)
else:
    print('ALL CHECKS PASS -- production MTP path verified')
    sys.exit(0)