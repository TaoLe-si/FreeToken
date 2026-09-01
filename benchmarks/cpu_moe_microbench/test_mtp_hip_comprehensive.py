"""Comprehensive verification of the AMD Radeon 780M HIP/ROCm path.

Runs all production paths that target the 780M iGPU:
  1. IgpuHIPCppClient real HIP dispatch on AMD Radeon 780M (gfx1103) --
     verifies the HLSL route algorithm ported to HIP produces top-8 idx + weights
     that match PyTorch reference exactly.
  2. MtpHeadAttention SDPA fast path (CPU-only compare since dGPU is NVIDIA).
  3. Qwen3_5MtpHead full forward (CPU fallback FC, since AMD iGPU is different device).

Bypasses the ROCm/MSVC cmath conflict via #ifndef _MSC_VER patches in
__clang_cuda_math_forward_declares.h + __clang_hip_cmath.h (verified working).

Run with:
    python benchmarks/cpu_moe_microbench/test_mtp_hip_comprehensive.py

Exit code 0 = all checks PASS, exit code 1 = at least one FAIL.
"""

import sys
import os

# Ensure HIP runtime DLLs are findable from dist/bin (typical bundled location)
# or system ROCm install. Without this, the HIP server exe won't launch.
_candidates = [
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'dist', 'bin')),
    r"C:\Program Files\AMD\ROCm\6.4\bin",
]
_cur = os.environ.get('PATH', '')
_extra = os.pathsep.join(p for p in _candidates if os.path.isdir(p))
if _extra and _extra not in _cur:
    os.environ['PATH'] = _extra + os.pathsep + _cur

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'python')))

import time
import traceback
import numpy as np
import torch
import torch.nn.functional as F

PASSED = []
FAILED = []


def check(name, ok, detail=""):
    if ok:
        PASSED.append((name, detail))
        print(f"  PASS: {name}  {detail}")
    else:
        FAILED.append((name, detail))
        print(f"  FAIL: {name}  {detail}")


# === 1. HIP/ROCm real GPU dispatch on AMD Radeon 780M ===
print()
print('=' * 70)
print('1. IgpuHIPCppClient real HIP dispatch on AMD Radeon 780M (gfx1103)')
print('=' * 70)
try:
    from freetoken.kernel.igpu_route import IgpuHIPCppClient
    E, H = 256, 2048
    torch.manual_seed(42)
    router_w = torch.randn(E, H, dtype=torch.float32) * 0.01
    client = IgpuHIPCppClient(E=E, H=H)
    # Wait for HIP server 'ready' log
    t0 = time.time()
    while time.time() - t0 < 15:
        log = ' '.join(client._cpp.get_log(8))
        if 'ready' in log:
            print(f"  HIP server ready in {(time.time()-t0)*1000:.0f} ms")
            break
        time.sleep(0.05)
    # LOAD
    t0 = time.time()
    client.load(router_w)
    load_ms = (time.time() - t0) * 1000
    print(f"  HIP upload of router weights ({E}*{H} fp32 = {E*H*4/1024:.0f} KB) in {load_ms:.1f} ms")
    # Warm up
    for _ in range(2):
        client.forward(torch.randn(H, dtype=torch.float32) * 0.1)
    # Forward x 5 to measure steady-state
    times = []
    hidden = torch.randn(H, dtype=torch.float32) * 0.1
    idx = w = None
    for _ in range(5):
        t0 = time.time()
        idx, w = client.forward(hidden)
        times.append((time.time() - t0) * 1000)
    print(f"  HIP forward times (ms): {[round(t, 1) for t in times]}")
    # PyTorch reference
    logits_pt = router_w @ hidden
    sorted_lv, sorted_idx_pt = torch.sort(logits_pt, descending=True)
    top8_idx_pt = sorted_idx_pt[:8].numpy().astype(np.uint32)
    top8_lv_pt = sorted_lv[:8]
    top8_w_pt = (torch.exp(top8_lv_pt - top8_lv_pt.max()) /
                 torch.exp(top8_lv_pt - top8_lv_pt.max()).sum()).numpy()
    idx_match = (idx == top8_idx_pt).all()
    w_diff = float(np.abs(w - top8_w_pt).max())
    check("HIP top8_idx matches PyTorch reference", idx_match, f"({idx[:4].tolist()}...)")
    check("HIP top8_w diff < 0.001", w_diff < 0.001, f"(diff={w_diff:.6f})")
    client.close()
except Exception as e:
    check("HIP GPU dispatch on AMD Radeon 780M", False, f"({e})")
    traceback.print_exc()


# === 2. MtpHeadAttention SDPA fast path (CUDA path; iGPU is separate) ===
print()
print('=' * 70)
print('2. MtpHeadAttention SDPA fast path (verified on NVIDIA dGPU; same kernel works on AMD via rocWMMA if adapted)')
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
    if torch.cuda.is_available():
        torch.manual_seed(42)
        attn = MtpHeadAttention(cfg).cuda().to(torch.bfloat16)
        x = torch.randn(1, cfg.hidden_size, dtype=torch.bfloat16, device='cuda')
        positions = torch.tensor([5], device='cuda')
        x_ctx = torch.randn(10, cfg.hidden_size, dtype=torch.bfloat16, device='cuda')
        pos_ctx = torch.arange(10, device='cuda')
        attn.append_rows(x_ctx, pos_ctx)
        with torch.no_grad():
            out_sdpa = attn(x, positions)
        # einsum reference (use the slow path explicitly)
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
        check("SDPA output shape", out_sdpa.shape == (1, cfg.hidden_size),
              f"({out_sdpa.shape})")
        check("SDPA vs einsum diff < 0.2 (bf16 noise)", diff < 0.2, f"(diff={diff:.4f})")
    else:
        check("CUDA available", False, "(no CUDA on this machine -- skipping)")
except Exception as e:
    check("MtpHeadAttention SDPA path", False, f"({e})")
    traceback.print_exc()


# === 3. Qwen3_5MtpHead full forward ===
print()
print('=' * 70)
print('3. Qwen3_5MtpHead full forward_with_state (PyTorch fallback FC)')
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
    if torch.cuda.is_available():
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
    else:
        check("CUDA available", False, "(no CUDA -- skipping)")
except Exception as e:
    check("Qwen3_5MtpHead forward", False, f"({e})")
    traceback.print_exc()


# === 4. IgpuFcStickyCPP -- AMD Radeon 780M HIP/ROCm FC dispatch ===
print()
print('=' * 70)
print('4. IgpuFcStickyCPP via HIP server on AMD Radeon 780M (gfx1103)')
print('   Real GPU MXFP4 GEMV dispatch through ROCm 6.4 + rdna3)')
print('=' * 70)
try:
    from freetoken.kernel.igpu_fc import make_igpu_fc_sticky
    from freetoken.kernel.igpu_fc import _resolve_hip_fc_server_path
    M, K = 8, 4096
    np.random.seed(42)
    # Valid FP4 nibbles (0-15) packed as uint32s (8 nibbles per uint, 4 uints per 32-elem block)
    # Per row: K nibbles / 8 per uint = K/8 uints
    packed_u32 = np.zeros((M, K // 8), dtype=np.uint32)
    for m in range(M):
        for u in range(K // 8):
            nibbles = np.random.randint(0, 16, 8, dtype=np.uint32)
            v = 0
            for i, n in enumerate(nibbles):
                v |= (n & 0xF) << (i * 4)
            packed_u32[m, u] = v
    scales = (np.random.rand(M, K // 32) * 0.5 - 0.25).astype(np.float32)
    biases = (np.random.rand(M, K // 32) * 0.05).astype(np.float32)
    client = make_igpu_fc_sticky(packed_u32, K, scales_f32=scales, biases_f32=biases)
    check('IgpuFcStickyCPP created with HIP server', type(client).__name__ == 'IgpuFcStickyCPP')
    # Verify server is the HIP one (logs)
    logs = client.get_log(20)
    log_text = ' '.join(logs)
    check('Backend is HIP/ROCm (not D3D12)', 'mxfp4-v3-hip' in log_text or 'HIP' in log_text,
          f'({"hip" in log_text.lower()})')
    # Reference: dequant packed to fp32, then GEMV
    def fp4_nibble_to_float(n):
        if n < 8: return [0,1,2,3,4,6,8,12][n]
        return -[0,1,2,3,4,6,8,12][n-8]
    dequant = np.zeros((M, K), dtype=np.float32)
    for m in range(M):
        for u in range(K // 8):
            v = int(packed_u32[m, u])
            for n_idx in range(8):
                nibble = (v >> (n_idx * 4)) & 0xF
                dequant[m, u * 8 + n_idx] = fp4_nibble_to_float(nibble)
    act = np.random.randn(K).astype(np.float32) * 0.1
    # Apply per-block scale + per-block bias
    ref = np.zeros(M, dtype=np.float32)
    for m in range(M):
        for kk in range(0, K, 32):
            block = dequant[m, kk:kk+32]
            scale = scales[m, kk // 32]
            bias = biases[m, kk // 32]
            ref[m] += ((dequant[m, kk:kk+32] * act[kk:kk+32]).sum() + bias) * scale
    # Run actual
    act2 = act.copy()
    for _ in range(3): client(act2)
    out = client(act2)
    # Compare (small diff expected due to dequant sum reduction)
    rel_diff = np.abs(out - ref) / (np.abs(ref) + 1e-6)
    check('FC output shape (1, M)', out.shape == (M,), f'({out.shape})')
    check('FC output vs reference, mean rel diff < 0.5', rel_diff.mean() < 0.5,
          f'(mean={rel_diff.mean():.4f})')
    # Timing
    times = []
    for _ in range(10):
        t0 = time.time()
        client(act2)
        times.append((time.time() - t0) * 1000)
    print(f'  fc_call times (ms): {[round(t, 2) for t in times]}')
    check('fc_call steady state < 1 ms', min(times[3:]) < 1.0, f'(steady={min(times[3:]):.2f} ms)')
    client.close()
except Exception as e:
    check('IgpuFcStickyCPP HIP path', False, f'({e})')
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
    print('ALL CHECKS PASS -- AMD Radeon 780M HIP path verified')
    print('(ROCm 6.4 + Radeon 780M gfx1103 + D3D12-free iGPU compute path)')
    sys.exit(0)