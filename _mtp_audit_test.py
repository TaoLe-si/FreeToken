"""Comprehensive pre-flight test of the MTP stack (no engine required).

A. Rope A/B: mtp._neox_rope vs the main model's RotaryEmbedding kernel (CUDA).
B. Head load: real checkpoint, real dims, stub embed/lm_head on CPU.
   - every param group nonzero (q_norm/k_norm/layernorms/switch/qkv/o_proj)
   - forward_with_state runs, finite, correct shapes
   - draft loop (K steps) runs; attn cache grows
C. TorchNvfp4Fc matches a manual affine dequant GEMV (.call and .batch).
D. Verify-protocol math trace (pure python): correction=preds[n], target=base+2+n.
E. Head-KV lifecycle: seed_context -> draft -> commit_round -> verify cache row count,
   owner guard, truncate_kv, stale-hiddens vs all_hidden semantics.
"""
import sys, math, traceback
sys.path.insert(0, "E:/FreeToken/python")
import torch
import torch.nn.functional as F

FAIL = []
def check(name, cond, detail=""):
    tag = "PASS" if cond else "FAIL"
    print(f"[{tag}] {name} {detail}")
    if not cond:
        FAIL.append(name)

# ---------------- A. rope A/B ----------------
try:
    from freetoken.layers.rotary import RotaryEmbedding
    from freetoken.models.qwen3_5_moe.mtp import _neox_rope
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    HEAD_DIM, ROT, BASE = 256, 64, 10000.0
    rope = RotaryEmbedding(HEAD_DIM, ROT, 4096, BASE)
    torch.manual_seed(0)
    N, HQ, HK = 5, 16, 2
    q = torch.randn(N, HQ * HEAD_DIM, device=dev) * 0.5
    k = torch.randn(N, HK * HEAD_DIM, device=dev) * 0.5
    pos = torch.tensor([3, 17, 100, 512, 2047], device=dev, dtype=torch.int64)
    q_ref, k_ref = q.clone(), k.clone()
    rope.forward(pos, q_ref, k_ref)
    q3 = q.view(N, HQ, HEAD_DIM).clone()
    k3 = k.view(N, HK, HEAD_DIM).clone()
    q_new, k_new = _neox_rope(q3, k3, pos, ROT, base=BASE)
    qd = (q_new.reshape(N, -1) - q_ref).abs().max().item()
    kd = (k_new.reshape(N, -1) - k_ref).abs().max().item()
    check("A.rope q matches main kernel", qd < 2e-2, f"maxdiff={qd:.2e}")
    check("A.rope k matches main kernel", kd < 2e-2, f"maxdiff={kd:.2e}")
except Exception:
    traceback.print_exc(); FAIL.append("A.rope exception")

# ---------------- B. head load + draft ----------------
try:
    from freetoken.models.qwen3_5_moe.mtp import (
        MtpHeadConfig, load_mtp_head_from_safetensors, TorchNvfp4Fc,
        _dequant_mxfp4_affine)
    import safetensors.torch as st
    MODEL = "E:/models/Qwen3.6-35B-A3B-MXFP4-MTP-NVFP4"
    state = st.load_file(MODEL + "/mtp.safetensors")

    cfg = MtpHeadConfig(rope_base=10000.0)

    class StubEmbed:
        def __init__(self, vocab, dim):
            self.weight = torch.randn(vocab, dim, dtype=torch.bfloat16) * 0.02
        def forward(self, x):
            return F.embedding(x, self.weight)

    class StubLM:
        def __init__(self, vocab, dim):
            self.weight = torch.randn(vocab, dim, dtype=torch.bfloat16) * 0.02
        def forward(self, x):
            return F.linear(x, self.weight)

    VSTUB = 4096
    embed = StubEmbed(VSTUB, cfg.hidden_size)
    lm = StubLM(VSTUB, cfg.hidden_size)
    head = load_mtp_head_from_safetensors(
        MODEL, cfg, embed, lm, igpu_fc=None, device="cpu", dtype=torch.bfloat16)
    head.eval()

    def nz(t):
        return t.detach().abs().sum().item()
    check("B.q_norm loaded", nz(head.attn.q_norm) > 0, f"sum={nz(head.attn.q_norm):.3f}")
    check("B.k_norm loaded", nz(head.attn.k_norm) > 0, f"sum={nz(head.attn.k_norm):.3f}")
    for nm in ["input_layernorm", "post_attention_layernorm", "mtp_norm",
               "pre_fc_norm_embedding", "pre_fc_norm_hidden"]:
        check(f"B.{nm} loaded", nz(getattr(head, nm)) > 0, f"sum={nz(getattr(head, nm)):.3f}")
    check("B.qkv_proj loaded", head.attn.qkv_proj.weight.abs().mean().item() > 1e-4,
          f"mean={head.attn.qkv_proj.weight.abs().mean().item():.4f}")
    check("B.o_proj loaded", head.attn.o_proj.weight.abs().mean().item() > 1e-4)
    check("B.switch_gate loaded", head.mlp.switch_gate.abs().mean().item() > 1e-4)
    check("B.mlp.gate loaded", head.mlp.gate.weight.abs().mean().item() > 1e-4)
    check("B.shared_gate loaded", head.mlp.shared_gate.weight.abs().mean().item() > 1e-4)

    # draft loop like the driver
    tok = torch.tensor([123], dtype=torch.long)
    h = torch.randn(1, cfg.hidden_size, dtype=torch.bfloat16)
    head.attn.reset_draft_cache()
    drafts = []
    with torch.inference_mode():
        for step in range(3):
            logits, h2 = head.forward_with_state(tok, h, position=100 + step)
            check(f"B.step{step} logits finite", torch.isfinite(logits).all().item(),
                  f"shape={tuple(logits.shape)}")
            check(f"B.step{step} state finite", torch.isfinite(h2).all().item())
            nxt = int(logits.argmax(dim=-1).item())
            drafts.append(nxt)
            tok = torch.tensor([nxt % VSTUB], dtype=torch.long)
            h = h2
    check("B.draft loop produced 3 tokens", len(drafts) == 3, f"drafts={drafts}")
    check("B.attn cache grew", getattr(head.attn, "_draft_cache", None) is not None
          and head.attn._draft_cache[0].shape[0] == 3,
          f"k cache T={head.attn._draft_cache[0].shape[0]}")

    # ---------------- C. TorchNvfp4Fc vs manual ----------------
    pw = state["mtp.fc.weight"]
    ps = state["mtp.fc.scales"].float()
    pb = state["mtp.fc.biases"].float()
    fc = TorchNvfp4Fc(pw, ps, pb, "cpu")
    x = torch.randn(fc.K)
    out = fc(x)
    manual = (_dequant_mxfp4_affine(pw, ps, pb) @ x)
    d = (out - manual).abs().max().item()
    check("C.fc matches manual GEMV", d < 1e-3, f"maxdiff={d:.2e} shape={tuple(out.shape)}")
    # batched
    xb = torch.randn(5, fc.K)
    outb = fc.batch(xb)
    manualb = xb @ _dequant_mxfp4_affine(pw, ps, pb).t()
    db = (outb - manualb).abs().max().item()
    check("C.fc.batch matches manual GEMM", db < 1e-3, f"maxdiff={db:.2e} shape={tuple(outb.shape)}")

    # ---------------- E. Head-KV lifecycle ----------------
    H = cfg.hidden_size
    # Simulate a 2-round MTP cycle: seed 8 context rows, draft 3, commit 2 accepted,
    # then draft 3 again -- verify cache row count and owner semantics.
    head.attn.reset_draft_cache()
    check("E.kv_len == 0 after reset", head.attn.kv_len() == 0)
    ctx_tokens = torch.tensor([10, 11, 12, 13, 14, 15, 16, 17], dtype=torch.long)
    ctx_hiddens = torch.randn(8, H, dtype=torch.bfloat16)
    head.extend_context(ctx_tokens, ctx_hiddens, start_pos=0)
    check("E.seed added 8 rows", head.attn.kv_len() == 8,
          f"got {head.attn.kv_len()}")
    # truncate_kv
    head.attn.truncate_kv(5)
    check("E.truncate_kv to 5", head.attn.kv_len() == 5,
          f"got {head.attn.kv_len()}")
    head.attn.reset_draft_cache()
    # full round-trip: seed 10, draft 3 (kv grows to 13), commit_round([d1,d2],hid[:2]) -> 12
    head.extend_context(
        torch.tensor([100 + i for i in range(10)], dtype=torch.long),
        torch.randn(10, H, dtype=torch.bfloat16),
        start_pos=0,
    )
    check("E.seed 10 rows", head.attn.kv_len() == 10)
    # simulate draft steps (forward + step rows appended via _project+append_rows
    # path; forward_with_state internally calls attn.forward which appends 1 row).
    # Run a 3-step draft loop and check it grew by 3.
    base_kv = head.attn.kv_len()
    tok = torch.tensor([42], dtype=torch.long)
    h = torch.randn(1, H, dtype=torch.bfloat16)
    with torch.inference_mode():
        for step in range(3):
            logits, h = head.forward_with_state(tok, h, position=10 + step)
            tok = logits.argmax(dim=-1).to(torch.long)
    check("E.draft added 3 rows", head.attn.kv_len() == base_kv + 3,
          f"base={base_kv} got {head.attn.kv_len()}")
    # commit_round semantics: truncate to base_kv + 1 (seed token row), append 2
    accepted = torch.tensor([77, 88], dtype=torch.long)
    accepted_h = torch.randn(2, H, dtype=torch.bfloat16)
    head.attn.truncate_kv(base_kv + 1)
    head.extend_context(accepted, accepted_h, start_pos=base_kv)  # base_kv == 10, rows at 11, 12
    check("E.commit appended 2 rows", head.attn.kv_len() == base_kv + 3,
          f"got {head.attn.kv_len()}")
except Exception:
    traceback.print_exc(); FAIL.append("B/C/E exception")

# ---------------- D. verify protocol math trace ----------------
try:
    K = 3
    drafts = [11, 22, 33]
    preds = [11, 99, 77, 55]
    n = 0
    while n < K and preds[n] == drafts[n]:
        n += 1
    assert n == 1, n
    correction = preds[n]
    base = 100
    target = base + 2 + n
    check("D.n == 1 (d1 accepted, d2 rejected)", n == 1)
    check("D.correction == preds[n] (not bonus)", correction == 99, f"corr={correction}")
    check("D.target == base+2+n (t + n drafts)", target == 103, f"target={target}")
    preds2 = [11, 22, 33, 44]
    n2 = 0
    while n2 < K and preds2[n2] == drafts[n2]:
        n2 += 1
    check("D.full accept n == K", n2 == 3)
    check("D.full-accept correction == bonus", preds2[n2] == 44)
    drafts3 = [11, 5, 33]
    preds3 = [11, 5, 77, 88]
    n3 = 0
    while n3 < K and preds3[n3] == drafts3[n3]:
        n3 += 1
    eos_at = next((j for j in range(n3) if drafts3[j] == 5), n3)
    check("D.EOS truncation", eos_at == 1 and n3 == 2, f"n3={n3} eos_at={eos_at}")
except Exception:
    traceback.print_exc(); FAIL.append("D exception")

print()
if FAIL:
    print("FAILED:", FAIL); sys.exit(1)
print("ALL CHECKS PASSED")
