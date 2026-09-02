"""MTP (Multi-Token Prediction) speculative decoding driver.

End-to-end MTP loop using the MTP head + cache_req_to_len for rollback.
This is the MTP integration MVP: it works as a standalone driver and can be
hooked into the engine's decode loop by replacing the sample step.
"""
from __future__ import annotations
import os
import time
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional

from freetoken.models.qwen3_5_moe.mtp import MtpHeadConfig, load_mtp_head_from_safetensors
from freetoken.kernel.igpu_fc import IgpuFcSticky, make_igpu_fc_sticky


class MtpDriver:
    """Loads the MTP head and exposes draft() / verify() / commit() / rollback().

    Design:
      - One MTP head instance per Engine.
      - draft(prev_token_id, prev_hidden, k): produce K draft token ids.
        Each draft token autoregressively uses the MTP head; prev_hidden for step i+1
        is the MTP head's last hidden state.
      - verify(prefix_input_ids): run the main model on the prefix; returns the
        main-model argmax for each position.
      - commit(req, n): advance req.cached_len / device_len by n accepted tokens.
      - rollback(req, n): call cache_req_to_len to retreat cached_len by n.
    """

    def __init__(self, engine, model_path: str, k: int = 3, use_igpu_fc: bool = False, fc_backend: str = "dgpu"):
        self.engine = engine
        self.k = k
        self.device = engine.device
        cfg_src = engine.model
        mc = engine.config.model_config
        # mc is the parsed ModelConfig (flat names: num_qo_heads / num_kv_heads /
        # rotary_config). HF-style configs (nested text_config, num_attention_heads)
        # are accepted via getattr fallbacks.
        tc = getattr(mc, "text_config", mc)
        rotary = getattr(tc, "rotary_config", None)
        head_dim = getattr(tc, "head_dim", 0) or getattr(rotary, "head_dim", 0) or 0
        partial = getattr(tc, "partial_rotary_factor", None)
        if partial is None and rotary is not None and getattr(rotary, "head_dim", 0):
            partial = rotary.rotary_dim / rotary.head_dim
        mtp_cfg = MtpHeadConfig(
            hidden_size=tc.hidden_size,
            vocab_size=tc.vocab_size,
            num_experts=tc.num_experts,
            num_experts_per_tok=tc.num_experts_per_tok,
            moe_intermediate=tc.moe_intermediate_size,
            shared_expert_intermediate=tc.shared_expert_intermediate_size,
            head_dim=head_dim,
            num_qo_heads=(getattr(tc, "num_qo_heads", 0)
                          or getattr(tc, "num_attention_heads", 0) or 16),
            num_kv_heads=(getattr(tc, "num_kv_heads", 0)
                          or getattr(tc, "num_key_value_heads", 0) or 2),
            partial_rotary_factor=partial if partial else 1.0,
            rms_norm_eps=tc.rms_norm_eps,
            rope_base=(getattr(rotary, "base", 0) or 10000.0),
            norm_topk_prob=getattr(tc, "norm_topk_prob", True),
        )
        embed = cfg_src.model.embed_tokens if hasattr(cfg_src.model, "embed_tokens") else None
        lm_head = cfg_src.lm_head if hasattr(cfg_src, "lm_head") else None
        if embed is None or lm_head is None:
            raise RuntimeError("Could not extract embed / lm_head from engine model")
        # Load MTP head. P2.1 (2026-09-02): the loader picks the FC executor based on
        # fc_backend -- "dgpu" (default) installs DgpuBf16Fc (bf16 F.linear, 14x faster than
        # iGPU IPC), "igpu" installs IgpuFcStickyCPP, "torch" installs TorchNvfp4Fc.
        # When use_igpu_fc=True we still call the loader with fc_backend="dgpu" first so
        # the head is functional even if the iGPU sticky init times out / fails below.
        self.head = load_mtp_head_from_safetensors(
            model_path, mtp_cfg, embed, lm_head, igpu_fc=None, device=self.device,
            dtype=torch.bfloat16, fc_backend=fc_backend,
        )
        self.head.eval()
        import os as _os_prof
        if _os_prof.environ.get("FT_MTP_PROF") == "1":
            self.head._perf = {"fc": 0.0, "fc_n": 0, "attn": 0.0, "moe": 0.0, "lmh": 0.0, "steps": 0}
        # Head-KV cache ownership: the single head instance serves one request's
        # persistent context rows at a time (uid guard; see seed_context).
        self._cache_owner_uid = None
        self._round_base_kv = 0
        # iGPU FC (sticky: full (M, K//8) weight matrix uploaded once via FC_LOAD).
        # The whole attempt runs under a watchdog: under mp.spawn on Windows the D3D12
        # server subprocess can wedge (the blocking pipe read never returns), which used
        # to stall Scheduler.__init__ forever. 45s without completion -> abandon the
        # sticky (the loader already installed the TorchNvfp4Fc fallback) and move on.
        self._igpu_sticky = None
        if use_igpu_fc:
            import threading as _threading
            result: dict = {}
            def _try_sticky() -> None:
                try:
                    fc_packed_t = self.head._packed_mxfp4.get("fc.weight")
                    if fc_packed_t is None:
                        result["skip"] = True
                        return
                    fc_packed = fc_packed_t.cpu().numpy().astype("uint32")  # (M, K//8)
                    M_fc, nb_fc = fc_packed.shape
                    K_fc = nb_fc * 8
                    ns_fc = K_fc // 32
                    fc_scales_t = self.head._packed_mxfp4.get("fc.scales")
                    fc_biases_t = self.head._packed_mxfp4.get("fc.biases")
                    fc_scales = (fc_scales_t.cpu().numpy().astype("float32")
                                 if fc_scales_t is not None
                                 else np.zeros((M_fc, ns_fc), dtype="float32"))
                    fc_biases = (fc_biases_t.cpu().numpy().astype("float32")
                                 if fc_biases_t is not None
                                 else np.zeros((M_fc, ns_fc), dtype="float32"))
                    # Phase 2.5 (ROCm 6.4, 2026-08-30): prefer C++ IgpuFcStickyCPP
                    # with HIP server on AMD Radeon 780M; falls back to D3D12 / Python if
                    # the .pyd is unavailable.
                    sticky = make_igpu_fc_sticky(fc_packed, K_fc,
                                                 scales_f32=fc_scales, biases_f32=fc_biases)
                    result["sticky"] = sticky
                except Exception as exc:  # noqa: BLE001 — any failure falls back to torch fc
                    result["err"] = repr(exc)
            th = _threading.Thread(target=_try_sticky, name="ft-mtp-igpu-init", daemon=True)
            th.start()
            th.join(timeout=45.0)
            if "sticky" in result:
                self._igpu_sticky = result["sticky"]
                self.head.igpu_fc = self._igpu_sticky.torch()  # device-adaptive bridge
                # Init-time iGPU FC warmup (same watchdog thread, same 45s budget):
                # run one zeros activation through the sticky so the HIP server
                # pipeline is hot before the first request. Never fatal.
                try:
                    self.warmup()
                except Exception as exc:  # noqa: BLE001 -- warmup must never block serve startup
                    print(f"[MTP] iGPU FC warmup failed (non-fatal): {exc!r}", flush=True)
            elif "err" in result:
                print(f"[MTP] iGPU FC sticky unavailable ({result['err']}); using torch fc", flush=True)
            elif th.is_alive():
                print("[MTP] iGPU FC sticky init timed out (>45s); using torch fc", flush=True)
        self.cfg = mtp_cfg

    def close(self):
        """Shut down the iGPU FC sticky server process."""
        if self._igpu_sticky is not None:
            self._igpu_sticky.close()
            self._igpu_sticky = None
            self.head.igpu_fc = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    @torch.inference_mode()
    def seed_context(self, uid: int, tokens: torch.Tensor, hiddens: torch.Tensor,
                      start_pos: int) -> None:
        """Seed the head's KV cache with the committed context of request uid.

        tokens[i] (rope position start_pos+1+i) pairs with the main-model hidden
        hiddens[i]. Claims cache ownership: a different owner's rows are dropped
        first (single head instance -- concurrent greedy requests fall back to
        window-only drafts for the interleaved one)."""
        if uid != self._cache_owner_uid:
            self.head.attn.reset_draft_cache()
            self._cache_owner_uid = uid
        self.head.extend_context(tokens, hiddens, start_pos)

    def draft(self, uid: int, prev_token_id: int, prev_hidden: torch.Tensor,
              position: int) -> list[int]:
        """Run the MTP head K times autoregressively, producing K draft token ids.

        uid: the owning request's uid (head KV cache ownership guard).
        prev_token_id: int -- the last *committed* token id (the seed for d1).
        prev_hidden: [1, H] bf16 on self.device -- the main-model post-norm hidden
            of the position that sampled prev_token_id.
        position: int -- the rope position of prev_token_id; step i runs at
            position + i.

        Returns [d1, ..., dK] (greedy argmax per step). The head's KV cache is
        PERSISTENT across rounds: it holds one row per committed token (seeded
        exactly + re-committed from verify hiddens), so each draft step attends
        over the full trained-on context. Steps 2..K self-feed the head's own
        state (the drafted tokens have no main hidden yet -- the verify forward
        computes those)."""
        if uid != self._cache_owner_uid:
            # Cache stolen by another request mid-flight: reset and degrade to
            # window-only drafts for this round (correct, just less accurate).
            self.head.attn.reset_draft_cache()
            self._cache_owner_uid = uid
        # FT_MTP_HEAD_NOCACHE=1: window-only drafts (exporter-validated mode --
        # vmlx_mtp_tuning.json "cache_mode": "off", speedup 1.564x). The head
        # attends only within the current round's rows; the persistent-KV seeding
        # path is bypassed entirely.
        if os.environ.get("FT_MTP_HEAD_NOCACHE") == "1":
            self.head.attn.reset_draft_cache()
        # Rows present before this round's step-1 row (the committed prefix).
        # Invariant: rows 0..position-1 must be exactly the committed prefix.
        # Stale rows beyond that (left by the previous round commit/draft
        # interplay) would duplicate positions and skew the head attention --
        # truncate to the committed prefix; step 1 then writes row `position`.
        _kv_now = self.head.attn.kv_len()
        if _kv_now > position:
            self.head.attn.truncate_kv(position)
        self._round_base_kv = self.head.attn.kv_len()
        if os.environ.get("FT_MTP_DEBUG"):
            print(f"[MTP-dbg] draft: kv_len={self._round_base_kv} position={position} "
                  f"prev_tok={int(prev_token_id)} h_norm={float(prev_hidden.float().norm()):.3f}",
                  flush=True)
        # Single sync at end: K-1 fewer GPU<->CPU trips than the original per-iter
        # `int(tok_t.item())` loop. Persistent GPU buffers for tok_t and the K-slot
        # draft collector; only one .tolist() sync at the very end.
        dev = self.device
        if not hasattr(self, "_draft_tok_buf") or self._draft_tok_buf.device != dev:
            self._draft_tok_buf = torch.empty(1, dtype=torch.long, device=dev)
            self._drafts_buf = torch.empty(self.k, dtype=torch.long, device=dev)
            self._seed_tok_buf = torch.empty(1, dtype=torch.long, device=dev)
        # Seed the first iter's input (no .tensor() alloc per call).
        self._seed_tok_buf.fill_(int(prev_token_id))
        tok_t = self._seed_tok_buf
        h = prev_hidden.view(1, -1)
        head_fwd = self.head.forward_with_state
        drafts_buf = self._drafts_buf
        argmax = torch.argmax
        for i in range(self.k):
            logits, h = head_fwd(tok_t, h, position)
            position += 1
            # argmax returns int64; write into the persistent slot. No per-iter alloc.
            argmax(logits, dim=-1, out=drafts_buf[i:i+1])
            # Next iter's input = this iter's argmax slot (1-elem view, no alloc).
            tok_t = drafts_buf[i:i+1] if i + 1 < self.k else tok_t
        # ONE sync. .tolist() on a GPU tensor materialises K ints as a Python list.
        result = drafts_buf.tolist()
        if os.environ.get("FT_MTP_DEBUG"):
            _lg, _h2 = head_fwd(self._seed_tok_buf, prev_hidden.view(1, -1), position)
            _top = torch.topk(_lg[0], 5)
            print(f"[MTP-dbg] head-logits max={float(_lg.max()):.3f} top5={_top.indices.tolist()} "
                  f"scores={[round(float(x), 2) for x in _top.values.tolist()]} h_norm={float(_h2.float().norm()):.3f}",
                  flush=True)
        # P2.1 (2026-09-02): dump cumulative per-step timing if profiling.
        # Lets us see which section of the MTP head is the real bottleneck.
        if os.environ.get("FT_MTP_PROF") == "1":
            perf = getattr(self.head, "_perf", None)
            if perf is not None:
                self._round_count = getattr(self, "_round_count", 0) + 1
                if self._round_count <= 3 or self._round_count % 20 == 0:
                    steps = max(1, perf.get("fc_n", 0) // max(1, self._round_count))
                    fc_avg = perf["fc"] / max(1, perf["fc_n"]) * 1e6
                    attn_avg = perf["attn"] / max(1, self._round_count) * 1e6
                    moe_avg = perf["moe"] / max(1, self._round_count) * 1e6
                    lmh_avg = perf["lmh"] / max(1, self._round_count) * 1e6
                    fc_obj = self.head.igpu_fc
                    # Unwrap _TorchFc to show real executor
                    underlying = type(fc_obj).__name__
                    if underlying == "_TorchFc":
                        underlying = type(fc_obj.sticky).__name__ if hasattr(fc_obj, "sticky") else underlying
                    print(f"[MTP-prof] round={self._round_count} fc={underlying} "
                          f"fc_avg={fc_avg:.1f}us attn_avg={attn_avg:.1f}us "
                          f"moe_avg={moe_avg:.1f}us lmh_avg={lmh_avg:.1f}us "
                          f"steps/round={steps}", flush=True)
        return result

    @torch.inference_mode()
    def reconcile(self, uid: int, tokens: torch.Tensor, hiddens: torch.Tensor,
                  start_pos: int) -> None:
        """Append exact head-KV rows for positions start_pos+1 .. start_pos+tokens.numel().
        Used by the scheduler to backfill the head cache so it always mirrors the
        main model committed tokens one-to-one (row j pairs (x_j, h_{j-1})).
        No ownership change: caller guarantees uid is the cache owner."""
        if uid != self._cache_owner_uid:
            return
        self.head.extend_context(tokens, hiddens, start_pos)
    def commit_round(self, uid: int, tokens: torch.Tensor | None,
                     hiddens: torch.Tensor | None) -> None:
        """Reconcile the head KV cache after a verify round for request uid.

        The K drafted rows (self-fed approximations) are rolled back to the
        committed prefix: keep the round's step-1 row (exact -- the seed token
        paired with a true main hidden), then append exact rows for the newly
        committed tokens: tokens[i] (accepted drafts + correction, in order)
        paired with hiddens[i] (the verify forward's per-position hiddens).
        tokens=None rolls back only (all drafts rejected)."""
        if uid != self._cache_owner_uid:
            return  # cache stolen mid-round; nothing to reconcile
        base = getattr(self, "_round_base_kv", 0)
        keep = base + 1  # + the seed token's own (exact) row
        self.head.attn.truncate_kv(keep)
        if tokens is not None and tokens.numel() > 0 and hiddens is not None:
            self.head.extend_context(tokens, hiddens, start_pos=keep)

    def accept_count(self, draft_ids: list[int], verify_ids: list[int], base: int) -> int:
        """How many drafts to accept. base is the index of the first draft in
        verify_ids (typically base=1 because verify_ids[0] should match prev_token_id).
        """
        n = 0
        for i, did in enumerate(draft_ids):
            v = verify_ids[base + i] if (base + i) < len(verify_ids) else None
            if v is None or v != did:
                break
            n += 1
        return n

    def commit_to_len(self, cache, req, new_cached_len: int) -> None:
        """Advance the request's KV cache to new_cached_len (hand-back the orphans)."""
        cache.cache_req_to_len(req, new_cached_len)

    def rollback(self, cache, req, n_accepted: int) -> None:
        """Retreat cached_len by k - n_accepted tokens (rollback rejected drafts)."""
        target = req.cached_len - (self.k - n_accepted)
        cache.cache_req_to_len(req, target)

    def commit_rollback(self, cache, req, draft_ids: list[int], verify_ids: list[int],
                         base: int = 1) -> int:
        """Compare drafts vs verify, commit accepted, rollback rejected.

        base is the index in verify_ids where drafts start (typically 1 because
        verify_ids[0] is the prev_token_id prediction).
        Returns the number of accepted drafts (0..K+1).
        """
        n_accept = self.accept_count(draft_ids, verify_ids, base)
        target = req.cached_len + 1 + n_accept  # 1 for prev_token, +n_accept drafts
        self.commit_to_len(cache, req, target)
        if n_accept < self.k:
            # Roll back the rejected drafts (the orphans above target)
            self.rollback(cache, req, n_accept)
        return n_accept

    def warmup(self):
        if self._igpu_sticky is not None:
            act = torch.zeros(self._igpu_sticky.K, dtype=torch.float32)
            self.head.igpu_fc(act)
