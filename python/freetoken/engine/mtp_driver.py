"""MTP (Multi-Token Prediction) speculative decoding driver.

End-to-end MTP loop using the MTP head + cache_req_to_len for rollback.
This is the MTP integration MVP: it works as a standalone driver and can be
hooked into the engine's decode loop by replacing the sample step.
"""
from __future__ import annotations
import time
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional

from freetoken.models.qwen3_5_moe.mtp import MtpHeadConfig, load_mtp_head_from_safetensors
from freetoken.kernel.igpu_fc import IgpuFcClient, IgpuFcSticky


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

    def __init__(self, engine, model_path: str, k: int = 3, use_igpu_fc: bool = True):
        self.engine = engine
        self.k = k
        self.device = engine.device
        cfg_src = engine.model
        mc = engine.config.model_config
        # text_config might be nested (VL models)
        tc = getattr(mc, "text_config", mc)
        mtp_cfg = MtpHeadConfig(
            hidden_size=tc.hidden_size,
            vocab_size=tc.vocab_size,
            num_experts=tc.num_experts,
            num_experts_per_tok=tc.num_experts_per_tok,
            moe_intermediate=tc.moe_intermediate_size,
            shared_expert_intermediate=tc.shared_expert_intermediate_size,
            head_dim=tc.head_dim,
            num_qo_heads=tc.num_attention_heads,
            num_kv_heads=tc.num_key_value_heads,
            partial_rotary_factor=tc.partial_rotary_factor,
            rms_norm_eps=tc.rms_norm_eps,
        )
        embed = cfg_src.model.embed_tokens if hasattr(cfg_src.model, "embed_tokens") else None
        lm_head = cfg_src.lm_head if hasattr(cfg_src, "lm_head") else None
        if embed is None or lm_head is None:
            raise RuntimeError("Could not extract embed / lm_head from engine model")
        # Load MTP head
        self.head = load_mtp_head_from_safetensors(
            model_path, mtp_cfg, embed, lm_head, igpu_fc=None, device=self.device, dtype=torch.bfloat16,
        )
        self.head.eval()
        # iGPU FC client
        self._igpu_client = None
        if use_igpu_fc:
            try:
                self._igpu_client = IgpuFcClient()
                if self.head._packed_mxfp4["fc.weight"] is not None:
                    fc_packed = self.head._packed_mxfp4["fc.weight"][0:1].cpu().numpy().astype("uint32")
                    # NVFP4 scales (fc.scales): stored as float32 already in safetensors?
                    # Check dtype — model stores as float32 (NVFP4 fp16 stored as float32 for GPU)
                    fc_scales_t = self.head._packed_mxfp4["fc.scales"]
                    fc_biases_t = self.head._packed_mxfp4["fc.biases"]
                    if fc_scales_t is not None:
                        # fc.scales shape: (M, ns) -- M=1, ns=K//32=128
                        fc_scales = fc_scales_t[0:1].cpu().numpy().astype("float32")
                    else:
                        fc_scales = None
                    if fc_biases_t is not None:
                        fc_biases = fc_biases_t[0:1].cpu().numpy().astype("float32")
                    else:
                        fc_biases = None
                    sticky = IgpuFcSticky(self._igpu_client, fc_packed, 4096,
                                          scales_f32=fc_scales, biases_f32=fc_biases)
                    self.head.igpu_fc = self._make_adapter(sticky)
            except FileNotFoundError:
                self._igpu_client = None
        self.cfg = mtp_cfg

    @staticmethod
    def _make_adapter(sticky):
        class _A:
            def __init__(self, s): self.s = s
            def __call__(self, act_flat):
                if act_flat.requires_grad:
                    act_flat = act_flat.detach()
                if act_flat.dtype == torch.bfloat16:
                    act_flat = act_flat.to(torch.float32)
                act_np = act_flat.cpu().numpy().astype("float32")
                outv = self.s(act_np)
                return torch.from_numpy(outv.copy()).to("cuda").to(torch.bfloat16)
        return _A(sticky)

    @torch.inference_mode()
    def draft(self, prev_token_id: int, prev_hidden: torch.Tensor) -> tuple[list[int], list[torch.Tensor]]:
        """Run the MTP head K times autoregressively, producing K draft token ids.

        prev_token_id: int, the last *committed* token id (the seed for the first draft).
        prev_hidden: [1, H] bf16 on self.device, the main-model hidden state at the
        last committed position. For a real integration this is captured from
        engine.model; for the demo, the caller supplies it.

        Returns (draft_ids, last_hidden_per_step).
        last_hidden_per_step[i] is the MTP head's last hidden state after producing
        draft_ids[i]; useful if the caller wants to seed the next MTP step from the
        MTP head's own last layer output (cleaner than reusing prev_hidden).
        """
        draft_ids: list[int] = []
        cur_token = int(prev_token_id)
        cur_hidden = prev_hidden
        for _ in range(self.k):
            tok_t = torch.tensor([cur_token], device=self.device, dtype=torch.long)
            logits = self.head(tok_t, cur_hidden)
            next_id = int(logits[0].argmax().item())
            draft_ids.append(next_id)
            cur_token = next_id
            cur_hidden = logits[0:1, :].detach()  # use the head's last hidden for the next MTP step
        return draft_ids, [cur_hidden]

    @torch.inference_mode()
    def verify_greedy(self, input_ids: torch.Tensor) -> list[int]:
        """Run the main model on the candidate input_ids and return greedy argmax per
        position. input_ids: [1, 1 + k] int64. Returns list of [1+k] int tokens.

        For the MVP we rely on the engine's batched forward_batch with bs=1.
        """
        from freetoken.core import Batch
        # Build a minimal batch with one request whose input_ids == the K+1 candidates.
        # This is a synchronous, single-request forward; the production path will
        # pre-allocate the request and reuse its KV cache + page table.
        with torch.inference_mode():
            logits = self.engine.model.model(input_ids.to(self.device))
            # Only the last position is the "verify next" — the earlier ones are
            # already-known drafts. For the MTP accept rule we want the model's
            # prediction at every position to compare with the draft.
            # logits: [1, 1+k, vocab]
            preds = logits[0].argmax(dim=-1)  # [1+k]
        return [int(x) for x in preds.cpu().tolist()]

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

    def step_speculative(self, prev_token_id: int, prev_hidden: torch.Tensor) -> tuple[list[int], int]:
        """One MTP step: returns (draft_ids, n_accepted).

        The caller is expected to:
          1. Run the main model forward on the K+1 candidate tokens
          2. Compare drafts vs main model argmax at each draft position
          3. Call self.commit_rollback(cache, req, drafts, verify_ids) to commit/rollback
        """
        return self.draft(prev_token_id, prev_hidden)

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
        if self._igpu_client is not None:
            act = np.random.randn(4096).astype("float32")
            self.head.igpu_fc(act)
