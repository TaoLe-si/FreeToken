from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from freetoken.core import get_global_ctx
from freetoken.layers import (
    BaseOP,
    GemmaRMSNorm,
    OPList,
    ParallelLMHead,
    VocabParallelEmbedding,
)
from freetoken.models.blocks import BaseLLMModel
from freetoken.utils import nvtx_annotate

from .attention import Qwen3_5Attention
from .gdn import Qwen3_5GatedDeltaNet
from .moe import Qwen3_5DenseMLP, Qwen3_5MoE

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class Qwen3_5DecoderLayer(BaseOP):
    """Pre-norm hybrid block: ``x = x + mixer(input_norm(x)); x = x + moe(post_norm(x))``,
    where the mixer is a GatedDeltaNet (linear layers) or gated attention (full layers).
    All norms are Gemma-style (1+weight)."""

    def __init__(self, config: ModelConfig, layer_id: int):
        self._layer_id = layer_id
        self._is_linear = config.is_linear_layer(layer_id)
        if self._is_linear:
            g = config.linear_attention_group()
            assert g is not None
            self.linear_attn = Qwen3_5GatedDeltaNet(
                hidden_size=config.hidden_size,
                num_k_heads=g.num_key_heads,
                num_v_heads=g.num_value_heads,
                head_k_dim=g.key_head_dim,
                head_v_dim=g.value_head_dim,
                conv_kernel_size=g.conv_kernel_dim,
                rms_norm_eps=config.rms_norm_eps,
                layer_id=layer_id,
                expert_quant=config.expert_quant,
                attn_quant=config.attn_quant,
            )
        else:
            self.self_attn = Qwen3_5Attention(config, layer_id)
        # Dense variants (num_experts==0, e.g. Qwen3.6-27B) use a plain SwiGLU MLP instead of
        # the routed MoE block; both expose ``forward(hidden)->hidden`` and the same key prefix.
        self.mlp = Qwen3_5MoE(config, layer_id) if config.moe_enabled else Qwen3_5DenseMLP(config, layer_id)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(self, hidden: torch.Tensor, residual: torch.Tensor | None):
        # Residual-stream form: fuse each residual-add into the next RMSNorm
        # (GemmaRMSNorm.forward_add_residual) so add + norm are one kernel per sublayer.
        if residual is None:
            residual = hidden
            hidden = self.input_layernorm.forward(hidden)
        else:
            hidden, residual = self.input_layernorm.forward_add_residual(hidden, residual)
        hidden = self.linear_attn.forward(hidden) if self._is_linear else self.self_attn.forward(hidden)
        hidden, residual = self.post_attention_layernorm.forward_add_residual(hidden, residual)
        hidden = self.mlp.forward(hidden)
        return hidden, residual


class Qwen3_5Model(BaseOP):
    def __init__(self, config: ModelConfig):
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = OPList(
            [Qwen3_5DecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self, input_ids: torch.Tensor, return_raw: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # G.3: if a graph handle is bound on the batch, replay the cached
        # CUDAGraph instead of running the 24-layer loop. The scheduler binds
        # the handle via ModelVerifyGraphBackend.prepare_for_replay. This
        # collapses ~265 kernel launches into a single dispatch (~10 us vs
        # ~2 ms minimum even on the 5090) for the MTP-verify path.
        from freetoken.core import get_global_ctx
        _ctx_for_graph = get_global_ctx()
        if _ctx_for_graph is not None and _ctx_for_graph.batch is not None:
            _g_handle = getattr(_ctx_for_graph.batch, "graph_handle", None)
            if _g_handle is not None and hasattr(_g_handle, "replay"):
                # Copy the live input_ids into the stable capture buffer before
                # replay (the graph was captured against the stable buffer).
                _buf = getattr(_ctx_for_graph.batch, "mtp_verify_input_buf", None)
                if _buf is not None and input_ids is not _buf:
                    _buf[: input_ids.shape[0]].copy_(input_ids)
                _g_handle.replay()
                _cached = getattr(_ctx_for_graph.batch, "mtp_verify_output", None)
                if _cached is not None:
                    _out, _raw = _cached
                    if return_raw:
                        return _out, _raw
                    return _out
                # Main-decode graph path: the engine may set batch.graph_handle
                # to the graph_runner graph + populate prev_hidden_buf / all_hidden_buf.
                _ah_buf = getattr(_ctx_for_graph.batch, "mtp_all_hidden_buf", None)
                if _ah_buf is not None and return_raw:
                    # Marker so forward_with_hidden knows the replay path took over and
                    # the lm_head call there should be skipped (logits already in buffer.logits).
                    _ctx_for_graph.batch.mtp_replayed = True
                    # Return raw pre-norm hidden for prev_hidden derivation.
                    return _ah_buf[: input_ids.shape[0]], _ah_buf[: input_ids.shape[0]]
                # If cache is missing, fall through to eager (shouldn't happen).

        x = self.embed_tokens.forward(input_ids)
        residual: torch.Tensor | None = None
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, residual)
        if return_raw:
            # Raw (pre-final-norm) hidden: the MTP head's expected input -- it has
            # its own pre_fc_norm_hidden, so feeding the post-norm hidden would
            # double-norm. Computed BEFORE the fused add-norm (which mutates the
            # residual buffer in place).
            raw = (x + residual) if residual is not None else x
            x, _ = self.norm.forward_add_residual(x, residual)
            return x, x
        x, _ = self.norm.forward_add_residual(x, residual)
        return x


class Qwen3_5MoEForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = Qwen3_5Model(config)
        if getattr(config, "lm_head_quant", "none") == "nvfp4":
            # checkpoint stores the (untied) lm_head as NVFP4: keep it native (W4A16) -- the
            # bf16 dequant of this ~1 GB matrix was the single largest decode kernel.
            from freetoken.kernel.triton.nvfp4_linear import Nvfp4LMHead

            assert not config.tie_word_embeddings, "NVFP4 lm_head assumes untied embeddings"
            self.lm_head = Nvfp4LMHead(
                num_embeddings=config.vocab_size, embedding_dim=config.hidden_size
            )
        elif getattr(config, "lm_head_quant", "none") == "fp8_pertensor":
            # Untied FP8-per-tensor head kept native (W8A16); see config.lm_head_quant.
            from freetoken.kernel.triton.fp8_pertensor_linear import Fp8LMHead

            assert not config.tie_word_embeddings, "FP8 lm_head assumes untied embeddings"
            self.lm_head = Fp8LMHead(
                num_embeddings=config.vocab_size, embedding_dim=config.hidden_size,
            )
        else:
            self.lm_head = ParallelLMHead(
                num_embeddings=config.vocab_size,
                embedding_dim=config.hidden_size,
                tie_word_embeddings=config.tie_word_embeddings,
                tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
            )
        super().__init__()

    def forward(self) -> torch.Tensor:
        output = self.model.forward(get_global_ctx().batch.input_ids)
        return self.lm_head.forward(output)

    def forward_with_hidden(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward that also returns the post-norm hidden state (before lm_head).

        The hidden state at each request's last token position is the prev_hidden
        consumed by the MTP head for that request's next speculative decode step.
        ``output`` here is paged-attention's flat [total_T, hidden] tensor (not a
        [bs, T, H] view), so we must index with ``attn_metadata.get_last_indices``
        to recover one row per request — not the slice ``output[:, -1:, :]``
        that would either error on the 2D shape or silently read the last H cols.

        Returns (logits, prev_hidden, all_hidden): prev_hidden is [bs, hidden_size]
        (0 rows when batch.size == 0); all_hidden is the full [total_T, H] hidden
        (MTP context seeding consumes it for prefill/verify batches).
        """
        batch = get_global_ctx().batch
        output, raw = self.model.forward(batch.input_ids, return_raw=True)
        if batch.size == 0:
            zeros = output.new_zeros((0, output.shape[-1]))
            return self.lm_head.forward(output), zeros, output.detach()
        indices = batch.attn_metadata.get_last_indices(batch.size)
        # MTP consumes the PRE-final-norm raw hidden -- the head has its own
        # pre_fc_norm_hidden and was trained against this distribution.
        prev_hidden = output[indices].detach()  # [bs, H]
        # If the inner forward already ran the captured graph (set the flag during
        # replay), the lm_head was captured too -- pull logits from buffer.logits
        # instead of re-running lm_head (which would re-trigger a fresh graph).
        if getattr(batch, "mtp_replayed", False):
            # Reset unconditionally: a stale True flag would make the NEXT eager
            # forward early-return the (wrong) buffered logits again.
            batch.mtp_replayed = False
            _logits_buf = getattr(batch, "mtp_logits_buf", None)
            if _logits_buf is not None:
                logits = _logits_buf[: batch.size].detach()
                return logits, prev_hidden, output.detach()
        logits = self.lm_head.forward(output)
        # Diagnostic (FT_MTP_DIAG=1): compare what the MTP head actually sees vs the
        # last-layer outputs the model produces. We log BOTH the pre-final-norm raw
        # (which MTP currently consumes via prev_hidden) AND the post-final-norm output
        # (the model's official "previous_hidden_state" per Qwen3-Next modeling code).
        import os as _os
        if _os.environ.get("FT_MTP_DIAG"):
            with torch.no_grad():
                pv = prev_hidden.float()
                op = output[indices].detach().float()
                rw = raw[indices].detach().float()
                print(
                    f"[MTP-diag] model_hidden: prev_hidden(raw)[n={pv.shape[0]}] "
                    f"mean={pv.mean().item():.3f} std={pv.std().item():.3f} "
                    f"min={pv.min().item():.3f} max={pv.max().item():.3f} "
                    f"norm={pv.norm(dim=-1).mean().item():.3f}",
                    flush=True,
                )
                print(
                    f"[MTP-diag] model_hidden: post_norm_output[n={op.shape[0]}] "
                    f"mean={op.mean().item():.3f} std={op.std().item():.3f} "
                    f"min={op.min().item():.3f} max={op.max().item():.3f} "
                    f"norm={op.norm(dim=-1).mean().item():.3f}",
                    flush=True,
                )
                print(
                    f"[MTP-diag] model_hidden: raw_pre_norm[n={rw.shape[0]}] "
                    f"mean={rw.mean().item():.3f} std={rw.std().item():.3f} "
                    f"min={rw.min().item():.3f} max={rw.max().item():.3f} "
                    f"norm={rw.norm(dim=-1).mean().item():.3f}",
                    flush=True,
                )
        return logits, prev_hidden, output.detach()

__all__ = ["Qwen3_5MoEForCausalLM"]
