"""MTP (Multi-Token Prediction) head for Qwen3.6 hybrid MoE models.

Architecture:
  1. embed[prev_token_id] -> RMSNorm (pre_fc_norm_embedding)
  2. prev_hidden          -> RMSNorm (pre_fc_norm_hidden)
  3. cat = concat of (1) and (2)
  4. fc GEMV (4096 -> 2048) with MXFP4 + bf16 per-block scale/bias (iGPU)
  5. + prev_hidden residual
  6. RMSNorm (input_layernorm) -> self_attn (gated MHA) -> +residual
  7. RMSNorm (post_attention_layernorm) -> MoE (256 routed + 1 shared) -> +residual
  8. RMSNorm (mtp.norm) -> LM head (tied with main model) -> draft logits

This is Phase 0 of MTP speculative decoding integration:
  * Standalone MTP head module loaded from Qwen3.6 safetensors
  * iGPU FC layer via D3D12 MXFP4 GEMV kernel (proven [P1a]/[P1b])
  * PyTorch for attn/MoE/RMSNorm (Phase 1 can move these to iGPU too)
  * No scheduler integration yet (Phase 2, requires KV partial rollback API)

Verified: MTP head loads from real checkpoint; FC layer numerically matches CPU ref
(see t_mtp_head_driver.py for the verification driver).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# Constants for Qwen3.6-35B-A3B-MXFP4-MTP MTP head
HIDDEN = 2048
VOCAB = 248320
NUM_EXPERTS = 256
NUM_EXPERTS_PER_TOK = 8
MOE_INTERMEDIATE = 512
SHARED_EXPERT_INTERMEDIATE = 512
HEAD_DIM = 256
NUM_QO_HEADS = 16
NUM_KV_HEADS = 2
PARTIAL_ROTARY_FACTOR = 0.25  # 64 / 256
RMS_NORM_EPS = 1e-6


@dataclass
class MtpHeadConfig:
    hidden_size: int = HIDDEN
    vocab_size: int = VOCAB
    num_experts: int = NUM_EXPERTS
    num_experts_per_tok: int = NUM_EXPERTS_PER_TOK
    moe_intermediate: int = MOE_INTERMEDIATE
    shared_expert_intermediate: int = SHARED_EXPERT_INTERMEDIATE
    head_dim: int = HEAD_DIM
    num_qo_heads: int = NUM_QO_HEADS
    num_kv_heads: int = NUM_KV_HEADS
    partial_rotary_factor: float = PARTIAL_ROTARY_FACTOR
    rms_norm_eps: float = RMS_NORM_EPS


def _rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = RMS_NORM_EPS) -> torch.Tensor:
    """RMSNorm in Gemma form: (x / rms(x)) * (1 + weight)."""
    inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    normed = x * inv_rms
    return normed * (1.0 + weight)


def _neox_rope(q, k, positions, rotary_dim):
    """Apply NeoX rotary embeddings to first `rotary_dim` columns.
    q, k: [N, num_heads, head_dim]. positions: [N]."""
    half = rotary_dim // 2
    cos = torch.cos(positions.float()).to(q.dtype).unsqueeze(-1).unsqueeze(-1)
    sin = torch.sin(positions.float()).to(q.dtype).unsqueeze(-1).unsqueeze(-1)
    q_rot = q.clone()
    k_rot = k.clone()
    i = torch.arange(half, device=q.device, dtype=torch.long)
    q1 = q_rot[..., i]; q2 = q_rot[..., i + half]
    q_rot[..., i] = q1 * cos - q2 * sin
    q_rot[..., i + half] = q1 * sin + q2 * cos
    k1 = k_rot[..., i]; k2 = k_rot[..., i + half]
    k_rot[..., i] = k1 * cos - k2 * sin
    k_rot[..., i + half] = k1 * sin + k2 * cos
    return q_rot, k_rot


class MtpHeadAttention(nn.Module):
    """Gated full attention in the MTP head."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_q = cfg.num_qo_heads
        self.num_kv = cfg.num_kv_heads
        self.head_dim = cfg.head_dim
        self.qo_dim = self.num_q * self.head_dim
        self.kv_dim = self.num_kv * self.head_dim
        self.split_sizes = [self.qo_dim * 2, self.kv_dim, self.kv_dim]
        self.qkv_proj = nn.Linear(cfg.hidden_size, sum(self.split_sizes), bias=False)
        self.q_norm = nn.Parameter(torch.zeros(self.head_dim))
        self.k_norm = nn.Parameter(torch.zeros(self.head_dim))
        self.o_proj = nn.Linear(self.qo_dim, cfg.hidden_size, bias=False)

    def forward(self, x, positions):
        """x: [N, H]. Returns: [N, H]."""
        qkv = self.qkv_proj(x)
        qg, k, v = torch.split(qkv, self.split_sizes, dim=-1)
        qg = qg.view(-1, self.num_q, self.head_dim * 2)
        q = qg[..., : self.head_dim]
        gate = qg[..., self.head_dim :]
        k = k.view(-1, self.num_kv, self.head_dim)
        v = v.view(-1, self.num_kv, self.head_dim)
        q = _rmsnorm(q, self.q_norm).reshape(-1, self.qo_dim)
        k = _rmsnorm(k, self.k_norm).reshape(-1, self.kv_dim)
        rotary_dim = int(self.head_dim * self.cfg.partial_rotary_factor)
        q = q.view(-1, self.num_q, self.head_dim)
        k = k.view(-1, self.num_kv, self.head_dim)
        q, k = _neox_rope(q, k, positions, rotary_dim)
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_logits = torch.einsum("nqd,nkd->nqk", q, k) * scale
        attn = F.softmax(attn_logits.float(), dim=-1).to(q.dtype)
        out = torch.einsum("nqk,nkd->nqd", attn, v)
        out = out * torch.sigmoid(gate)
        out = out.reshape(-1, self.qo_dim)
        return self.o_proj(out)


class MtpHeadMoe(nn.Module):
    """MoE block in MTP head: 256 switch experts (8 active) + 1 shared expert.

    switch_mlp weights: MXFP4 packed uint32 [256, K/8] per proj.
    Dequantized to bf16 at load time (~5s one-time cost, 1.5GB total).
    Forward: routing gate, top-8 expert selection, per-token 8-expert via torch.bmm.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_experts = cfg.num_experts
        self.top_k = cfg.num_experts_per_tok
        self.intermediate = cfg.moe_intermediate
        # Routing gate (bf16)
        self.gate = nn.Linear(cfg.hidden_size, cfg.num_experts, bias=False)
        # 256 switch experts (dequantized at load)
        self.switch_gate = nn.Parameter(torch.zeros(cfg.num_experts, cfg.moe_intermediate, cfg.hidden_size))
        self.switch_up = nn.Parameter(torch.zeros(cfg.num_experts, cfg.moe_intermediate, cfg.hidden_size))
        self.switch_down = nn.Parameter(torch.zeros(cfg.num_experts, cfg.hidden_size, cfg.moe_intermediate))
        # Shared expert (bf16)
        self.shared_gate = nn.Linear(cfg.hidden_size, cfg.shared_expert_intermediate, bias=False)
        self.shared_up = nn.Linear(cfg.hidden_size, cfg.shared_expert_intermediate, bias=False)
        self.shared_down = nn.Linear(cfg.shared_expert_intermediate, cfg.hidden_size, bias=False)
        self.shared_expert_gate = nn.Parameter(torch.zeros(1, cfg.hidden_size))

    def forward(self, x):
        """x: [N, H]. Returns: [N, H]. Real MoE forward with dequantized bf16 weights."""
        N = x.shape[0]
        # Routing gate
        gate_logits = self.gate(x)
        top_w, top_idx = torch.topk(gate_logits, self.top_k, dim=-1)
        top_w = top_w.softmax(dim=-1).to(x.dtype)
        # Per-token expert forward via batched matmul
        routed = torch.zeros_like(x)
        for k in range(self.top_k):
            eidx = top_idx[:, k]
            w = top_w[:, k].unsqueeze(-1)
            sg = F.silu(torch.bmm(self.switch_gate[eidx], x.unsqueeze(-1)).squeeze(-1))
            su = torch.bmm(self.switch_up[eidx], x.unsqueeze(-1)).squeeze(-1)
            sh_act = F.silu(sg) * su
            sd = torch.bmm(self.switch_down[eidx], sh_act.unsqueeze(-1)).squeeze(-1)
            routed = routed + sd * w
        # Shared expert
        ssg = F.silu(self.shared_gate(x))
        ssu = self.shared_up(x)
        shared = self.shared_down(F.silu(ssg) * ssu)
        shared = shared * torch.sigmoid(self.shared_expert_gate)
        return routed + shared


class Qwen3_5MtpHead(nn.Module):
    """Full MTP head forward (1 transformer layer)."""

    def __init__(self, cfg, embed_table, lm_head, igpu_fc=None, dtype=torch.bfloat16):
        super().__init__()
        self.cfg = cfg
        self.pre_fc_norm_embedding = nn.Parameter(torch.zeros(cfg.hidden_size))
        self.pre_fc_norm_hidden = nn.Parameter(torch.zeros(cfg.hidden_size))
        self.input_layernorm = nn.Parameter(torch.zeros(cfg.hidden_size))
        self.post_attention_layernorm = nn.Parameter(torch.zeros(cfg.hidden_size))
        self.mtp_norm = nn.Parameter(torch.zeros(cfg.hidden_size))
        self.attn = MtpHeadAttention(cfg)
        self.mlp = MtpHeadMoe(cfg)
        self.embed_table = embed_table
        self.lm_head = lm_head
        self.igpu_fc = igpu_fc
        self.dtype = dtype

    def forward(self, prev_token_id, prev_hidden):
        """prev_token_id: scalar int64. prev_hidden: [1, hidden] (bf16).
        Returns: logits [1, vocab_size] (bf16) for the next draft token."""
        emb = self.embed_table(prev_token_id)
        emb_n = _rmsnorm(emb, self.pre_fc_norm_embedding)
        hid_n = _rmsnorm(prev_hidden, self.pre_fc_norm_hidden)
        cat = torch.cat([emb_n, hid_n], dim=-1)
        cat_flat = cat.view(-1).to(torch.float32)
        if self.igpu_fc is not None:
            fc_out = self.igpu_fc(cat_flat).view(1, -1).to(self.dtype)
        else:
            fc_out = cat[:, :self.cfg.hidden_size]
        h = fc_out + prev_hidden
        h = _rmsnorm(h, self.input_layernorm)
        positions = torch.zeros(1, dtype=torch.long, device=h.device)
        attn_out = self.attn(h, positions)
        h = attn_out + h
        h = _rmsnorm(h, self.post_attention_layernorm)
        h = self.mlp(h) + h
        h = _rmsnorm(h, self.mtp_norm)
        logits = self.lm_head(h)
        return logits


def load_mtp_head_from_safetensors(model_path, cfg, embed_table, lm_head, igpu_fc=None, device="cuda", dtype=torch.bfloat16):
    """Two-pass: gather ALL mtp tensors across all safetensors files into one dict,
    then dequantize MXFP4 packed -> bf16 into the nn modules.

    fc stays packed (iGPU D3D12 path).
    attn + MoE switch + shared_expert dequant to bf16 (~5s one-time, 1.5GB RAM).
    """
    import safetensors.torch
    import json as _json
    from pathlib import Path
    import time as _time
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "benchmarks" / "cpu_moe_microbench"))
    from t_mxfp4_dequant import dequant_mxfp4_packed_row, dequant_mxfp4_weight_v2, dequant_mxfp4_expert_block, kE2M1

    idx_path = Path(model_path) / "model.safetensors.index.json"
    with open(idx_path) as f:
        idx = _json.load(f)
    wm = idx["weight_map"]
    mtp_keys = [k for k in wm if k.startswith("mtp.")]

    head = Qwen3_5MtpHead(cfg, embed_table, lm_head, igpu_fc=igpu_fc, dtype=dtype)
    head._packed_mxfp4 = {"fc.weight": None, "fc.biases": None, "fc.scales": None}
    # Two-pass: first gather ALL mtp tensors across all files, then process.
    all_state = {}
    for fname in set(wm[k] for k in mtp_keys):
        path = Path(model_path) / fname
        all_state.update(safetensors.torch.load_file(str(path)))

    import time as _time
    t_load0 = _time.time()
    for k in mtp_keys:
        if k not in all_state:
            continue
        tensor = all_state[k]
        rel = k[4:]

        # MXFP4 packed -- needs dequant
        if tensor.dtype == torch.uint32:
            # fc stays packed (iGPU path)
            if rel in ("fc.weight", "fc.biases", "fc.scales"):
                head._packed_mxfp4[rel] = tensor
                continue
            # attn and MoE -- dequant
            if rel.startswith("layers.0."):
                lr = rel[len("layers.0."):]
                if lr.startswith("self_attn."):
                    sa = lr[len("self_attn."):]
                    if sa.endswith(".weight") and not sa.endswith(".biases") and not sa.endswith(".scales"):
                        proj = sa[:-len(".weight")].rsplit(".", 1)[0]
                        s_key = f"mtp.layers.0.self_attn.{proj}.scales"
                        b_key = f"mtp.layers.0.self_attn.{proj}.biases"
                        if s_key in all_state and b_key in all_state:
                            dq = dequant_mxfp4_weight_v2(tensor.unsqueeze(0), all_state[s_key], all_state[b_key], K=tensor.shape[1]*8).squeeze(0)
                            dq = dq.to(dtype)
                            if proj == "q_proj":
                                head.attn.qkv_proj.weight.data[:cfg.num_qo_heads*cfg.head_dim*2, :].copy_(dq)
                            elif proj == "k_proj":
                                off = cfg.num_qo_heads*cfg.head_dim*2
                                head.attn.qkv_proj.weight.data[off:off+cfg.num_kv_heads*cfg.head_dim, :].copy_(dq)
                            elif proj == "v_proj":
                                off = cfg.num_qo_heads*cfg.head_dim*2 + cfg.num_kv_heads*cfg.head_dim
                                head.attn.qkv_proj.weight.data[off:off+cfg.num_kv_heads*cfg.head_dim, :].copy_(dq)
                            elif proj == "o_proj":
                                head.attn.o_proj.weight.data.copy_(dq)
                elif lr.startswith("mlp."):
                    mlr = lr[len("mlp."):]
                    if mlr.startswith("switch_mlp.") and mlr.endswith(".weight") and not mlr.endswith(".biases") and not mlr.endswith(".scales"):
                        proj = mlr.split(".")[-2]
                        s_key = f"mtp.layers.0.mlp.switch_mlp.{proj}.scales"
                        b_key = f"mtp.layers.0.mlp.switch_mlp.{proj}.biases"
                        if s_key in all_state and b_key in all_state:
                            t0 = _time.time()
                            dq = dequant_mxfp4_expert_block(tensor, all_state[s_key], all_state[b_key])
                            t1 = _time.time()
                            print(f"  Dequant {proj} ({tensor.shape}): {(t1-t0)*1000:.0f}ms -> {dq.shape}")
                            dq = dq.to(dtype)
                            if proj == "gate_proj":
                                head.mlp.switch_gate.data.copy_(dq)
                            elif proj == "up_proj":
                                head.mlp.switch_up.data.copy_(dq)
                            elif proj == "down_proj":
                                head.mlp.switch_down.data.copy_(dq)
                    elif mlr.startswith("shared_expert.") and mlr.endswith(".weight") and not mlr.endswith(".biases") and not mlr.endswith(".scales"):
                        proj = mlr.split(".")[-2]
                        s_key = f"mtp.layers.0.mlp.shared_expert.{proj}.scales"
                        b_key = f"mtp.layers.0.mlp.shared_expert.{proj}.biases"
                        if s_key in all_state and b_key in all_state:
                            dq = dequant_mxfp4_weight_v2(tensor.unsqueeze(0), all_state[s_key], all_state[b_key], K=tensor.shape[1]*8).squeeze(0)
                            dq = dq.to(dtype)
                            if proj == "gate_proj":
                                head.mlp.shared_gate.weight.data.copy_(dq)
                            elif proj == "up_proj":
                                head.mlp.shared_up.weight.data.copy_(dq)
                            elif proj == "down_proj":
                                head.mlp.shared_down.weight.data.copy_(dq)
                elif layer_rel == "input_layernorm.weight":
                    head.input_layernorm.data.copy_(tensor.float())
                elif layer_rel == "post_attention_layernorm.weight":
                    head.post_attention_layernorm.data.copy_(tensor.float())
            elif rel == "norm.weight":
                head.mtp_norm.data.copy_(tensor.float())
            elif rel == "pre_fc_norm_embedding.weight":
                head.pre_fc_norm_embedding.data.copy_(tensor.float())
            elif rel == "pre_fc_norm_hidden.weight":
                head.pre_fc_norm_hidden.data.copy_(tensor.float())
        else:
            # bf16 weights -> direct
            if rel == "pre_fc_norm_embedding.weight":
                head.pre_fc_norm_embedding.data.copy_(tensor.float())
            elif rel == "pre_fc_norm_hidden.weight":
                head.pre_fc_norm_hidden.data.copy_(tensor.float())
            elif rel == "norm.weight":
                head.mtp_norm.data.copy_(tensor.float())
            elif rel.startswith("layers.0."):
                lr = rel[len("layers.0."):]
                if lr == "input_layernorm.weight":
                    head.input_layernorm.data.copy_(tensor.float())
                elif lr == "post_attention_layernorm.weight":
                    head.post_attention_layernorm.data.copy_(tensor.float())
                elif lr == "mlp.gate.weight":
                    head.mlp.gate.weight.data.copy_(tensor)
                elif lr == "mlp.shared_expert_gate.weight":
                    head.mlp.shared_expert_gate.data.copy_(tensor.float())

    t_load1 = _time.time()
    print(f"  Total load: {t_load1-t_load0:.1f}s")
    return head.to(device).to(dtype)


__all__ = ["MtpHeadConfig", "Qwen3_5MtpHead", "MtpHeadAttention", "MtpHeadMoe", "load_mtp_head_from_safetensors"]
