"""MTP (Multi-Token Prediction) head for Qwen3.6 hybrid MoE models.

Architecture:
  1. embed[prev_token_id] -> RMSNorm (pre_fc_norm_embedding)
  2. prev_hidden          -> RMSNorm (pre_fc_norm_hidden)
  3. cat = concat of (1) and (2)
  4. fc GEMV (4096 -> 2048) with MXFP4 + bf16 per-block scale/bias (iGPU/dGPU)
  5. + prev_hidden residual
  6. RMSNorm (input_layernorm) -> self_attn (gated MHA, full-context) -> +residual
  7. RMSNorm (post_attention_layernorm) -> MoE (256 routed + 1 shared) -> +residual
  8. RMSNorm (mtp.norm) -> LM head (tied with main model) -> draft logits

Working baseline restored after P3.0 mis-rollback.
"""
from __future__ import annotations

import math
import os
import time as _time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


_CUDA_AVAILABLE: bool = torch.cuda.is_available()
_MTP_DIAG: bool = os.environ.get("FT_MTP_DIAG") == "1"
_MTP_PROF: bool = os.environ.get("FT_MTP_PROF") == "1"


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
PARTIAL_ROTARY_FACTOR = 0.25
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
    rope_base: float = 10000.0
    norm_topk_prob: bool = True


_UAFF16 = torch.tensor([
    # The Qwen3.6 MXFP4 export is uint4-AFFINE (see checkpoint/quantize.py): nibbles
# are plain unsigned ints 0..15; value = nibble * scale + bias per 32-block. NOT
# e2m1-coded. (The original 24-entry e2m1-looking table mapped nibbles 8-15 to
# 0.0, zeroing half the weights -> garbage head output, flat logits, m=0.)
0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
], dtype=torch.float32)


def _dequant_mxfp4_affine(packed, scales, biases):
    if not torch.is_tensor(packed):
        packed = torch.from_numpy(packed)
    if not torch.is_tensor(scales):
        scales = torch.from_numpy(scales) if scales is not None else None
    if not torch.is_tensor(biases):
        biases = torch.from_numpy(biases) if biases is not None else None
    kw = packed.shape[-1]
    K = kw * 8
    lead = list(packed.shape[:-1])
    p = packed.to(torch.int64).reshape(-1, kw)
    shifts = (torch.arange(8, dtype=torch.int64) * 4).view(1, 1, 8)
    nibs = (p[:, :, None] >> shifts) & 0xF
    vals = _UAFF16.to(packed.device)[nibs.reshape(-1)].reshape(-1, K)
    ns = K // 32
    s = scales.reshape(-1, ns, 1).float()
    b = biases.reshape(-1, ns, 1).float()
    out = (vals.view(-1, ns, 32) * s + b).reshape(-1, K)
    return out.reshape(lead + [K])


def _rmsnorm(x, weight, eps=RMS_NORM_EPS):
    x32 = x.float()
    inv_rms = torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + eps)
    normed = x32 * inv_rms
    return normed.to(weight.dtype) * (1.0 + weight)


def _neox_rope(q, k, positions, rotary_dim, base=10000.0):
    half = rotary_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32, device=q.device) * 2.0 / rotary_dim))
    angles = torch.einsum("n,d->nd", positions.float(), inv_freq)
    cos = angles.cos().to(q.dtype).unsqueeze(1)
    sin = angles.sin().to(q.dtype).unsqueeze(1)
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
    """Persistent KV cache via torch.cat (working baseline behavior)."""

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
        self._draft_cache = None

    def reset_draft_cache(self):
        self._draft_cache = None

    def kv_len(self):
        cache = self._draft_cache
        return 0 if cache is None else cache[0].shape[0]

    def truncate_kv(self, n):
        if self._draft_cache is not None and self._draft_cache[0].shape[0] > n:
            k, v = self._draft_cache
            self._draft_cache = (k[:n], v[:n])

    def _project(self, x, positions):
        qkv = self.qkv_proj(x)
        qg, k, v = torch.split(qkv, self.split_sizes, dim=-1)
        qg = qg.view(-1, self.num_q, self.head_dim * 2)
        q = qg[..., : self.head_dim]
        gate = qg[..., self.head_dim:]
        k = k.view(-1, self.num_kv, self.head_dim)
        v = v.view(-1, self.num_kv, self.head_dim)
        q = _rmsnorm(q, self.q_norm).reshape(-1, self.num_q, self.head_dim)
        k = _rmsnorm(k, self.k_norm).reshape(-1, self.num_kv, self.head_dim)
        rotary_dim = int(self.head_dim * self.cfg.partial_rotary_factor)
        q, k = _neox_rope(q, k, positions, rotary_dim, base=getattr(self.cfg, "rope_base", 10000.0))
        return q, k, v, gate

    def append_rows(self, x, positions):
        _, k, v, _ = self._project(x, positions)
        cache = self._draft_cache
        if cache is None:
            self._draft_cache = (k, v)
        else:
            self._draft_cache = (
                torch.cat([cache[0], k], dim=0),
                torch.cat([cache[1], v], dim=0),
            )

    def forward(self, x, positions):
        q, k, v, gate = self._project(x, positions)
        cache = self._draft_cache
        if cache is None:
            self._draft_cache = (k, v)
        else:
            self._draft_cache = (
                torch.cat([cache[0], k], dim=0),
                torch.cat([cache[1], v], dim=0),
            )
        k_all, v = self._draft_cache
        rep = self.num_q // self.num_kv
        if hasattr(F, "scaled_dot_product_attention") and q.dtype != torch.float32:
            q4 = q.unsqueeze(2)
            k4 = k_all.unsqueeze(0).permute(0, 2, 1, 3)
            v4 = v.unsqueeze(0).permute(0, 2, 1, 3)
            out4 = F.scaled_dot_product_attention(
                q4, k4, v4, attn_mask=None, is_causal=False, scale=None, enable_gqa=True
            )
            out = out4.squeeze(2)
        else:
            kg = k_all.repeat_interleave(rep, dim=1)
            vg = v.repeat_interleave(rep, dim=1)
            scale = 1.0 / math.sqrt(self.head_dim)
            attn_logits = torch.einsum("nqd,mqd->nqm", q, kg) * scale
            attn = F.softmax(attn_logits.float(), dim=-1).to(q.dtype)
            out = torch.einsum("nqm,mqd->nqd", attn, vg)
        out = out * torch.sigmoid(gate)
        out = out.reshape(-1, self.qo_dim)
        return self.o_proj(out)


class MtpHeadMoe(nn.Module):
    """256 switch experts (top-8 active) + 1 shared expert."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_experts = cfg.num_experts
        self.top_k = cfg.num_experts_per_tok
        self.intermediate = cfg.moe_intermediate
        self.norm_topk_prob = bool(getattr(cfg, "norm_topk_prob", True))
        self.gate = nn.Linear(cfg.hidden_size, cfg.num_experts, bias=False)
        # Routed experts stay in MXFP4-packed form (u32 words + fp16 scales/biases).
        # The old bf16 dequant-at-load materialised 3 x [256, 512, 2048] bf16 = 1.5 GB
        # of VRAM that overflowed into WDDM shared memory on 8 GB GPUs and slowed
        # EVERY downstream step. Packed form is ~0.5 GB; forward dequantizes only the
        # top-k selected experts (<= 8 x 3 MB transient).
        E, I, H = cfg.num_experts, cfg.moe_intermediate, cfg.hidden_size
        # Packed words stored as int32: uint32 tensor indexing (index_cuda) is not
        # implemented on CUDA, and int32 keeps the same 4-byte bit pattern.
        self.register_buffer("sw_gate_packed", torch.zeros(E, I, H // 8, dtype=torch.int32), persistent=False)
        self.register_buffer("sw_gate_scales", torch.zeros(E, I, H // 32, dtype=torch.float16), persistent=False)
        self.register_buffer("sw_gate_biases", torch.zeros(E, I, H // 32, dtype=torch.float16), persistent=False)
        self.register_buffer("sw_up_packed", torch.zeros(E, I, H // 8, dtype=torch.int32), persistent=False)
        self.register_buffer("sw_up_scales", torch.zeros(E, I, H // 32, dtype=torch.float16), persistent=False)
        self.register_buffer("sw_up_biases", torch.zeros(E, I, H // 32, dtype=torch.float16), persistent=False)
        self.register_buffer("sw_down_packed", torch.zeros(E, H, I // 8, dtype=torch.int32), persistent=False)
        self.register_buffer("sw_down_scales", torch.zeros(E, H, I // 32, dtype=torch.float16), persistent=False)
        self.register_buffer("sw_down_biases", torch.zeros(E, H, I // 32, dtype=torch.float16), persistent=False)
        self.shared_gate = nn.Linear(cfg.hidden_size, cfg.shared_expert_intermediate, bias=False)
        self.shared_up = nn.Linear(cfg.hidden_size, cfg.shared_expert_intermediate, bias=False)
        self.shared_down = nn.Linear(cfg.shared_expert_intermediate, cfg.hidden_size, bias=False)
        self.shared_expert_gate = nn.Parameter(torch.zeros(1, cfg.hidden_size))
        self._shifts = (torch.arange(8, dtype=torch.int64) * 4).view(1, 1, 1, 8)

    def _dequant_sel(self, packed, scales, biases, eidx, dtype):
        """Dequantize the selected experts' MXFP4-affine weights.

        packed: [E, R, kw] u32 (kw words x 8 fp4 = K columns); scales/biases:
        [E, R, K//32] fp16. eidx: [S] expert ids. Returns [S, R, K] in ``dtype``
        with the exact same value formula as _dequant_mxfp4_affine (uaff16 table
        lookup, then fp32 (v * s + b))."""
        kw = packed.shape[-1]
        K = kw * 8
        ns = K // 32
        p = packed[eidx].to(torch.int64)             # [S, R, kw]
        shifts = self._shifts.to(packed.device)
        nibs = (p.unsqueeze(-1) >> shifts) & 0xF     # [S, R, kw, 8]
        vals = _UAFF16.to(packed.device)[nibs.reshape(-1)].reshape(-1, ns, 32)
        s = scales[eidx].reshape(-1, ns, 1).float()
        b = biases[eidx].reshape(-1, ns, 1).float()
        out = vals * s + b                           # [S*R, ns, 32] fp32
        return out.to(dtype).reshape(eidx.shape[0], -1, K)

    def forward(self, x):
        N = x.shape[0]
        gate_logits = self.gate(x).float()
        router_probs = F.softmax(gate_logits, dim=-1)
        top_w, top_idx = torch.topk(router_probs, self.top_k, dim=-1)
        if self.norm_topk_prob:
            top_w = top_w / top_w.sum(dim=-1, keepdim=True)
        top_w = top_w.to(x.dtype)
        # Batch-dequant ALL top-k selected experts once (instead of the old 8-iter
        # loop with 3 gathers + 3 bmms per iter), then one batched bmm per matrix.
        S = N * self.top_k
        eidx = top_idx.reshape(-1)
        g = self._dequant_sel(self.sw_gate_packed, self.sw_gate_scales, self.sw_gate_biases, eidx, x.dtype)
        u = self._dequant_sel(self.sw_up_packed, self.sw_up_scales, self.sw_up_biases, eidx, x.dtype)
        d = self._dequant_sel(self.sw_down_packed, self.sw_down_scales, self.sw_down_biases, eidx, x.dtype)
        x_rep = x.unsqueeze(1).expand(N, self.top_k, x.shape[-1]).reshape(S, -1)   # [S, H]
        sg = torch.bmm(g, x_rep.unsqueeze(-1)).squeeze(-1)                          # [S, I]
        su = torch.bmm(u, x_rep.unsqueeze(-1)).squeeze(-1)                          # [S, I]
        sh_act = F.silu(sg) * su
        sd = torch.bmm(d, sh_act.unsqueeze(-1)).squeeze(-1).reshape(N, self.top_k, -1)
        routed = (sd * top_w.unsqueeze(-1)).sum(dim=1)                              # [N, H]
        ssg = F.silu(self.shared_gate(x))
        ssu = self.shared_up(x)
        shared = self.shared_down(ssg * ssu)
        shared = shared * torch.sigmoid(self.shared_expert_gate)
        return routed + shared


class TorchNvfp4Fc:
    def __init__(self, packed_u32, scales_f32, biases_f32, device):
        w = _dequant_mxfp4_affine(packed_u32, scales_f32, biases_f32)
        self.w = w.to(device)
        self.K = w.shape[-1]

    def __call__(self, cat_flat):
        x = cat_flat.view(1, -1)
        return (self.w * x).sum(dim=1)

    def batch(self, x):
        return x @ self.w.t()


class DgpuBf16Fc:
    def __init__(self, packed_u32, scales_f32, biases_f32, device, dtype=torch.bfloat16):
        if not torch.is_tensor(packed_u32):
            packed_u32 = torch.from_numpy(packed_u32)
        if not torch.is_tensor(scales_f32):
            scales_f32 = torch.from_numpy(scales_f32) if scales_f32 is not None else None
        if not torch.is_tensor(biases_f32):
            biases_f32 = torch.from_numpy(biases_f32) if biases_f32 is not None else None
        w32 = _dequant_mxfp4_affine(packed_u32, scales_f32, biases_f32)
        self.w_bf16 = w32.to(dtype).to(device)
        self.K = self.w_bf16.shape[-1]
        self.dtype = dtype
        self.device = device

    def __call__(self, cat_flat):
        x = cat_flat.view(1, -1).to(self.dtype)
        out = F.linear(x, self.w_bf16)
        return out.view(-1)

    def batch(self, x):
        x_bf = x.to(self.dtype)
        return F.linear(x_bf, self.w_bf16)


class IgpuFcStickyCPP:
    def __init__(self, sticky):
        self._sticky = sticky

    def __call__(self, cat_flat):
        return self._sticky.torch()

    def batch(self, x):
        return torch.stack([self._sticky.torch() for _ in range(x.shape[0])], dim=0)


class Qwen3_5MtpHead(nn.Module):
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
        self._packed_mxfp4 = {"fc.weight": None, "fc.biases": None, "fc.scales": None}

    def extend_context(self, tokens, hiddens, start_pos):
        if tokens.numel() == 0:
            return
        emb = self.embed_table.forward(tokens.view(-1))
        emb_n = _rmsnorm(emb, self.pre_fc_norm_embedding)
        hid_n = _rmsnorm(hiddens, self.pre_fc_norm_hidden)
        cat = torch.cat([emb_n, hid_n], dim=-1)
        fc = self.igpu_fc
        if fc is not None and hasattr(fc, "batch"):
            fc_out = fc.batch(cat.float()).to(self.dtype)
        elif fc is not None:
            fc_out = torch.stack(
                [fc(cat[i].float().view(-1)) for i in range(cat.shape[0])]
            ).to(self.dtype)
        else:
            fc_out = cat[:, : self.cfg.hidden_size]
        # Must match forward_with_state: residual + main-model hidden before the
        # post-Norm layer that feeds the head's QKV. Without this the head-KV
        # rows diverge from what the MTP forward would compute at the same
        # position, so the head's attention queries against a different K/V
        # distribution than what it was trained with.
        h = fc_out + hiddens
        h = _rmsnorm(h, self.input_layernorm)
        positions = torch.arange(
            start_pos + 1, start_pos + 1 + tokens.numel(),
            device=h.device, dtype=torch.long,
        )
        self.attn.append_rows(h, positions)

    def forward_with_state(self, prev_token_id, prev_hidden, position=0):
        emb = self.embed_table.forward(prev_token_id.view(-1))
        emb_n = _rmsnorm(emb, self.pre_fc_norm_embedding)
        hid_n = _rmsnorm(prev_hidden, self.pre_fc_norm_hidden)
        cat = torch.cat([emb_n, hid_n], dim=-1)
        cat_flat = cat.view(-1).to(torch.float32)
        fc = self.igpu_fc
        t_fc0 = _time.perf_counter() if _MTP_PROF else 0.0
        if fc is not None:
            fc_out = fc(cat_flat).view(1, -1).to(self.dtype)
        else:
            fc_out = cat[:, : self.cfg.hidden_size]
        if _MTP_PROF:
            torch.cuda.synchronize()
            self._perf["fc"] += (_time.perf_counter() - t_fc0) * 1e6
            self._perf["fc_n"] += 1
        h = fc_out + prev_hidden
        if os.environ.get("FT_MTP_DEBUG"):
            print(f"[MTP-dbg] fcprobe: fc_out_norm={float(fc_out.float().norm()):.3f} "
                  f"in_hid_norm={float(prev_hidden.float().norm()):.3f} "
                  f"emb_n_norm={float(emb_n.float().norm()):.3f} hid_n_norm={float(hid_n.float().norm()):.3f}",
                  flush=True)
        h = _rmsnorm(h, self.input_layernorm)
        positions = torch.full((1,), int(position), dtype=torch.long, device=h.device)
        t_a0 = _time.perf_counter() if _MTP_PROF else 0.0
        attn_out = self.attn(h, positions)
        if _MTP_PROF:
            torch.cuda.synchronize()
            self._perf["attn"] += (_time.perf_counter() - t_a0) * 1e6
        h = attn_out + h
        h = _rmsnorm(h, self.post_attention_layernorm)
        t_m0 = _time.perf_counter() if _MTP_PROF else 0.0
        h = self.mlp(h) + h
        if _MTP_PROF:
            torch.cuda.synchronize()
            self._perf["moe"] += (_time.perf_counter() - t_m0) * 1e6
        h = _rmsnorm(h, self.mtp_norm)
        t_l0 = _time.perf_counter() if _MTP_PROF else 0.0
        logits = self.lm_head.forward(h)
        if _MTP_PROF:
            torch.cuda.synchronize()
            self._perf["lmh"] += (_time.perf_counter() - t_l0) * 1e6
        return logits, h

    def forward(self, prev_token_id, prev_hidden):
        return self.forward_with_state(prev_token_id, prev_hidden)[0]


def _to_device_streamed(head, device):
    for _, param in head.named_parameters():
        param.data = param.data.to(device)
    for _, buf in head.named_buffers():
        buf.data = buf.data.to(device)
    return head


def load_mtp_head_from_safetensors(
    model_path, cfg, embed_table, lm_head, igpu_fc=None,
    device="cuda", dtype=torch.bfloat16, fc_backend="dgpu",
):
    import safetensors.torch
    import json as _json
    from pathlib import Path

    sidecar_path = Path(model_path) / "mtp.safetensors"
    idx_path = Path(model_path) / "model.safetensors.index.json"
    if sidecar_path.is_file():
        mtp_files = [str(sidecar_path)]
    else:
        if not idx_path.is_file():
            raise FileNotFoundError(
                f"MTP head loader: no mtp.safetensors sidecar at {sidecar_path} "
                f"and no index at {idx_path}"
            )
        with open(idx_path) as f:
            idx = _json.load(f)
        wm = idx["weight_map"]
        mtp_files = sorted({str(Path(model_path) / fname) for k, fname in wm.items() if k.startswith("mtp.")})
        if not mtp_files:
            raise RuntimeError(f"No mtp.* tensors in {idx_path}")

    head = Qwen3_5MtpHead(cfg, embed_table, lm_head, igpu_fc=igpu_fc, dtype=dtype)
    head._packed_mxfp4 = {"fc.weight": None, "fc.biases": None, "fc.scales": None}

    all_state = {}
    for path in mtp_files:
        all_state.update(safetensors.torch.load_file(path))

    t_load0 = _time.time()
    mtp_keys = [k for k in all_state.keys() if k.startswith("mtp.")]
    for k in mtp_keys:
        tensor = all_state[k]
        rel = k[4:]
        # FC packed tensors: stored as-is regardless of dtype (weight=uint32,
        # biases+scales=fp16). The FC executor builder reads them and dequants.
        if rel in ("fc.weight", "fc.biases", "fc.scales"):
            head._packed_mxfp4[rel] = tensor
            continue
        if tensor.dtype == torch.uint32:
            if rel.startswith("layers.0."):
                lr = rel[len("layers.0."):]
                if lr.startswith("self_attn."):
                    sa = lr[len("self_attn."):]
                    if sa.endswith(".weight") and not sa.endswith(".biases") and not sa.endswith(".scales"):
                        proj = sa[:-len(".weight")].rsplit(".", 1)[0]
                        s_key = f"mtp.layers.0.self_attn.{proj}.scales"
                        b_key = f"mtp.layers.0.self_attn.{proj}.biases"
                        if s_key in all_state and b_key in all_state:
                            dq = _dequant_mxfp4_affine(tensor, all_state[s_key], all_state[b_key]).to(dtype)
                            if proj == "q_proj":
                                head.attn.qkv_proj.weight.data[:cfg.num_qo_heads * cfg.head_dim * 2, :].copy_(dq)
                            elif proj == "k_proj":
                                off = cfg.num_qo_heads * cfg.head_dim * 2
                                head.attn.qkv_proj.weight.data[off:off + cfg.num_kv_heads * cfg.head_dim, :].copy_(dq)
                            elif proj == "v_proj":
                                off = cfg.num_qo_heads * cfg.head_dim * 2 + cfg.num_kv_heads * cfg.head_dim
                                head.attn.qkv_proj.weight.data[off:off + cfg.num_kv_heads * cfg.head_dim, :].copy_(dq)
                            elif proj == "o_proj":
                                head.attn.o_proj.weight.data.copy_(dq)
                elif lr.startswith("mlp."):
                    mlr = lr[len("mlp."):]
                    if mlr.startswith("switch_mlp.") and mlr.endswith(".weight") and not mlr.endswith(".biases") and not mlr.endswith(".scales"):
                        proj = mlr.split(".")[-2]
                        s_key = f"mtp.layers.0.mlp.switch_mlp.{proj}.scales"
                        b_key = f"mtp.layers.0.mlp.switch_mlp.{proj}.biases"
                        if s_key in all_state and b_key in all_state:
                            # Keep experts PACKED (MXFP4 u32 + fp16 scales/biases).
                            # Forward dequantizes only the top-k selected experts;
                            # bf16 materialization here would cost 1.5 GB VRAM.
                            if proj == "gate_proj":
                                head.mlp.sw_gate_packed.data.copy_(tensor.view(torch.int32).reshape(head.mlp.sw_gate_packed.shape).to(head.mlp.sw_gate_packed.device))
                                head.mlp.sw_gate_scales.data.copy_(all_state[s_key].reshape(head.mlp.sw_gate_scales.shape).to(head.mlp.sw_gate_scales.device))
                                head.mlp.sw_gate_biases.data.copy_(all_state[b_key].reshape(head.mlp.sw_gate_biases.shape).to(head.mlp.sw_gate_biases.device))
                            elif proj == "up_proj":
                                head.mlp.sw_up_packed.data.copy_(tensor.view(torch.int32).reshape(head.mlp.sw_up_packed.shape).to(head.mlp.sw_up_packed.device))
                                head.mlp.sw_up_scales.data.copy_(all_state[s_key].reshape(head.mlp.sw_up_scales.shape).to(head.mlp.sw_up_scales.device))
                                head.mlp.sw_up_biases.data.copy_(all_state[b_key].reshape(head.mlp.sw_up_biases.shape).to(head.mlp.sw_up_biases.device))
                            elif proj == "down_proj":
                                head.mlp.sw_down_packed.data.copy_(tensor.view(torch.int32).reshape(head.mlp.sw_down_packed.shape).to(head.mlp.sw_down_packed.device))
                                head.mlp.sw_down_scales.data.copy_(all_state[s_key].reshape(head.mlp.sw_down_scales.shape).to(head.mlp.sw_down_scales.device))
                                head.mlp.sw_down_biases.data.copy_(all_state[b_key].reshape(head.mlp.sw_down_biases.shape).to(head.mlp.sw_down_biases.device))
                            print(f"  Loaded packed {proj} ({tensor.shape})", flush=True)
                    elif mlr.startswith("shared_expert.") and mlr.endswith(".weight") and not mlr.endswith(".biases") and not mlr.endswith(".scales"):
                        proj = mlr.split(".")[-2]
                        s_key = f"mtp.layers.0.mlp.shared_expert.{proj}.scales"
                        b_key = f"mtp.layers.0.mlp.shared_expert.{proj}.biases"
                        if s_key in all_state and b_key in all_state:
                            dq = _dequant_mxfp4_affine(tensor, all_state[s_key], all_state[b_key]).to(dtype)
                            if proj == "gate_proj":
                                head.mlp.shared_gate.weight.data.copy_(dq)
                            elif proj == "up_proj":
                                head.mlp.shared_up.weight.data.copy_(dq)
                            elif proj == "down_proj":
                                head.mlp.shared_down.weight.data.copy_(dq)
                elif lr == "input_layernorm.weight":
                    head.input_layernorm.data.copy_(tensor.float())
                elif lr == "post_attention_layernorm.weight":
                    head.post_attention_layernorm.data.copy_(tensor.float())
            elif rel == "norm.weight":
                head.mtp_norm.data.copy_(tensor.float())
            elif rel == "pre_fc_norm_embedding.weight":
                head.pre_fc_norm_embedding.data.copy_(tensor.float())
            elif rel == "pre_fc_norm_hidden.weight":
                head.pre_fc_norm_hidden.data.copy_(tensor.float())
        else:
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
    print(f"  Total load: {t_load1-t_load0:.1f}s", flush=True)

    if head.igpu_fc is None:
        packed_w = head._packed_mxfp4["fc.weight"]
        packed_s = head._packed_mxfp4["fc.scales"]
        packed_b = head._packed_mxfp4["fc.biases"]
        if packed_w is not None:
            if fc_backend == "dgpu":
                head.igpu_fc = DgpuBf16Fc(packed_w, packed_s, packed_b, device=device, dtype=dtype)
                print(f"  FC executor: DgpuBf16Fc (bf16, dGPU tensor cores)", flush=True)
            elif fc_backend == "torch":
                head.igpu_fc = TorchNvfp4Fc(packed_w, packed_s, packed_b, device=device)
                print(f"  FC executor: TorchNvfp4Fc (pure torch)", flush=True)
            elif fc_backend == "igpu":
                head.igpu_fc = DgpuBf16Fc(packed_w, packed_s, packed_b, device=device, dtype=dtype)
                print(f"  FC executor: DgpuBf16Fc (igpu requested but no wrapper)", flush=True)
            else:
                raise ValueError(f"Unknown fc_backend: {fc_backend!r}")

    return _to_device_streamed(head.to(device).to(dtype), device)


__all__ = [
    "MtpHeadConfig",
    "Qwen3_5MtpHead",
    "MtpHeadAttention",
    "MtpHeadMoe",
    "load_mtp_head_from_safetensors",
    "TorchNvfp4Fc",
    "DgpuBf16Fc",
    "IgpuFcStickyCPP",
]
