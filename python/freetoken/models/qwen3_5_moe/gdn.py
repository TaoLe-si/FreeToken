from __future__ import annotations

import torch
import torch.nn.functional as F
from freetoken.core import get_global_ctx
from freetoken.kernel.causal_conv1d import causal_conv1d_decode, causal_conv1d_varlen
from freetoken.layers import BaseOP, LinearColParallelMerged

from freetoken.kernel.triton.fp8_block_linear import Fp8BlockColMerged
from freetoken.kernel.triton.fp8_pertensor_linear import Fp8PerTensorColMerged

from .gdn_kernels import gdn_decode_fla, gdn_prefill_chunk_fla
from .quant_linear import make_replicated_quant


class _DepthwiseConv1d(BaseOP):
    """Holds the depthwise conv weight ``[conv_dim, 1, K]`` (key ``conv1d.weight``)."""

    def __init__(self, conv_dim: int, kernel: int):
        self.weight = torch.empty(conv_dim, 1, kernel)


class _GatedRMSNorm(BaseOP):
    """RMSNorm of x followed by a silu(z) gate (HF Qwen3_5MoeRMSNormGated).

    Uses the fused fla ``rms_norm_gated`` triton kernel (norm(x) * silu(z) in one
    kernel) instead of the unfused pow/mean/rsqrt/mul/silu chain, matching sglang's
    ``RMSNormGated`` -- collapses ~8 elementwise kernels per GDN layer into one."""

    def __init__(self, dim: int, eps: float):
        self.weight = torch.empty(dim)
        self.eps = eps

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        from freetoken.kernel.fla import rms_norm_gated

        return rms_norm_gated(
            x=x, weight=self.weight, bias=None, z=z, eps=self.eps,
            is_rms_norm=True, norm_before_gate=True, activation="silu",
        )


class Qwen3_5GatedDeltaNet(BaseOP):
    """GatedDeltaNet op using the vendored flash-linear-attention triton kernels
    (``freetoken.kernel.fla``) for the recurrence and a per-request
    recurrent + conv state held in ``ctx.linear_state_pool`` (keyed by ``Req.table_idx``).

    Parameter names match HF (``in_proj_qkv``/``in_proj_z``/``in_proj_b``/``in_proj_a``/
    ``conv1d``/``A_log``/``dt_bias``/``norm``/``out_proj``). Handles prefill (incl. chunked
    continuation) and single-token decode; state is fresh when ``req.cached_len == 0``.
    """

    def __init__(
        self, hidden_size, num_k_heads, num_v_heads, head_k_dim, head_v_dim,
        conv_kernel_size, rms_norm_eps, layer_id, expert_quant: str = "none",
        attn_quant: str = "none",
    ):
        self.layer_id = layer_id
        # The fla chunk/decode kernels read+write the recurrent state and the per-chunk h as
        # [V, K] while the LinearStatePool declares it [K, V]; these coincide (and the
        # hybrid-radix snapshot scatter h[h_row]->slot is a plain copy) only when the two head
        # dims are equal. Qwen3.5/3.6 satisfy this (128/128); guard any future config.
        assert head_k_dim == head_v_dim, (
            f"GatedDeltaNet requires head_k_dim == head_v_dim, got {head_k_dim} != {head_v_dim}"
        )
        self.num_k_heads = num_k_heads
        self.num_v_heads = num_v_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.key_dim = num_k_heads * head_k_dim
        self.value_dim = num_v_heads * head_v_dim
        self.conv_dim = 2 * self.key_dim + self.value_dim
        self.conv_kernel_size = conv_kernel_size
        # qkv|z carry a weight scale (block-fp8 weight_scale_inv, or per-tensor FP8
        # weight_scale); b|a stay bf16. Both quant modes therefore split the four-way
        # fusion into an fp8 qkvz GEMM + a bf16 ba GEMM (matches sglang/vLLM).
        self._block_fp8 = expert_quant == "fp8_block"
        self._pertensor_fp8 = attn_quant == "fp8_pertensor"
        # compressed-tensors / converted checkpoints with attn_quant=="nvfp4" keep the
        # qkv|z GEMM native W4A16 as well (Nvfp4DenseColMerged); b|a stay bf16 exactly
        # like the fp8 split. Same in_proj_qkvz/in_proj_ba split shape either way.
        self._nvfp4_attn = attn_quant == "nvfp4"
        self._fp8 = self._block_fp8 or self._pertensor_fp8
        self._split_proj = self._fp8 or self._nvfp4_attn

        self._in_proj_split = [self.conv_dim, self.value_dim, num_v_heads, num_v_heads]
        if self._fp8:
            ColMerged = Fp8BlockColMerged if self._block_fp8 else Fp8PerTensorColMerged
            self.in_proj_qkvz = ColMerged(
                hidden_size, [self.conv_dim, self.value_dim], has_bias=False
            )
            self.in_proj_ba = LinearColParallelMerged(
                hidden_size, [num_v_heads, num_v_heads], has_bias=False
            )
        elif self._nvfp4_attn:
            from freetoken.kernel.triton.nvfp4_linear import Nvfp4DenseColMerged
            self.in_proj_qkvz = Nvfp4DenseColMerged(
                hidden_size, [self.conv_dim, self.value_dim], has_bias=False
            )
            self.in_proj_ba = LinearColParallelMerged(
                hidden_size, [num_v_heads, num_v_heads], has_bias=False
            )
        else:
            # Fused input projection (one GEMM instead of four): qkv | z | b | a.
            self.in_proj = LinearColParallelMerged(hidden_size, self._in_proj_split, has_bias=False)
        self.conv1d = _DepthwiseConv1d(self.conv_dim, conv_kernel_size)
        # Recurrence-gating params kept in fp32 (exp/softplus is precision-sensitive,
        # and the fla kernel reads them as fp32) -- matches HF/sglang, and avoids a
        # per-call .float() upcast in the decode wrapper. The weight loader exempts
        # *.A_log / *.dt_bias from the model-dtype downcast.
        self.dt_bias = torch.empty(num_v_heads, dtype=torch.float32)
        self.A_log = torch.empty(num_v_heads, dtype=torch.float32)
        self.norm = _GatedRMSNorm(head_v_dim, eps=rms_norm_eps)
        # out_proj follows the checkpoint quant: block-fp8 / per-tensor-fp8 / compressed-tensors
        # NVFP4 (W4A16) / bf16. in_proj_* stay bf16 in every mode (above), so a compressed-tensors
        # NVFP4 checkpoint (attn_quant=="nvfp4") only makes out_proj native FP4.
        self.out_proj = make_replicated_quant(
            expert_quant, attn_quant, self.value_dim, hidden_size, has_bias=False
        )

    def _gate_params(self, a: torch.Tensor, b: torch.Tensor):
        beta = b.sigmoid()
        g = -self.A_log.exp() * F.softplus(a.float() + self.dt_bias)
        return g, beta

    def _conv_weight(self) -> torch.Tensor:
        return self.conv1d.weight.squeeze(1)  # [conv_dim, kernel] for the fused kernel

    def _conv_prefill(self, conv_in, pool, cu_seqlens, cache_indices, has_initial_state) -> torch.Tensor:
        """Varlen causal conv (fused sgl_kernel) with silu; reads/updates each request's
        conv state in place by ``cache_indices`` slot. ``conv_in`` [total, conv_dim].
        ``cu_seqlens`` / ``cache_indices`` / ``has_initial_state`` come from FLAMetadata."""
        li = pool.local_index(self.layer_id)
        # --kv-device cpu builds FLAMetadata index tensors on the host; the fused
        # CUDA conv kernel rejects host pointers, so stage them onto the compute
        # device here (tiny int32 payloads, negligible vs the prefill itself).
        dev = conv_in.device
        cu_seqlens = cu_seqlens.to(dev, non_blocking=True) if isinstance(cu_seqlens, torch.Tensor) else cu_seqlens
        cache_indices = cache_indices.to(dev, non_blocking=True) if isinstance(cache_indices, torch.Tensor) else cache_indices
        has_initial_state = has_initial_state.to(dev, non_blocking=True) if isinstance(has_initial_state, torch.Tensor) else has_initial_state
        x = conv_in.transpose(0, 1).contiguous()  # [conv_dim, total]
        out = causal_conv1d_varlen(x, self._conv_weight(), pool.conv_states[li],
                                   cu_seqlens, cache_indices, has_initial_state)
        return out.transpose(0, 1)  # [total, conv_dim]

    def _conv_decode(self, conv_in: torch.Tensor, table_idx: torch.Tensor, pool) -> torch.Tensor:
        """Single-token causal conv update (fused sgl_kernel) by ``table_idx`` slot;
        updates conv state in place, no host loop -> CUDA-graph capturable.
        ``conv_in`` [B, conv_dim] -> silu(conv) [B, conv_dim]."""
        li = pool.local_index(self.layer_id)
        table_idx = table_idx.to(conv_in.device, non_blocking=True) if isinstance(table_idx, torch.Tensor) else table_idx
        return causal_conv1d_decode(conv_in, pool.conv_states[li], self._conv_weight(), table_idx)

    def _conv_varlen(self, conv_in, pool, cu_seqlens, indices, has_initial_state, max_seq_len=None):
        """Varlen causal conv1d for the MTP-verify single-request multi-token sequential
        case. Drives the sgl_kernel causal_conv1d_fwd kernel with cu_seqlens=[0, K+1] and
        indices=[slot]; updates conv_states[li][slot] in place across the K+1 tokens.
        Returns silu(conv) of shape [total, conv_dim].

        ``max_seq_len`` MUST be a host int (K+1) passed by the caller -- deriving it
        from cu_seqlens via .item() would sync and break CUDA graph capture."""
        from freetoken.kernel.causal_conv1d import causal_conv1d_varlen
        li = pool.local_index(self.layer_id)
        dev = conv_in.device
        x = conv_in.transpose(0, 1).contiguous()  # [conv_dim, total]
        # Verify path: batch=1 (single request). max_seq_len comes from the caller as a
        # host int so CUDA graph capture avoids any device->host sync inside the launch.
        out = causal_conv1d_varlen(
            x, self._conv_weight(), pool.conv_states[li], cu_seqlens, indices, has_initial_state,
            batch=1, max_seq_len=max_seq_len
        )
        return out.transpose(0, 1)  # [total, conv_dim]

    # Max snap slots per verify (covers K up to _MAX_VERIFY_SNAP-1). The persistent
    # dst_t buffer is sized to this so it can be reused across all K values.
    _MAX_VERIFY_SNAP = 16

    def _get_verify_buffers(self, device: torch.device):
        """Lazy-init persistent buffers for graph-capture-friendly MTP verify.

        Three buffers on the compute device, all stable across replays:
          cu_seqlens_int64 : [2] int64 copy of the int32 mtp_verify_cu_seqlens_varlen.
                              Persistent so the kernel call sees a stable address.
          dst_t             : [_MAX_VERIFY_SNAP] int64 of the snap dst indices.
                              Updated in place from snap_host_slots; the kernel uses
                              dst_t[:n] so the unused tail is ignored.
          host_t            : [_MAX_VERIFY_SNAP] int64 scratch used to stage the host
                              snap list into device memory without per-call allocs.
                              Persisted so the .copy_() in graph_capture mode targets
                              a fixed device address (otherwise graph replay would
                              read a freed/overwritten address).
        One allocation per (layer, device); reused for the rest of the process.
        """
        bufs = getattr(self, "_verify_buffers", None)
        if bufs is None or bufs[0].device != device:
            cu_seqlens_int64 = torch.empty(2, dtype=torch.int64, device=device)
            dst_t = torch.empty(self._MAX_VERIFY_SNAP, dtype=torch.int64, device=device)
            # host_t must be PINNED CPU memory: it stages the host snap list and the
            # capturable copy_ into dst_t is then a pinned H2D (legal inside capture).
            # A device-resident staging tensor would force per-element scalar fills ->
            # unpinned H2D -> 'Cannot copy between CPU and CUDA tensors during capture'.
            host_t = torch.empty(self._MAX_VERIFY_SNAP, dtype=torch.int64, pin_memory=True)
            bufs = (cu_seqlens_int64, dst_t, host_t)
            self._verify_buffers = bufs
        return bufs

    def stage_verify_snap(self, snap_host_slots, device):
        """Re-stage the persistent pinned host_t buffer with the CURRENT round's snap
        slots. Must be called at replay-BIND time (outside the captured graph): the
        captured d2d copy re-executes on every replay and reads whatever host_t holds.
        Pure CPU writes -- legal anywhere, no CUDA ops."""
        cu_seqlens_int64, dst_t_buf, host_t_buf = self._get_verify_buffers(device)
        dst = snap_host_slots[1:] if len(snap_host_slots) > 1 else []
        n = min(len(dst), self._MAX_VERIFY_SNAP)
        for _i in range(n):
            host_t_buf[_i] = int(dst[_i])
        return n

    def _forward_mtp_verify(self, hidden_states, fla, pool, dtype):
        """C.4 per-step decode path for MTP verify batches (qwen3_5_moe hybrid GDN).

        Runs the K+1 verify tokens as ONE varlen-style sequential decode against the
        request live state slot -- NOT as K+1 Python-loop kernel launches. Keeps
        Python-side overhead per verify forward at ~5ms vs ~50ms for a per-step Python
        loop (each launch pays ~50us in GIL + dict-lookup + device-sync chain).

        Per-layer cost (24 GDN layers, K=2):
          - 1 _conv_varlen call with cu_seqlens=[0, K+1], indices=[slot]
          - 1 gdn_decode_fla call with IS_VARLEN=True, cu_seqlens=[0, K+1], indices=[slot]
          - 1 batched snap copy (recurrent + conv) into snap_slots[1..K]
        Total launches per verify forward (K=2): 24*(1+1+1) = 72 (was 96). Wall time
        ~40ms on this build (vs ~190ms for the chunk prefill path which triggers the
        iGPU service extra dense-FFN latency on each prefill-shaped step).

        G.4 graph-capture-friendly: cu_seqlens_int64 and dst_t are persistent buffers
        (reused across replays); only their contents change, never their address.
        """
        # --- Hoist batch-level metadata to locals (was 4 getattr/forward, now 4 reads).
        #     All set by scheduler._prepare_batch and stable for this verify batch's
        #     lifetime -- no per-call hasattr/getattr fallbacks needed (those were dead
        #     code in the verify path; the kernel accepts the scheduler's tensors directly).
        batch = get_global_ctx().batch
        cu_seqlens_varlen = fla.mtp_verify_cu_seqlens_varlen
        has_initial_state = fla.mtp_verify_has_initial_state
        snap_host_slots = fla.mtp_verify_snap_host_slots
        live_slot = fla.mtp_verify_live_slot
        indices = fla.cache_indices
        graph_capture = getattr(batch, "graph_capture_active", False)
        # Bind hot-path methods to locals (avoid attribute lookups in the compute path).
        in_proj_qkvz_fwd = self.in_proj_qkvz.forward
        in_proj_ba_fwd = self.in_proj_ba.forward
        # self.in_proj only exists when _split_proj is False (single merged projection).
        # Bind lazily to avoid AttributeError when running in split_proj mode.
        in_proj_fwd = self.in_proj.forward if not self._split_proj else None
        norm_fwd = self.norm.forward
        out_proj_fwd = self.out_proj.forward
        num_v_heads = self.num_v_heads
        head_v_dim = self.head_v_dim
        num_k_heads = self.num_k_heads
        head_k_dim = self.head_k_dim
        key_dim = self.key_dim
        value_dim = self.value_dim
        conv_dim = self.conv_dim
        in_proj_split = self._in_proj_split
        scale = head_k_dim ** -0.5

        total = hidden_states.shape[0]
        K_plus_1 = total
        # --- In-proj: split into conv_in, b, a, z.
        if self._split_proj:
            qkvz = in_proj_qkvz_fwd(hidden_states)
            conv_in, z = torch.split(qkvz, [conv_dim, value_dim], dim=-1)
            ba = in_proj_ba_fwd(hidden_states)
            b, a = torch.split(ba, [num_v_heads, num_v_heads], dim=-1)
        else:
            proj = in_proj_fwd(hidden_states)
            conv_in, z, b, a = torch.split(proj, in_proj_split, dim=-1)
        z = z.reshape(total, num_v_heads, head_v_dim)

        # --- Pool / layer / indices (all stable per forward).
        li = pool.local_index(self.layer_id)
        if not isinstance(indices, torch.Tensor):
            indices = torch.as_tensor(indices, dtype=torch.int32, device=conv_in.device)
        # G.4: persistent int64 cu_seqlens buffer. We can't use .to(int64) because
        # that creates a new tensor each call (breaks graph replay). The kernel
        # internally does tl.load(...).to(tl.int64) so we can pass int32 directly,
        # but the FLA kernel signature wants int64 -- use the persistent buffer.
        #
        # Refresh: use a single-element vectorized cast + copy (no .item() syncs).
        cu_seqlens_int64, dst_t_buf, host_t_buf = self._get_verify_buffers(conv_in.device)
        if cu_seqlens_varlen.dtype == torch.int64:
            cu_seqlens_int64.copy_(cu_seqlens_varlen)
        else:
            # int32 -> int64 elementwise copy into the persistent buffer (vectorized,
            # no host syncs -- all kernel launches). The persistent int64 buffer's
            # .copy_(int32_input) does the dtype cast automatically.
            cu_seqlens_int64.copy_(cu_seqlens_varlen)

        # --- 1) Conv: varlen over [0, K+1] -- 1 launch per layer.
        # K_plus_1 is a host int (line above) -- no device->host sync during capture.
        mixed = self._conv_varlen(conv_in, pool, cu_seqlens_varlen, indices, has_initial_state,
                                  max_seq_len=K_plus_1)
        # --- 2) Recurrent: gdn_decode_fla with cu_seqlens=[0, K+1] -- 1 launch per layer.
        qf, kf, vf = torch.split(mixed, [key_dim, key_dim, value_dim], dim=-1)
        # Reshape + dtype cast. .to(dtype) is a no-op when the conv kernel already produces
        # the target dtype (the common case).
        q = qf.reshape(1, K_plus_1, num_k_heads, head_k_dim).to(dtype)
        k = kf.reshape(1, K_plus_1, num_k_heads, head_k_dim).to(dtype)
        v = vf.reshape(1, K_plus_1, num_v_heads, head_v_dim).to(dtype)
        core_out = gdn_decode_fla(
            q, k, v, a, b,
            A_log=self.A_log, dt_bias=self.dt_bias,
            state_source=pool.recurrent_states[li], indices=indices,
            cu_seqlens=cu_seqlens_int64, scale=scale,
        )

        # --- 3) Batched snap copy. The C.4 varlen path only writes the FINAL recurrent
        # state; the per-step entries in the snap are all post-GDN copies of the live
        # state. Collapse K-1 individual .copy_() calls into a single indexed assignment
        # (`rec[dst_indices] = rec[live:live+1].expand(...)`) -- one launch per layer for
        # each of recurrent + conv, instead of K-1 launches each. Pool.copy_from() copies
        # ALL 24 layers (24x bandwidth waste) so we keep the per-layer slice.
        rec = pool.recurrent_states[li]
        rec_live = rec[live_slot:live_slot + 1]  # [1, ...] view, 0 alloc
        conv_li = pool.conv_states[li]
        conv_live = conv_li[live_slot:live_slot + 1]  # [1, ...] view, 0 alloc
        # Build dst_indices as a device int64 tensor. Skip the live_slot itself and any
        # negative / sentinel slots; under graph_capture the original code skipped the
        # dst_slot != live_slot check (the graph sees a constant stream of snap[1..K]), so
        # the equivalent here is: always include the snap slots, never include live.
        snap_dst = snap_host_slots[1:K_plus_1]  # K-1 host ints
        n_dst = len(snap_dst)
        if n_dst > 0:
            if graph_capture:
                # G.4: refresh the persistent dst_t buffer in place. snap_dst is a host
                # list of length K-1 <= _MAX_VERIFY_SNAP; the unused tail is ignored because
                # we only call index_copy_ on dst_t[:n_dst].
                if n_dst > self._MAX_VERIFY_SNAP:
                    raise RuntimeError(
                        f"MTP verify K={n_dst+1} exceeds persistent snap buffer "
                        f"({self._MAX_VERIFY_SNAP}); bump _MAX_VERIFY_SNAP"
                    )
                # Stage host ints into the persistent host_t buffer (in-place fill).
                # The fill_ kernel targets a fixed address (host_t_buf) so the .copy_
                # from host_t_buf[:n_dst] into dst_t_buf[:n_dst] (recorded in the graph)
                # has stable source/dest addresses across replays.
                host_t_buf[:n_dst].fill_(0)
                for _i, _s in enumerate(snap_dst):
                    host_t_buf[_i] = _s
                dst_t_buf[:n_dst].copy_(host_t_buf[:n_dst])
                dst_t = dst_t_buf[:n_dst]
            else:
                # Filter the snap list: drop the live_slot and any negatives. Recurrent + conv
                # state are small enough that the host-side filter is free; we only build a
                # device tensor when the filtered list is non-empty.
                filtered = [s for s in snap_dst if s != live_slot and s >= 0]
                if filtered:
                    dst_t = torch.as_tensor(filtered, dtype=torch.int64, device=rec.device)
                else:
                    dst_t = None
            if dst_t is not None:
                # .expand() returns a view that aliases rec_live memory; under CUDA graph capture the
                # index_copy_ needs a tensor that does NOT alias rec -- clone the expanded view.
                rec_src = rec_live.expand(dst_t.shape[0], *rec.shape[1:]).contiguous().clone()
                rec.index_copy_(0, dst_t, rec_src)
                conv_src = conv_live.expand(dst_t.shape[0], *conv_li.shape[1:]).contiguous().clone()
                conv_li.index_copy_(0, dst_t, conv_src)

        # Per-head norm: flatten (T, HV, V) -> (T*HV, V) so weight (head_v_dim,) lines up.
        # Then reshape back to (T, value_dim) before out_proj.
        core_out = core_out.reshape(-1, head_v_dim)
        z = z.reshape(-1, head_v_dim)
        out = norm_fwd(core_out, z).reshape(total, -1)
        return out_proj_fwd(out)

    def _write_track_snapshot(self, pool, li: int, conv_in: torch.Tensor,
                              h: torch.Tensor, fla) -> None:
        """Snapshot this layer's recurrent + conv state at the chunk-aligned track boundary
        into a donatable pool slot, on the forward stream (hybrid-radix extra_buffer path).
        SSM: ``recurrent_states[li, dst] = h[0, h_row]`` -- a DIRECT copy (h is [V,K], the
        state pool is [K,V]; they coincide because GDN requires head_k_dim == head_v_dim).
        Conv: the last (kernel-1) raw conv-input timesteps ending at the boundary."""
        rec = pool.recurrent_states[li]
        rec.index_copy_(0, fla.track_dst, h[0, fla.track_h_row].to(rec.dtype))
        cv = pool.conv_states[li]
        # conv_in [total, conv_dim]; gather the (kernel-1) window per tracked req.
        conv_win = conv_in[fla.track_conv_src].transpose(-1, -2).contiguous()  # [nt, conv_dim, K-1]
        cv.index_copy_(0, fla.track_dst, conv_win.to(cv.dtype))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        batch = ctx.batch
        pool = ctx.linear_state_pool
        total = hidden_states.shape[0]
        dtype = hidden_states.dtype

        # Per-forward GDN metadata (cu_seqlens / cache_indices / continuation flags),
        # built once and shared by all GDN layers. The scheduler/graph set it; build it
        # lazily here (cached on the batch) for direct-op callers (tests).
        fla = batch.fla_metadata
        if fla is None:
            from freetoken.attention.linear import build_fla_metadata

            fla = build_fla_metadata(batch, hidden_states.device)
            batch.fla_metadata = fla

        if self._split_proj:
            qkvz = self.in_proj_qkvz.forward(hidden_states)
            conv_in, z = torch.split(qkvz, [self.conv_dim, self.value_dim], dim=-1)
            ba = self.in_proj_ba.forward(hidden_states)
            b, a = torch.split(ba, [self.num_v_heads, self.num_v_heads], dim=-1)
        else:
            proj = self.in_proj.forward(hidden_states)
            conv_in, z, b, a = torch.split(proj, self._in_proj_split, dim=-1)
        z = z.reshape(total, self.num_v_heads, self.head_v_dim)
        li = pool.local_index(self.layer_id)

        # MTP verify path (C.4 re-enabled 2026-08-29). The chunk prefill path (used for
        # both regular prefill AND mtp verify since the previous C.4 revert) trips an
        # extra-dense kernel-launch sequence on the chunk prefill attention backend
        # which interacts badly with the iGPU-service-backed dense FFN (--dense-ffn-engine
        # igpu / --mtp-igpu-fc) -- empirically K=2 verify forward on this build takes
        # ~190ms via chunk prefill, vs ~70ms via per-step decode, dropping K=2 MTP from
        # ~25 tok/s theoretical to 6.4 tok/s observed. C.4 is the per-step decode +
        # per-step GDN snap path that mirrors llama.cpp PR #22400's "GDN intermediates":
        #
        #   For step in range(K+1):
        #     conv_in_step = conv_in[step:step+1]
        #     mixed_step = self._conv_decode(conv_in_step, indices, pool)   # [1, conv_dim]
        #     # ... split qf/kf/vf, single-token gdn_decode_fla ...
        #     core_out_step = gdn_decode_fla(..., cu_seqlens=[0, 1], indices=indices)
        #     out_list.append(core_out_step)
        #     if step + 1 < K+1: snap[step+1] <- pool.copy_from(slot, snap[step+1])
        #                              + conv_states[li][snap[step+1]] <- conv_states[li][slot]
        #
        # where snap[0] is pre-verify (filled by _build_mtp_verify_batch) and snap[step] is
        # the live-slot state after the step-th decode. snap[K] == live slot post-verify
        # (no copy needed). _mtp_process_verify rolls back to snap[n] for partial accept n
        # in [0, K-1]; full accept (n == K) leaves the live slot untouched.
        #
        # Why this is net-faster than the chunk prefill path despite K+1 sequential
        # gdn_decode_fla launches per layer: the kernel-launch sequence stays on the
        # decode-shaped attention backend (FlashInfer paged decode wrapper, no chunk
        # prefill kernel) which keeps the iGPU service's per-step latency off the
        # verify critical path.
        is_mtp_verify = getattr(batch, "mtp_verify", False)
        if is_mtp_verify:
            return self._forward_mtp_verify(hidden_states, fla, pool, dtype)
        if batch.is_decode:
            # Fused fla decode kernel: gating + in-kernel l2norm + recurrent update +
            # per-request state read/write-by-index, all in one kernel (no gather/scatter,
            # no clone, no external l2norm). q/k stay at num_k_heads (kernel handles GQA).
            mixed = self._conv_decode(conv_in, fla.cache_indices, pool)  # [B, conv_dim]
            B = mixed.shape[0]
            qf, kf, vf = torch.split(mixed, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
            q = qf.reshape(1, B, self.num_k_heads, self.head_k_dim).to(dtype)
            k = kf.reshape(1, B, self.num_k_heads, self.head_k_dim).to(dtype)
            v = vf.reshape(1, B, self.num_v_heads, self.head_v_dim).to(dtype)
            core_out = gdn_decode_fla(
                q, k, v, a, b, A_log=self.A_log, dt_bias=self.dt_bias,
                state_source=pool.recurrent_states[li], indices=fla.cache_indices,
                cu_seqlens=fla.cu_seqlens, scale=self.head_k_dim ** -0.5,
            )
        else:
            mixed = self._conv_prefill(
                conv_in, pool, fla.cu_seqlens, fla.cache_indices, fla.has_initial_state)
            # fla chunk handles GQA in-kernel: q/k stay at num_k_heads, v at num_v_heads.
            qf, kf, vf = torch.split(mixed, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
            q = qf.reshape(1, total, self.num_k_heads, self.head_k_dim).to(dtype)
            k = kf.reshape(1, total, self.num_k_heads, self.head_k_dim).to(dtype)
            v = vf.reshape(1, total, self.num_v_heads, self.head_v_dim).to(dtype)
            g, beta = self._gate_params(a, b)
            g = g.reshape(1, total, self.num_v_heads)
            beta = beta.float().reshape(1, total, self.num_v_heads)
            # The chunk kernel reads + writes back initial_state[cache_indices] in place;
            # fresh sequences (cached_len==0) must start from a zeroed slot.
            if fla.fresh_state_indices is not None:
                pool.recurrent_states[li].index_fill_(0, fla.fresh_state_indices, 0.0)
            track = fla.track_dst is not None
            result = gdn_prefill_chunk_fla(
                q, k, v, g, beta,
                state_source=pool.recurrent_states[li], indices=fla.cache_indices,
                cu_seqlens=fla.cu_seqlens, scale=self.head_k_dim ** -0.5,
                return_h=track,
            )
            if track:
                core_out, h = result
                self._write_track_snapshot(pool, li, conv_in, h, fla)
            else:
                core_out = result

        core_out = core_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        out = self.norm.forward(core_out, z).reshape(total, -1)
        return self.out_proj.forward(out)


__all__ = ["Qwen3_5GatedDeltaNet"]
