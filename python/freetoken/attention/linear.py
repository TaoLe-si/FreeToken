from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from freetoken.core import Batch


@dataclass
class FLAMetadata:
    """Per-forward GatedDeltaNet (flash-linear-attention) metadata, built once per
    forward and shared by every GDN layer -- mirrors ``BaseAttnMetadata``. Replaces the
    per-layer rebuilds the GDN op used to do (``cu_seqlens`` arange, per-request
    ``cache_indices``/``has_initial_state``), which were pageable, synchronous H2D copies
    issued in each of the 30 GDN layers.

    Fields:
      cu_seqlens          query indptr; decode = arange(bs+1) (1 token/req), prefill =
                          cumsum of extend_len. int32 on device.
      cache_indices       per-request recurrent/conv state slot (= Req.table_idx). int32.
      has_initial_state   prefill only: whether each request continues a cached prefix
                          (cached_len > 0). None for decode (state always present).
      fresh_state_indices prefill only: the state-pool slots whose sequence is fresh
                          (cached_len == 0) and must be zeroed before the chunk kernel
                          reads them in place. None if there are none / for decode.
    """

    cu_seqlens: torch.Tensor
    cache_indices: torch.Tensor
    has_initial_state: torch.Tensor | None = None
    fresh_state_indices: torch.Tensor | None = None

    # --- MTP-verify partial-rollback fields (analog to track_* below); all default -1 / None ---
    # Per-verify-step GDN state trace (K+1 slots: snap[0]=pre-verify, snap[1..K]=post each
    # verify step, snap[K]=post-verify). Populated by scheduler._build_mtp_verify_batch on
    # MTP verify rounds; kernel writes per-step recurrent + conv snapshots when
    # mtp_verify_step_idx >= 0. Mirrors track_dst/track_h_row/track_conv_src on the next
    # fields -- the GDN op (_write_track_snapshot for track, _write_mtp_step_snapshot for MTP)
    # reads these in the same way. -1 means "not an MTP verify round"; the kernel then takes
    # its existing single-h-output path and leaves the snap slots untouched.
    mtp_verify_step_idx: int = -1                              # current verify step (0..K)
    mtp_verify_snap_slots: torch.Tensor | None = None          # [K+1] int64 dst pool slot per step

    # --- MTP-verify per-step decode persistent buffers (G.2; CUDA-graph-capture-friendly) ---
    # The C.4 per-step decode path needs three stable tensors + two stable host ints; without
    # these it would either allocate new tensors every forward (graph capture fails) or sync
    # via ``.item()`` (capture fails too). Pre-built in :meth:`Scheduler._prepare_batch` for
    # mtp_verify rounds and reused across graph replays; stable addresses are required for
    # graph capture / replay. ``mtp_verify_cu_seqlens_varlen = [0, K+1]`` (1 req, K+1 tokens),
    # ``mtp_verify_has_initial_state = [True]`` (live slot always has the prior decode state).
    mtp_verify_cu_seqlens_varlen: torch.Tensor | None = None    # [2] int32, persistent
    mtp_verify_has_initial_state: torch.Tensor | None = None    # [1] bool, persistent
    mtp_verify_snap_host_slots: list[int] | None = None          # K+1 host ints, stable per-req
    mtp_verify_live_slot: int = -1                               # host int, live slot for this req

    # --- hybrid-radix track-checkpoint (extra_buffer) fields; all None when not caching ---
    # For each request crossing a chunk-aligned (×CHUNK) boundary this forward, snapshot its
    # recurrent + conv state into a donatable pool slot, written on the forward stream by the
    # GDN op (see Qwen3_5GatedDeltaNet._write_track_snapshot). Built by the scheduler in P2;
    # left None by build_fla_metadata so the existing path is unchanged.
    track_dst: torch.Tensor | None = None        # [nt] int64 dst pool slot per tracked req
    track_h_row: torch.Tensor | None = None      # [nt] int64 row into h (boh_i + aligned//CHUNK)
    track_conv_src: torch.Tensor | None = None   # [nt, kernel-1] int64 conv-input token positions


def build_fla_metadata(batch: "Batch", device: torch.device) -> FLAMetadata:
    """C++ only -- no Python fallback by user request.

    Decode is one token per request, so ``cu_seqlens`` is a plain ``arange(bs+1)`` and
    ``cache_indices`` is ``batch.linear_table_idx`` (already int32). Prefill builds
    cu_seqlens (cumsum of extend_len), cache_indices (gdn_slot), has_initial_state, and
    fresh_state_indices. The C++ implementation replaces the per-req Python list
    comprehensions with one tight loop over a single host pinned tensor.
    """
    from freetoken.scheduler import _freetoken_sched as _sched_cpp
    reqs = batch.padded_reqs

    if batch.is_decode:
        cu_seqlens, cache_indices = _sched_cpp.build_decode_fla_meta(
            len(reqs), batch.linear_table_idx, device)
        return FLAMetadata(cu_seqlens=cu_seqlens, cache_indices=cache_indices)

    # prefill path
    pin = {"device": "cpu", "pin_memory": True}
    extend_lens = [int(r.extend_len) for r in reqs]
    cached_lens = [int(r.cached_len) for r in reqs]
    linear_slots = [int(r.linear_slot_idx) if r.linear_slot_idx is not None else -1 for r in reqs]
    table_idxs = [int(r.table_idx) for r in reqs]
    cu_host, idx_host, has_init_host, fresh_host = _sched_cpp.build_prefill_fla_meta(
        extend_lens, cached_lens, linear_slots, table_idxs)
    track_dst, track_h_row, track_conv_src = _build_track_metadata(reqs, cu_host, device, pin)

    return FLAMetadata(
        cu_seqlens=cu_host.to(device, non_blocking=True),
        cache_indices=idx_host.to(device, non_blocking=True),
        has_initial_state=has_init_host.to(device, non_blocking=True),
        fresh_state_indices=(
            fresh_host.to(device, non_blocking=True) if fresh_host.numel() > 0 else None
        ),
        track_dst=track_dst, track_h_row=track_h_row, track_conv_src=track_conv_src,
    )


def _build_track_metadata(reqs, cu_host, device, pin):
    """Hybrid-radix (extra_buffer): for each request that crosses a ×CHUNK boundary this
    prefill forward, snapshot its GDN state at the deepest mid-chunk boundary into its current
    ping-pong slot. Returns (track_dst, track_h_row, track_conv_src) device int64 tensors, or
    (None, None, None) when no request tracks (non-hybrid, or all extends < CHUNK+1)."""
    if not any(r.mamba_ping_pong is not None for r in reqs):
        return None, None, None
    from freetoken.core import get_global_ctx
    from freetoken.kernel.fla.chunk import CHUNK_SIZE
    from freetoken.kernel.fla.index import prepare_chunk_offsets

    km1 = get_global_ctx().linear_state_pool.conv_states.shape[-1]  # conv_kernel_dim - 1
    boh = prepare_chunk_offsets(cu_host, CHUNK_SIZE).tolist()
    dst, h_row, conv_src = [], [], []
    for i, r in enumerate(reqs):
        if r.mamba_ping_pong is None:
            continue
        # deepest mid-chunk boundary strictly inside the extend (h has the per-chunk state;
        # the exact extend-end / aligned-final state lives in the live slot -> finish-donate).
        c = (r.extend_len - 1) // CHUNK_SIZE
        if c < 1:
            continue
        off = int(cu_host[i])
        boundary = r.cached_len + c * CHUNK_SIZE
        dst.append(r.mamba_ping_pong[r.mamba_next_track_idx])
        h_row.append(boh[i] + c)
        conv_src.append([off + c * CHUNK_SIZE - km1 + j for j in range(km1)])
        r.mamba_last_track_seqlen = boundary
        r.mamba_next_track_idx = 1 - r.mamba_next_track_idx
    if not dst:
        return None, None, None
    to = lambda xs, **kw: torch.tensor(xs, **pin, **kw).to(device, non_blocking=True)
    return (to(dst, dtype=torch.int64), to(h_row, dtype=torch.int64),
            to(conv_src, dtype=torch.int64))


__all__ = ["FLAMetadata", "build_fla_metadata"]
