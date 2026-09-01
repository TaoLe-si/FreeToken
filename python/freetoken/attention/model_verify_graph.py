"""Model-level CUDA graph wrapper for the MTP verify forward (G.3 + G.4, 2026-09-02).

Mirrors freetoken.attention.fi.FlashInferBackend's graph_wrappers /
init_capture_graph / prepare_for_replay pattern but wraps the WHOLE 24-layer
Qwen3_5Model.forward(input_ids, return_raw=True) call so a single graph
replay replaces 24 sequential layer forwards + their per-layer attn/GDN/MoE
kernel launches.

Why model-level (not GDN-only like the G.3 prototype): the MTP verify
forward is dominated by ~265 kernel launches (~2 ms minimum even on the
5090). CUDA graph collapses this to a single dispatch (~10 us) for ~50-200x
reduction in launch overhead.

G.4 changes (2026-09-02):
  * Graph cache is keyed on (padded_size, num_tokens) instead of just
    padded_size. The MTP verify path processes K+1 tokens per single-request
    batch, where K varies between 1 and --mtp-k; each K produces a different
    input shape and a different graph.
  * Stable per-(bs, num_tokens) input buffer. The scheduler's batch.input_ids
    is a freshly materialised tensor each round (token_pool[input_mapping]
    creates a new tensor), so the graph CANNOT capture it as the address
    would change. Instead we pre-allocate a stable input buffer and copy
    the live batch.input_ids into it before each replay (the copy happens
    OUTSIDE the captured kernel sequence, so the graph still has a fixed
    input address).
  * gdn.py's _forward_mtp_verify now uses persistent buffers (cu_seqlens_int64,
    dst_t) whose ADDRESSES are stable across replays and whose CONTENTS are
    refreshed in place each call.

Constraints (G.2 must hold): no .item() syncs; stable device tensors; no
host-side branching that differs between capture and replay -- gdn.py's
_forward_mtp_verify now uses persistent buffers to satisfy this.

Usage:
  backend = ModelVerifyGraphBackend(model, max_k=8)
  backend.init_capture_graph(bs=1, num_tokens=4)   # one capture per (bs, K+1)
  batch.graph_handle = backend.graph_wrappers[(bs, num_tokens)]
  batch.graph_capture_active = True
  out, raw = model.forward(input_ids, return_raw=True)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from freetoken.models.qwen3_5_moe.model import Qwen3_5Model


class ModelVerifyGraphBackend:
    """Per-(bs, num_tokens) CUDAGraph wrapper for the whole MTP-verify model forward.

    Caches 1 CUDAGraph per (padded_size, num_tokens) pair. The graph captures
    model.forward(input_ids, return_raw=True) and replays it on every subsequent
    call with the same shape -- the launch overhead collapses from ~2 ms to ~10 us.

    For MTP verify, padded_size is always 1 (single-request batch), and num_tokens
    = K+1 where K is the MTP draft count. Lazy capture: each new (bs, K+1) gets
    its graph on first use, then every subsequent verify with that K replays it.

    Each (bs, num_tokens) shape has its own stable input buffer (allocated once on
    first use). The model's forward call copies batch.input_ids into this stable
    buffer before the replay, so the captured kernel sequence always reads from
    the SAME memory address.
    """

    def __init__(self, model, max_k: int = 8) -> None:
        self.model = model
        self.max_k = max_k
        self.graph_wrappers = {}
        self.capture_warmed = set()
        self.capture_stream = None
        self.capture_inputs = {}      # (bs, num_tokens) -> stable input buffer
        self.capture_outputs = {}     # (bs, num_tokens) -> (output, raw) captured result

    def _key(self, bs: int, num_tokens: int) -> tuple[int, int]:
        return (bs, num_tokens)

    def init_capture_graph(self, bs: int, num_tokens: int) -> None:
        """Capture a graph for a (bs, num_tokens) shape. Idempotent."""
        key = self._key(bs, num_tokens)
        if key in self.graph_wrappers:
            return
        if not torch.cuda.is_available():
            raise RuntimeError("ModelVerifyGraphBackend: CUDA required for graph capture")

        # Resolve the capture device. The wrapped model may be a BaseOP (no
        # nn.Module .parameters()) or an nn.Module with no parameters yet
        # (e.g. during a cold-init regression test). Both paths fall back to
        # the live batch's input_ids, then to cuda:0.
        device = None
        params = getattr(self.model, "parameters", None)
        if callable(params):
            try:
                for p in self.model.parameters():
                    device = p.device
                    break
            except Exception:
                device = None
        if device is None:
            tensors = getattr(self.model, "state_dict", lambda: {})()
            for v in tensors.values():
                if isinstance(v, torch.Tensor):
                    device = v.device
                    break
        if device is None:
            from freetoken.core import get_global_ctx
            ctx = get_global_ctx()
            if ctx is not None and getattr(ctx, "batch", None) is not None:
                batch = ctx.batch
                if getattr(batch, "input_ids", None) is not None:
                    device = batch.input_ids.device
        if device is None:
            device = torch.device("cuda:0")
        if self.capture_stream is None:
            self.capture_stream = torch.cuda.Stream(device=device)

        from freetoken.core import get_global_ctx
        ctx = get_global_ctx()
        batch = ctx.batch if ctx is not None else None
        if batch is None or not hasattr(batch, "input_ids") or batch.input_ids is None:
            raise RuntimeError(
                "ModelVerifyGraphBackend.init_capture_graph: batch.input_ids not "
                "available; capture must be called from within a forward_batch context"
            )
        # Verify the actual input matches the requested shape.
        actual_tokens = batch.input_ids.shape[0]
        if actual_tokens != num_tokens:
            raise RuntimeError(
                f"ModelVerifyGraphBackend: requested num_tokens={num_tokens} but "
                f"batch.input_ids has {actual_tokens} rows; capture shape mismatch"
            )

        # G.4 stable buffer: allocate the input buffer ONCE for this (bs, num_tokens)
        # and copy the live batch.input_ids into it. The model.forward call later
        # copies new live input into this SAME buffer before each replay, so the
        # graph always reads from a fixed address.
        capture_in = torch.empty(num_tokens, dtype=torch.long, device=device)
        capture_in.copy_(batch.input_ids)
        self.capture_inputs[key] = capture_in

        # Warmup.
        if key not in self.capture_warmed:
            with torch.cuda.stream(self.capture_stream):
                self._capture_run(key, warmup=True)
            torch.cuda.current_stream().wait_stream(self.capture_stream)
            self.capture_warmed.add(key)

        # Capture.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=self.capture_stream):
            out, raw = self._capture_run(key, warmup=False)
            self.capture_outputs[key] = (out, raw)
        torch.cuda.current_stream().wait_stream(self.capture_stream)
        self.graph_wrappers[key] = graph

    def _capture_run(self, key, warmup):
        from freetoken.core import get_global_ctx
        ctx = get_global_ctx()
        batch = ctx.batch if ctx is not None else None
        if batch is None:
            raise RuntimeError(
                "ModelVerifyGraphBackend: no batch context available during capture"
            )
        batch.graph_capture_active = True
        try:
            # Capture uses the stable buffer (same address each capture/replay).
            return self.model.forward(self.capture_inputs[key], return_raw=True)
        finally:
            # Do not leak the staging-mode flag into subsequent eager calls on the
            # same batch object (warmup runs eagerly BEFORE the capture context).
            batch.graph_capture_active = False

    def prepare_for_replay(self, batch) -> None:
        """Bind the captured graph for this batch's (bs, num_tokens) on the batch.

        Also pre-copies the live batch.input_ids into the stable buffer so the
        next replay reads the right tokens.
        """
        bs = batch.padded_size
        num_tokens = int(batch.input_ids.shape[0])
        key = self._key(bs, num_tokens)
        if key not in self.graph_wrappers:
            self.init_capture_graph(bs, num_tokens)
        # Refresh the stable input buffer in place -- kernel reads from the SAME
        # address on every replay. The copy here is a kernel launch on the same
        # stream as the upcoming graph.replay(); PyTorch's stream ordering ensures
        # the copy completes before the replay starts.
        self.capture_inputs[key].copy_(batch.input_ids)
        batch.graph_handle = self.graph_wrappers[key]
        batch.mtp_verify_input_buf = self.capture_inputs[key]
        batch.mtp_verify_output = self.capture_outputs[key]
        # Re-stage the GDN per-step snap slots for THIS round. The captured graph
        # re-executes the recorded pinned->device copy on every replay, so host_t
        # must hold the current request's snap slots before replay starts.
        snap_slots = getattr(batch.fla_metadata, "mtp_verify_snap_host_slots", None)
        if snap_slots:
            for layer in self.model.layers.op_list:
                gdn = getattr(layer, "linear_attn", None)
                if gdn is not None and hasattr(gdn, "stage_verify_snap"):
                    gdn.stage_verify_snap(snap_slots, batch.input_ids.device)
        # Flag stays on for the duration of replay.
        batch.graph_capture_active = True

    def clear(self, batch) -> None:
        batch.graph_handle = None
        batch.mtp_verify_input_buf = None
        batch.mtp_verify_output = None
        batch.graph_capture_active = False


__all__ = ["ModelVerifyGraphBackend"]
