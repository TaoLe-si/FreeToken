from __future__ import annotations

from typing import Sequence

import torch
from freetoken.distributed import get_tp_info
from freetoken.utils import div_even

from .base import BaseKVCachePool


class MHAKVCache(BaseKVCachePool):
    """
    Base class for key-value caches.
    This class defines the interface for key-value caches used in LLMs.

    ``layer_ids`` lets the pool back only a *subset* of the model's layers while
    callers keep indexing by their global ``layer_id``. Hybrid models (e.g. the
    Qwen3.5 GatedDeltaNet/full-attention stack) interleave linear-attention layers
    that hold no paged KV; passing the full-attention layer ids here allocates one
    storage slab per KV layer (not per model layer) and remaps the global id to its
    dense slot, avoiding a multiple-x over-allocation of unused slabs.
    """

    def __init__(
        self,
        num_kv_heads: int,
        num_layers: int,
        head_dim: int,
        num_pages: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
        layer_ids: Sequence[int] | None = None,
        kv_quant: str = "bf16",
    ) -> None:
        assert kv_quant in ("bf16", "q8_0"), (
            f"MHAKVCache only supports bf16 / q8_0, got {kv_quant!r}"
        )
        tp_info = get_tp_info()
        local_kv_heads = div_even(num_kv_heads, tp_info.size, allow_replicate=True)
        self._num_layers = num_layers
        if layer_ids is None:
            num_storage_layers = num_layers
            self._layer_map: list[int] | None = None
        else:
            num_storage_layers = len(layer_ids)
            layer_map = [-1] * num_layers
            for dense, global_id in enumerate(layer_ids):
                if global_id < 0 or global_id >= num_layers:
                    raise ValueError(f"KV layer id {global_id} outside [0, {num_layers})")
                layer_map[global_id] = dense
            self._layer_map = layer_map
        self._device = device
        self._head_dim = head_dim
        self._quant = kv_quant
        if kv_quant == "q8_0":
            # llama.cpp-style block-wise q8_0: 32 int8 values + 1 fp16 scale per block.
            # Storage: separate int8 data + fp16 scale buffers per K/V; dequant staging
            # buffer (bf16) is what attention sees via k_cache() / v_cache().
            assert dtype == torch.bfloat16, "q8_0 path requires bf16 dtype for the dequant stage"
            assert head_dim % 32 == 0, f"q8_0 needs head_dim % 32 == 0, got head_dim={head_dim}"
            block_dim = 32
            num_blocks = head_dim // block_dim
            data_shape = (num_storage_layers, num_pages, page_size, local_kv_heads, head_dim)
            scale_shape = (num_storage_layers, num_pages, page_size, local_kv_heads, num_blocks)
            self._k_data = torch.empty(data_shape, dtype=torch.int8, device=device)
            self._v_data = torch.empty(data_shape, dtype=torch.int8, device=device)
            self._k_scales = torch.empty(scale_shape, dtype=torch.float16, device=device)
            self._v_scales = torch.empty(scale_shape, dtype=torch.float16, device=device)
            self._k_dequant = torch.empty(data_shape, dtype=torch.bfloat16, device=device)
            self._v_dequant = torch.empty(data_shape, dtype=torch.bfloat16, device=device)
            # _k_buffer / _v_buffer point at the dequant stage so the existing
            # k_cache(idx) / v_cache(idx) views stay bf16-typed and the triton kernel
            # reads the same tensor layout it always has.
            self._kv_buffer = None  # rebuild() uses the per-tensor q8_0 layout
            self._k_buffer = self._k_dequant
            self._v_buffer = self._v_dequant
            self._storage_shape = (num_pages * page_size, local_kv_heads, head_dim)
            self._dirty = True  # force a dequant pass before the first read
        else:
            self._kv_buffer = torch.empty(
                (2, num_storage_layers, num_pages, page_size, local_kv_heads, head_dim),
                device=device,
                dtype=dtype,
            )
            self._k_buffer = self._kv_buffer[0]
            self._v_buffer = self._kv_buffer[1]
            self._storage_shape = (num_pages * page_size, local_kv_heads, head_dim)
            self._dirty = False

    def rebuild(self, num_pages: int) -> None:
        """Reallocate the KV buffer for ``num_pages`` pages IN PLACE.

        Geometry (storage layers, page_size, kv heads, head_dim) is taken from the
        existing buffer; only the page count changes. Views and ``_storage_shape`` are
        refreshed. Object identity is preserved so cached backend references stay valid.
        For q8_0 the K and V live in separate int8 data + fp16 scale + bf16 dequant
        tensors; each is reallocated independently to match ``num_pages``.
        """
        device = self._device
        if self._quant == "q8_0":
            # Re-derive page-fixed geometry from the existing dequant stage (it owns the
            # authoritative (storage_layers, page_size, kv heads, head_dim) tuple).
            _, num_storage_layers, _old_pages, page_size, local_kv_heads, head_dim = self._k_dequant.shape
            self._k_buffer = None
            self._v_buffer = None
            self._k_dequant = None
            self._v_dequant = None
            self._k_data = None
            self._v_data = None
            self._k_scales = None
            self._v_scales = None
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()
            data_shape = (num_storage_layers, num_pages, page_size, local_kv_heads, head_dim)
            scale_shape = (num_storage_layers, num_pages, page_size, local_kv_heads, head_dim // 32)
            self._k_data = torch.empty(data_shape, dtype=torch.int8, device=device)
            self._v_data = torch.empty(data_shape, dtype=torch.int8, device=device)
            self._k_scales = torch.empty(scale_shape, dtype=torch.float16, device=device)
            self._v_scales = torch.empty(scale_shape, dtype=torch.float16, device=device)
            self._k_dequant = torch.empty(data_shape, dtype=torch.bfloat16, device=device)
            self._v_dequant = torch.empty(data_shape, dtype=torch.bfloat16, device=device)
            self._k_buffer = self._k_dequant
            self._v_buffer = self._v_dequant
            self._storage_shape = (num_pages * page_size, local_kv_heads, head_dim)
            self._dirty = True
            return
        _, num_storage_layers, _old_pages, page_size, local_kv_heads, head_dim = self._kv_buffer.shape
        dtype = self._kv_buffer.dtype
        self._k_buffer = None
        self._v_buffer = None
        self._kv_buffer = None
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
        self._kv_buffer = torch.empty(
            (2, num_storage_layers, num_pages, page_size, local_kv_heads, head_dim),
            device=device,
            dtype=dtype,
        )
        self._k_buffer = self._kv_buffer[0]
        self._v_buffer = self._kv_buffer[1]
        self._storage_shape = (num_pages * page_size, local_kv_heads, head_dim)
        self._dirty = False

    @classmethod
    def kv_cost(cls, config) -> tuple[int, int, int, int]:
        from .base import spec_kv_bytes_per_token

        per_token = sum(
            spec_kv_bytes_per_token(spec, config)
            for spec in config.model_config.kv_cache_group_specs()
            if not spec.is_swa
        )
        # q8_0 halves the per-element footprint: (32 int8 + 2 fp16 scale) / 32 = 1.0625
        # byte/elem vs 2 for bf16. spec_kv_bytes_per_token assumes the dtype itemsize
        # the engine was created with (bf16=2), so rescale the K/V slabs here. The DSA
        # index-key slab is NOT quantized (kept bf16 by current design).
        if getattr(config, "kv_quant", "bf16") == "q8_0":
            # Rescale the K + V slab footprint (the index-key slab stays bf16):
            # q8_0 = (32 * 1 + 2) / 32 = 1.0625 byte/elem vs bf16 = 2.
            from .base import spec_kv_bytes_per_token as _spec_bpt
            kv_only = sum(
                _spec_bpt(spec, config) - spec.index_head_dim * spec.num_index_layers * 2
                for spec in config.model_config.kv_cache_group_specs()
                if not spec.is_swa
            )
            index_only = per_token - kv_only
            per_token = int(round(kv_only * 1.0625 / 2.0)) + index_only
        return per_token * config.page_size, 0, config.page_size, 0

    def rebuild_from_config(
        self, config, num_pages: int, *, num_swa_pages: int | None = None
    ) -> None:
        self.rebuild(num_pages + 1)  # +1 for the dummy page (matches create_kvcache_pool)

    def unit_bytes(self) -> tuple[int, int]:
        # Cache-slider denominators: per-token live VRAM cost. The bf16 layout stores
        # 2 bytes/elem; the q8_0 layout stores ~1.0625 bytes/elem of K+V plus the bf16
        # dequant staging buffer (kept in case a deferred dequant is needed). Sum the
        # real buffers, do not pretend the dequant stage is free -- callers budget from
        # this number.
        if self._quant == "q8_0":
            tokens = int(self._k_data.shape[2]) * int(self._k_data.shape[3])
            data_bytes = self._k_data.numel() + self._v_data.numel()  # int8
            scale_bytes = self._k_scales.numel() * self._k_scales.element_size()
            scale_bytes += self._v_scales.numel() * self._v_scales.element_size()
            dequant_bytes = (self._k_dequant.numel() + self._v_dequant.numel()) * 2
            return (data_bytes + scale_bytes + dequant_bytes) // tokens, 0
        buf = self._kv_buffer
        tokens = int(buf.shape[2]) * int(buf.shape[3])
        return int(buf.numel() * buf.element_size()) // tokens, 0

    def _dense(self, layer_id: int) -> int:
        if self._layer_map is None:
            return layer_id
        dense = self._layer_map[layer_id]
        if dense < 0:
            raise KeyError(f"layer {layer_id} has no paged KV storage")
        return dense

    def k_cache(self, index: int) -> torch.Tensor:
        return self._k_buffer[self._dense(index)]

    def v_cache(self, index: int) -> torch.Tensor:
        return self._v_buffer[self._dense(index)]

    def store_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        out_loc: torch.Tensor,
        layer_id: int,
    ) -> None:
        dense = self._dense(layer_id)
        if self._quant == "q8_0":
            self._store_kv_q8(k, v, out_loc, dense)
            return
        from freetoken.kernel import store_cache

        store_cache(
            k_cache=self._k_buffer[dense].view(self._storage_shape),
            v_cache=self._v_buffer[dense].view(self._storage_shape),
            indices=out_loc,
            k=k,
            v=v,
        )

    def _store_kv_q8(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        out_loc: torch.Tensor,
        dense: int,
    ) -> None:
        # llama.cpp-style block-wise q8_0 quantization of the newly written tokens.
        # k, v arrive as bf16 [T, Hh * D] from the attention layers -- the same 2D
        # contract as the bf16 store_cache path (attention.py reshapes to [-1, kv_dim]).
        # Normalize to the 3D block view this path quantizes; 3D input also accepted.
        # out_loc: [T] row indices (int32/int64) into the pool's row axis.
        if k.dim() == 2:
            k = k.view(k.shape[0], self._k_data.shape[-2], self._head_dim)
        if v.dim() == 2:
            v = v.view(v.shape[0], self._v_data.shape[-2], self._head_dim)
        T, Hh, D = k.shape
        nb = D // 32
        # Block-wise amax + scale (cast to fp32 for the reduction math).
        k_blocks = k.view(T, Hh, nb, 32).to(torch.float32)
        v_blocks = v.view(T, Hh, nb, 32).to(torch.float32)
        k_amax = k_blocks.abs().amax(dim=-1, keepdim=True)
        v_amax = v_blocks.abs().amax(dim=-1, keepdim=True)
        eps = torch.finfo(torch.float32).eps
        k_scale = (k_amax / 127.0).clamp(min=eps)
        v_scale = (v_amax / 127.0).clamp(min=eps)
        k_q = (k_blocks / k_scale).round().clamp(-128, 127).to(torch.int8)
        v_q = (v_blocks / v_scale).round().clamp(-128, 127).to(torch.int8)
        # Flatten the block dim back into the head_dim axis for storage.
        k_q_flat = k_q.view(T, Hh, D)
        v_q_flat = v_q.view(T, Hh, D)
        k_scale_flat = k_scale.view(T, Hh, nb).to(torch.float16)
        v_scale_flat = v_scale.view(T, Hh, nb).to(torch.float16)
        # Scatter into the pool. index_copy_ takes care of duplicate / ordering; the
        # dequant stage is NOT updated here -- attention triggers it on demand.
        idx = out_loc.long()
        self._k_data[dense].view(-1, Hh, D).index_copy_(0, idx, k_q_flat)
        self._v_data[dense].view(-1, Hh, D).index_copy_(0, idx, v_q_flat)
        self._k_scales[dense].view(-1, Hh, nb).index_copy_(0, idx, k_scale_flat)
        self._v_scales[dense].view(-1, Hh, nb).index_copy_(0, idx, v_scale_flat)

    def ensure_layer_dequanted(self, layer_id: int) -> None:
        # Stage the int8 / fp16 storage into the bf16 dequant buffer the triton kernel
        # actually reads. Cheap relative to attention; always run when q8_0 is in use.
        # (Per-layer dirty tracking would let us skip the re-dequant when only OTHER
        # layers wrote since the last call here, but the kv-cache size per layer is
        # small enough that the saving is not worth the bookkeeping.)
        if self._quant != "q8_0":
            return
        dense = self._dense(layer_id)
        head_dim = self._head_dim
        nb = head_dim // 32
        for buf_data, buf_scales, buf_out in (
            (self._k_data[dense], self._k_scales[dense], self._k_dequant[dense]),
            (self._v_data[dense], self._v_scales[dense], self._v_dequant[dense]),
        ):
            flat_d = buf_data.view(-1, head_dim)
            flat_s = buf_scales.view(-1, nb)
            flat_o = buf_out.view(-1, head_dim)
            dq = (
                flat_d.view(-1, nb, 32).to(torch.float32)
                * flat_s.unsqueeze(-1).to(torch.float32)
            ).view(-1, head_dim).to(torch.bfloat16)
            flat_o.copy_(dq)

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        # The dequant stage (bf16) is what downstream code reads; report bf16 here so
        # triton attention's dtype checks pass unchanged when q8_0 is in use.
        if self._quant == "q8_0":
            return self._k_dequant.dtype
        return self._kv_buffer.dtype

    @property
    def num_layers(self) -> int:
        return self._num_layers
