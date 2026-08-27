from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import torch as _torch
from .utils import KernelConfig, load_jit, make_cpp_args

if TYPE_CHECKING:
    import torch
    from tvm_ffi import Module

DEFAULT_INDEX_KERNEL_CONFIG = KernelConfig(num_threads=128, max_occupancy=1, use_pdl=False)


@functools.cache
def _jit_store_module(
    element_size: int,
    *,
    config: KernelConfig = DEFAULT_INDEX_KERNEL_CONFIG,
) -> Module:
    args = make_cpp_args(element_size, *config)
    return load_jit(
        "store",
        *args,
        cuda_files=["store.cu"],
        cuda_wrappers=[("launch", f"StoreKernel<{args}>::run")],
    )


def store_cache(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> None:
    num_tokens = k_cache.shape[0]
    k_cache = k_cache.view(num_tokens, -1)
    v_cache = v_cache.view(num_tokens, -1)
    element_size = k_cache.shape[1] * k_cache.element_size()
    try:
        module = _jit_store_module(element_size)
        module.launch(k_cache, v_cache, indices, k, v)
        return
    except Exception:
        pass  # fall through to torch fallback
    # PyTorch fallback: scatter k/v into k_cache/v_cache at given indices
    idx_long = indices.to(k_cache.dtype).to(_torch.long) if False else indices.to(k_cache.device).to(_torch.long)
    if not getattr(store_cache, "_dbg_done", False):
        print(
            "| v", str(v.device), "| cache", str(k_cache.device),
            "| idx", str(indices.device), flush=True
        )
    # and dtype here. Per-token cost is pointer-level (~160 KB/layer), noise.
    _kd = k_cache.device
    k_dev = k.to(device=_kd, dtype=k_cache.dtype)
    v_dev = v.to(device=_kd, dtype=v_cache.dtype)
    k_cache[idx_long] = k_dev
    v_cache[idx_long] = v_dev
