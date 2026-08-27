from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Tuple

from .utils import KernelConfig, load_jit, make_cpp_args

if TYPE_CHECKING:
    import torch
    from tvm_ffi import Module

DEFAULT_INDEX_KERNEL_CONFIG = KernelConfig(num_threads=128, max_occupancy=1, use_pdl=False)


@functools.cache
def _jit_index_module(
    element_size: int,
    *,
    num_splits: int = 1,
    config: KernelConfig = DEFAULT_INDEX_KERNEL_CONFIG,
) -> Module:
    args = make_cpp_args(element_size, num_splits, *config)
    return load_jit(
        "index",
        *args,
        cuda_files=["index.cu"],
        cuda_wrappers=[("launch", f"IndexKernel<{args}>::run")],
    )


def num_splits_for(element_size: int) -> int:
    """Split factor for a row of ``element_size`` bytes; also used by the AOT
    shape table (kernel/aot_models.py), which must reproduce it exactly."""
    if element_size % 2048 == 0:
        return 4
    if element_size % 1024 == 0:
        return 2
    return 1


def indexing(
    weights: torch.Tensor,
    indices: torch.Tensor,
    *,
    output: torch.Tensor | None = None,
    vocab_range: Tuple[int, int] | None = None,  # (start, length)
) -> torch.Tensor:
    if output is None:
        output = weights.new_empty(indices.shape[0], weights.shape[1])

    # JIT 路径：优先尝试编译内核
    element_size = weights.shape[1] * weights.element_size()
    jit_error: str | None = None
    try:
        module = _jit_index_module(element_size, num_splits=num_splits_for(element_size))
        module.launch(weights, indices, output, vocab_range)
        return output
    except Exception as ex:
        jit_error = str(ex)

    # JIT 失败时优雅降级：纯 PyTorch（Fallback，eager 模式）
    import torch as _torch
    idx = indices.to(_torch.long)
    if vocab_range is not None:
        start, length = vocab_range
        w = weights[start:start + length]
        idx_shifted = idx - start
        valid = (idx_shifted >= 0) & (idx_shifted < length)
        idx_clamped = idx_shifted.clamp(0, length - 1)
        gathered = w.index_select(0, idx_clamped.view(-1))
        mask = valid.view(-1).to(gathered.dtype)
        gathered = gathered * mask
    else:
        gathered = weights.index_select(0, idx.view(-1))
    gathered = gathered.view(indices.shape[0], weights.shape[1])
    output.copy_(gathered)
    return output
