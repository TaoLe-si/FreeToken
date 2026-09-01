from __future__ import annotations

import torch

from .base import BaseMoeBackend


class IgpuMoeBackend(BaseMoeBackend):
    """Marker backend for iGPU (AMD D3D12) W4A8 expert decode.

    Like CpuOffloadMoeBackend, the real work lives in OffloadMoELayer:
    prefill streams whole expert layers into the GPU double buffer and runs the
    GEMM on the GPU (identical to offload), while decode ships the activations
    to the iGPU D3D12 compute service, which computes the experts straight from
    the pinned host banks (no PCIe round-trip, no CPU-core occupation; measured
    26-36 GB/s on AMD 780M vs 41-44 GB/s upload-read ceiling), and ships the
    results back to the GPU.

    This is the B-group compute path of the dense_host_offload architecture
    (docs/design/dense_host_offload.md sec 18.3): the iGPU complements the CPU
    executor for multi-request decode, and the two never contend for the same
    compute units.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
        activation: str,
        apply_router_weight_on_input: bool,
    ) -> torch.Tensor:
        raise RuntimeError("iGPU MoE is handled by OffloadMoELayer")
