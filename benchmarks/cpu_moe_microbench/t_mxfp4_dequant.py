"""MXFP4 dequantization utilities matching the D3D12 kernel formula."""
from __future__ import annotations
import torch
import numpy as np

kE2M1 = np.array([0,1,2,3,4,6,8,12, 0,-1,-2,-3,-4,-6,-8,-12], dtype=np.int32)


def dequant_mxfp4_packed_row(packed_row_u32, K):
    """Unpack one row of MXFP4 packed uint32s into a signed int32 weight vector.
    Accepts torch.Tensor or numpy array of shape [K/8] uint32.
    Returns: [K] int32 (kE2M1 LUT applied to each nibble)."""
    if hasattr(packed_row_u32, 'numpy'):
        pb_view = packed_row_u32.view(torch.uint8)
    else:
        pb_view = packed_row_u32.view(np.uint8)
    pb_flat = pb_view.reshape(-1)  # [K/2] bytes
    n = np.arange(K, dtype=np.int64)
    uint_idx = n // 8
    byte_idx = (n // 2) % 4
    bit = (n % 2) * 4
    flat = uint_idx * 4 + byte_idx
    b = pb_flat[flat]
    if hasattr(b, 'numpy'):
        b_np = b.numpy()
    else:
        b_np = b
    nibble = (b_np >> bit.astype(np.uint8)) & np.uint8(0xF)
    return torch.from_numpy(kE2M1[nibble.astype(np.int64)].astype(np.int32))


def dequant_mxfp4_weight(packed: torch.Tensor, scales: torch.Tensor, biases: torch.Tensor, K: int) -> torch.Tensor:
    """Full MXFP4 dequant: (weight_nibble * scale + bias) per 32-element micro-block.
    packed: [M, K/8] uint32
    scales: [M, K/32] float (bf16 scales stored as float)
    biases: [M, K/32] float
    Returns: [M, K] float"""
    M = packed.shape[0]
    out = np.empty((M, K), dtype=np.float32)
    for r in range(M):
        W = dequant_mxfp4_packed_row(packed[r], K).numpy().astype(np.float32)
        S = scales[r].view(torch.uint8).numpy()  # NOT correct - let me redo
    raise NotImplementedError  # placeholder


# Better: keep scales/biases as flat uint8 view to get exact byte order matching D3D12 shader
def dequant_mxfp4_weight_v2(packed: torch.Tensor, scales: torch.Tensor, biases: torch.Tensor, K: int) -> torch.Tensor:
    """Full MXFP4 dequant matching the D3D12 shader byte ordering."""
    M = packed.shape[0]
    nb = K // 32  # micro-blocks per row
    out = np.empty((M, K), dtype=np.float32)
    for r in range(M):
        W = dequant_mxfp4_packed_row(packed[r], K).numpy().astype(np.float32)
        # scales: [nb] float. Per micro-block, one scale applied to 32 elements.
        S_row = scales[r].numpy().astype(np.float32)  # [nb]
        S_full = np.repeat(S_row, 32)  # [K]
        B_row = biases[r].numpy().astype(np.float32)  # [nb]
        B_full = np.repeat(B_row, 32)  # [K]
        out[r] = W * S_full + B_full
    return torch.from_numpy(out)


def dequant_mxfp4_expert_block(packed: torch.Tensor, scales: torch.Tensor, biases: torch.Tensor) -> torch.Tensor:
    """Dequant a single expert's weight from packed MXFP4 (used for MoE 256 experts).
    packed: [..., K/8] uint32 (last dim is the packed bytes per row)
    scales: [..., K/32] float
    biases: [..., K/32] float
    Returns: [..., K] float"""
    *packed_shape, K_div_8 = packed.shape
    K = K_div_8 * 8
    # We process last dim
    *batch, K_div_8 = packed.shape
    K = K_div_8 * 8
    # Unpack nibbles along the last dim
    packed_bytes = packed.view(torch.uint8)
    # Reshape to expose pairs of nibbles: [batch..., K/2, 2]
    n = torch.arange(K, dtype=torch.int64)
    uint_idx = n // 8
    byte_idx = (n // 2) % 4
    bit = ((n % 2) * 4).to(torch.uint8)
    flat = uint_idx * 4 + byte_idx
    # expand flat across batch dims: [..., K]
    b = torch.gather(packed_bytes, -1, flat.expand(*packed_bytes.shape[:-1], K))
    nibble = ((b >> bit) & 0xF).to(torch.int64)
    kE2M1_t = torch.tensor(kE2M1, dtype=torch.int32)
    W = kE2M1_t[nibble].to(torch.float32)  # [..., K]
    # Broadcast scales [..., K/32] to [..., K]
    S = scales.repeat_interleave(32, dim=-1)  # [..., K]
    B = biases.repeat_interleave(32, dim=-1)  # [..., K]
    return W * S + B


__all__ = ["kE2M1", "dequant_mxfp4_packed_row", "dequant_mxfp4_weight_v2", "dequant_mxfp4_expert_block"]