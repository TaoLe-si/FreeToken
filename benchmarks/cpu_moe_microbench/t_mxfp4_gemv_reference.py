"""MXFP4 GEMV CPU PyTorch reference for numerical validation of D3D12 MXFP4 GEMV kernel.

Layout matches E:\FreeToken\benchmarks\cpu_moe_microbench\t_mxfp4_gemv_sk.hlsl:
  packed : uint32 [M, K/8], each uint = 4 bytes = 4 nibbles (8 nibbles/uint actually)
           Wait — let me re-derive. Each K-element is 4-bit (e2m1). Packed bytewise:
           2 elements per byte. So 1 uint = 4 bytes = 8 K-elements. packed[M, K/8] uint.
  scl    : uint32 [M, K/32], each uint = 4 e8m0 bytes. Each micro-block (32 K-elements)
           gets one e8m0 scale byte. K/32 micro-blocks per row, 4 bytes per uint ->
           K/32 uint per row. matches.
  act    : int8 [K]
  bias   : float32 [M]
  gbl    : float32 [M]
  outv   : float32 [M], outv[r] = (sum_k W[r,k]*act[k] + bias[r]) * gbl[r]
"""
from __future__ import annotations
import numpy as np
import torch
import struct, os, sys

kE2M1 = np.array([0,1,2,3,4,6,8,12, 0,-1,-2,-3,-4,-6,-8,-12], dtype=np.int32)


def _unpack_packed_row(packed_row_u32: np.ndarray, K: int) -> np.ndarray:
    """Unpack one row's packed uint32s into int32 weights [K] via kE2M1 LUT.
    Each uint32 = 4 bytes = 8 nibbles (little-endian, byte 0 has the lower 2 nibbles).
    Element n is in uint n//8, byte (n//2)%4, bit (n%2)*4..(n%2)*4+3.
    """
    packed_bytes = packed_row_u32.view(np.uint8)  # flat [K/8 * 4] bytes
    n = np.arange(K, dtype=np.int64)
    uint_idx = n // 8
    byte_idx = (n // 2) % 4
    bit = (n % 2) * 4
    flat = uint_idx * 4 + byte_idx
    b = packed_bytes[flat]
    nibble = (b >> bit.astype(np.uint8)) & np.uint8(0xF)
    return kE2M1[nibble.astype(np.int64)]


def _unpack_scl_row(scl_row_u32: np.ndarray, K: int) -> np.ndarray:
    """Unpack one row's scl uint32s into float32 scales [K/32].

    K/32 micro-blocks per row, packed as (K/32)/4 uints each holding 4 e8m0 bytes.
    Micro-block m's byte is at flat byte index: u*4 + b where u=m//4, b=m%4.
    Flat byte index into a (K/32 * 4) byte array = u*4 + b = (m//4)*4 + (m%4) = m + 3*(m//4).
    Equivalent simpler form: the bytes are stored in little-endian uint32 order:
    byte 0 of uint 0 = micro-block 0, byte 1 of uint 0 = micro-block 1, etc.
    """
    nb = K // 32  # number of micro-blocks per row
    scl_bytes = scl_row_u32.view(np.uint8)  # [nb * 4] bytes flat
    # build index: micro-block m -> flat byte index
    m = np.arange(nb)
    flat_idx = m // 4 * 4 + (m % 4)
    sb = scl_bytes[flat_idx].astype(np.int32)
    return np.where(sb == 0, 0.0, np.exp2(sb.astype(np.float32) - 127.0)).astype(np.float32)


def mxfp4_gemv_reference(packed, scl, act, bias, gbl):
    """packed [M, K/8] uint32, scl [M, K/32] uint32, act [K] int8, bias [M] f32, gbl [M] f32 -> outv [M] f32."""
    M, K = packed.shape[0], act.shape[0]
    assert packed.shape == (M, K // 8), f"packed shape {packed.shape} != ({M}, {K//8})"
    assert scl.shape == (M, K // 32), f"scl shape {scl.shape} != ({M}, {K//32})"
    assert bias.shape == (M,) and gbl.shape == (M,)

    outv = np.zeros(M, dtype=np.float32)
    for r in range(M):
        W = _unpack_packed_row(packed[r], K)         # [K] int32
        S = _unpack_scl_row(scl[r], K)              # [K/32] float32
        # broadcast S to [K]
        S_full = np.repeat(S, 32)
        W_scaled = W.astype(np.float32) * S_full
        acc = (W_scaled * act.astype(np.float32)).sum()
        outv[r] = (acc + bias[r]) * gbl[r]
    return outv


def gen_random_inputs(M: int, K: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    # packed: each uint = 4 random bytes = 8 random nibbles. Sample uint32 uniformly.
    packed = rng.integers(0, 2**32, size=(M, K // 8), dtype=np.uint32)
    # scl: each uint = 4 e8m0 bytes around 127
    scl_bytes = np.clip(rng.normal(127, 5, size=(M, K // 32, 4)).astype(np.int32), 100, 154)
    scl = np.zeros((M, K // 32), dtype=np.uint32)
    for b in range(4):
        scl |= (scl_bytes[:, :, b].astype(np.uint32) << (b * 8))
    # act: random int8
    act = rng.integers(-128, 128, size=(K,), dtype=np.int8)
    bias = rng.uniform(-0.5, 0.5, size=(M,)).astype(np.float32)
    gbl = rng.uniform(0.5, 2.0, size=(M,)).astype(np.float32)
    return packed, scl, act, bias, gbl


def dump_inputs(packed, scl, act, bias, gbl, outv, path):
    """Save inputs + reference output to a binary file for the cpp host to consume/compare."""
    with open(path, 'wb') as f:
        for arr in (packed, scl, act, bias, gbl, outv):
            f.write(np.ascontiguousarray(arr).tobytes())


if __name__ == "__main__":
    M, K = 2048, 4096
    print(f"M={M} K={K}")
    packed, scl, act, bias, gbl = gen_random_inputs(M, K)
    outv = mxfp4_gemv_reference(packed, scl, act, bias, gbl)
    print(f"outv[:5]   = {outv[:5].tolist()}")
    print(f"outv[-5:]  = {outv[-5:].tolist()}")
    print(f"max abs    = {np.abs(outv).max():.4f}")
    print(f"mean abs   = {np.abs(outv).mean():.4f}")
    print(f"finite all = {bool(np.isfinite(outv).all())}")
    assert outv.shape == (M,), outv.shape
    assert outv.dtype == np.float32, outv.dtype
    assert np.isfinite(outv).all()
    out_path = os.path.join(os.path.dirname(__file__), "t_mxfp4_ref_output.npy")
    np.save(out_path, outv)
    print(f"saved reference outv -> {out_path}")
