"""Run MXFP4 GEMV reference using inputs from D3D12 cpp."""
import numpy as np
import os

base = os.path.dirname(os.path.abspath(__file__))
sys_path = os.path.join(base, "t_mxfp4_gemv_reference.py")
import sys
sys.path.insert(0, base)
from t_mxfp4_gemv_reference import mxfp4_gemv_reference

with open(os.path.join(base, "t_mxfp4_inputs.bin"), "rb") as f:
    magic = f.read(4)
    assert magic == b"PK10", f"bad magic: {magic}"
    M, K = np.frombuffer(f.read(8), dtype=np.uint32)
    M, K = int(M), int(K)
    nbPerRow = K // 8
    nsPerRow = K // 32
    packed = np.frombuffer(f.read(M * nbPerRow * 4), dtype=np.uint32).reshape(M, nbPerRow)
    scl    = np.frombuffer(f.read(M * nsPerRow * 4), dtype=np.uint32).reshape(M, nsPerRow)
    act    = np.frombuffer(f.read(K), dtype=np.int8)
    bias   = np.frombuffer(f.read(M * 4), dtype=np.float32)
    gbl    = np.frombuffer(f.read(M * 4), dtype=np.float32)

print(f"M={M} K={K}")
outv = mxfp4_gemv_reference(packed, scl, act, bias, gbl)
print(f"ref[:5] = {outv[:5]}")
print(f"ref[-5:] = {outv[-5:]}")

# Compare with D3D12 output
d3d_out = np.fromfile(os.path.join(base, "t_mxfp4_gemv_output.bin"), dtype=np.float32)
print(f"\nD3D12 out[:5]  = {d3d_out[:5]}")
print(f"D3D12 out[-5:] = {d3d_out[-5:]}")

abs_diff = np.abs(d3d_out - outv)
rel_diff = abs_diff / (np.abs(outv) + 1e-6)
print(f"\n=== Numerical comparison ===")
print(f"max abs diff  = {abs_diff.max():.6e}")
print(f"mean abs diff = {abs_diff.mean():.6e}")
print(f"max rel diff  = {rel_diff.max():.6e}")
print(f"mean rel diff = {rel_diff.mean():.6e}")

tol = 1e-2
ok = (rel_diff.max() < tol) and (abs_diff.max() < 1e-1)
print(f"\n{'PASS' if ok else 'FAIL'} (rel tol {tol:.0e}, abs tol 1e-1)")

np.save(os.path.join(base, "t_mxfp4_ref_output.npy"), outv)
