"""Compare D3D12 MXFP4 GEMV output to CPU PyTorch reference."""
import numpy as np
import os

base = os.path.dirname(os.path.abspath(__file__))

# Read D3D12 output
d3d_out = np.fromfile(os.path.join(base, "t_mxfp4_gemv_output.bin"), dtype=np.float32)
print(f"D3D12 out shape = {d3d_out.shape}")
print(f"D3D12 out[:5]   = {d3d_out[:5]}")
print(f"D3D12 out[-5:]  = {d3d_out[-5:]}")

# Read reference output (saved by t_mxfp4_gemv_reference.py)
ref = np.load(os.path.join(base, "t_mxfp4_ref_output.npy"))
print(f"ref shape = {ref.shape}")
print(f"ref[:5]   = {ref[:5]}")
print(f"ref[-5:]  = {ref[-5:]}")

# Compare
abs_diff = np.abs(d3d_out - ref)
rel_diff = abs_diff / (np.abs(ref) + 1e-6)
print(f"\n=== Numerical comparison ===")
print(f"max abs diff   = {abs_diff.max():.6e}")
print(f"mean abs diff  = {abs_diff.mean():.6e}")
print(f"max rel diff   = {rel_diff.max():.6e}")
print(f"mean rel diff  = {rel_diff.mean():.6e}")

# Relative tolerance check
tol = 1e-3  # 0.1%
ok = rel_diff.max() < tol
print(f"\n{'PASS' if ok else 'FAIL'} (rel tol {tol:.0e})")

# Check that not all zeros (sanity)
nz = np.count_nonzero(d3d_out)
print(f"D3D12 nonzero count = {nz}/{d3d_out.size}")
