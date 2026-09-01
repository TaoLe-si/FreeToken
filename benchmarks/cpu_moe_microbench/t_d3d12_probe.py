import comtypes
comtypes._check_version = lambda *a, **k: None
import pyd3d12
from pyd3d12 import D3D12
dev = pyd3d12.D3D12CreateDevice(None, D3D12.D3D_FEATURE_LEVEL_12_0)
print("device:", type(dev))
print("methods:", [m for m in dir(dev) if not m.startswith("_")][:30])
