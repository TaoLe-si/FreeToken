import sys, time
sys.path.insert(0, "E:/FreeToken/python")
import numpy as np
from freetoken.kernel.igpu_fc import IgpuFcSticky

M, K = 2048, 4096
packed = np.zeros((M, K // 8), dtype=np.uint32)
scales = np.ones((M, K // 32), dtype=np.float32)
biases = np.zeros((M, K // 32), dtype=np.float32)
t0 = time.time()
print("constructing sticky...", flush=True)
sticky = IgpuFcSticky(packed, K, scales_f32=scales, biases_f32=biases)
print(f"sticky OK in {time.time()-t0:.1f}s", flush=True)
act = np.ones(K, dtype=np.float32)
out = sticky(act)
print("call OK:", out.shape, flush=True)
sticky.close()
