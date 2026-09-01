import sys, time, threading
sys.path.insert(0, "E:/FreeToken/python")
import torch
from freetoken.kernel.triton.norm import gemma_fused_add_rmsnorm

x = torch.randn(4, 2048, device="cuda", dtype=torch.bfloat16)
r = torch.randn(4, 2048, device="cuda", dtype=torch.bfloat16)
w = torch.randn(2048, device="cuda", dtype=torch.bfloat16)
done = []
def go():
    try:
        t0 = time.time()
        gemma_fused_add_rmsnorm(x, r, w, 1e-6)
        torch.cuda.synchronize()
        done.append(("ok", round(time.time() - t0, 2)))
    except Exception as e:
        done.append(("err", repr(e)))
th = threading.Thread(target=go, daemon=True)
th.start(); th.join(timeout=60)
print("RESULT:", done or "HUNG >60s", flush=True)
