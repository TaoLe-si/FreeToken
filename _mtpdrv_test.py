import sys, time, threading
sys.path.insert(0, "E:/FreeToken/python")
import numpy as np, torch
import safetensors.torch as st

state = st.load_file("E:/models/Qwen3.6-35B-A3B-MXFP4-MTP-NVFP4/mtp.safetensors")
fcw = state["mtp.fc.weight"].numpy().astype("uint32")
fcs = state["mtp.fc.scales"].numpy().astype("float32")
fcb = state["mtp.fc.biases"].numpy().astype("float32")
M, nb = fcw.shape
K = nb * 8
print(f"fc: M={M} K={K}", flush=True)

from freetoken.kernel.igpu_fc import IgpuFcSticky
t0 = time.time()
done = []
def build():
    try:
        s = IgpuFcSticky(fcw, K, scales_f32=fcs, biases_f32=fcb)
        out = s(np.ones(K, dtype=np.float32))
        done.append(("ok", out.shape, float(time.time() - t0)))
        s.close()
    except Exception as e:
        done.append(("err", repr(e), float(time.time() - t0)))
th = threading.Thread(target=build, daemon=True)
th.start()
th.join(timeout=45)
if done:
    print("RESULT:", done[0], flush=True)
else:
    print("HUNG >45s (sticky construction blocks with real weights)", flush=True)
