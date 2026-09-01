import sys, torch
sys.path.insert(0, "E:/FreeToken/python")
import safetensors.torch as st
state = st.load_file("E:/models/Qwen3.6-35B-A3B-MXFP4-MTP-NVFP4/mtp.safetensors")
print("keys:", len(state))
for k in sorted(state):
    t = state[k]
    print(f"  {k}: {tuple(t.shape)} {t.dtype}")
fcw = state.get("mtp.fc.weight")
fcs = state.get("mtp.fc.scales")
fcb = state.get("mtp.fc.biases")
if fcw is not None:
    print("fc.weight uint32 words:", fcw.shape, "sample:", hex(int(fcw[0,0].item() & 0xFFFFFFFF)))
    if fcs is not None:
        print("fc.scales sample:", fcs[0,:4].tolist(), "absmax:", fcs.abs().max().item())
    if fcb is not None:
        print("fc.biases sample:", fcb[0,:4].tolist(), "absmax:", fcb.abs().max().item())
