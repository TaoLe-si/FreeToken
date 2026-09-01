"""Simplified: use fc_w[0] real data + file act."""
import subprocess
import struct
import time
import os
import numpy as np
import json, safetensors.torch
import torch

base = "E:\\FreeToken\\benchmarks\\cpu_moe_microbench"
exe = os.path.join(base, "t_mxfp4_gemv_v3_server.exe")

# Load real fc_w[0], fc_b[0], fc_s[0]
mdl = "E:\\models\\Qwen3.6-35B-A3B-MXFP4-MTP"
with open(os.path.join(mdl, "model.safetensors.index.json")) as f: idx = json.load(f)
state = safetensors.torch.load_file(os.path.join(mdl, idx["weight_map"]["mtp.fc.weight"]))
fc_w = state["mtp.fc.weight"].cpu().numpy()  # (M, K/8)
fc_b = state["mtp.fc.biases"].cpu().numpy().astype(np.float32)
fc_s = state["mtp.fc.scales"].cpu().numpy().astype(np.float32)

# Use row 0
packed = fc_w[0:1].copy()  # (1, K/8)
scl = fc_s[0:1].copy()  # (1, K/32)
bias = fc_b[0:1].copy()  # (1, K/32)

# Load act from file
with open(os.path.join(base, "t_mtp_fc_with_act.bin"), "rb") as f:
    M_, K_, nb_, ns_ = struct.unpack("IIII", f.read(16))
    f.read(M_ * nb_ * 4)
    f.read(M_ * ns_ * 4)
    f.read(M_ * ns_ * 4)
    act = np.frombuffer(f.read(K_ * 4), dtype=np.float32)
act_float = act.astype(np.float32)

M, K = 1, 4096
szP = M * (K // 8) * 4
szS = M * (K // 32) * 4
szA = K * 4
szB = M * (K // 32) * 4

cmd = ("STATELESS %d %d %d %d %d %d\n" % (M, K, szP, szS, szA, szB)).encode()
body = packed.tobytes() + scl.tobytes() + act_float.tobytes() + bias.tobytes()

print(f"M={M} K={K} szP={szP} szS={szS} szA={szA} szB={szB}")
print(f"packed.shape={packed.shape} scl.shape={scl.shape} bias.shape={bias.shape}")

proc = subprocess.Popen([exe], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=base, bufsize=0)
import threading
def read_stderr():
    while True:
        l = proc.stderr.readline()
        if not l: break
        print("STDERR:", l.decode(errors='replace').rstrip())
t = threading.Thread(target=read_stderr, daemon=True)
t.start()
time.sleep(1.5)
proc.stdin.write(cmd)
proc.stdin.write(body)
proc.stdin.flush()
rl = proc.stdout.read(4)
sz = struct.unpack('<I', rl)[0]
print(f"response sz={sz}")
data = proc.stdout.read(sz)
outv = np.frombuffer(data, dtype=np.float32).copy()
proc.stdin.write(b"QUIT\n")
proc.stdin.flush()
time.sleep(2.0)
try: proc.kill()
except: pass
time.sleep(0.5)
print(f"\noutv = {outv}")
print(f"expected = -1.7111")
