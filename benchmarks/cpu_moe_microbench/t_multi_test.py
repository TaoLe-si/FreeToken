"""MULTI_GEMV batch verification: B rows in one dispatch."""
import subprocess, struct, time, os
import numpy as np
import json, safetensors.torch

base = r"E:\FreeToken\benchmarks\cpu_moe_microbench"
exe = os.path.join(base, "t_mxfp4_gemv_v3_server.exe")

mdl = r"E:\models\Qwen3.6-35B-A3B-MXFP4-MTP"
with open(os.path.join(mdl, "model.safetensors.index.json")) as f:
    idx = json.load(f)
state = safetensors.torch.load_file(os.path.join(mdl, idx["weight_map"]["mtp.fc.weight"]))
fc_w = state["mtp.fc.weight"].cpu().numpy()      # (M, K/8) uint32
fc_b = state["mtp.fc.biases"].cpu().numpy().astype(np.float32)   # (M, K/32)
fc_s = state["mtp.fc.scales"].cpu().numpy().astype(np.float32)   # (M, K/32)

B, K = 8, 4096
nb, ns = K // 8, K // 32

packed = fc_w[0:B].copy()      # (B, 512)
scl    = fc_s[0:B].copy()      # (B, 128)
bias   = fc_b[0:B].copy()      # (B, 128)

# item 0 用文件 act (已知 -> -1.7111), 其余用随机 act
with open(os.path.join(base, "t_mtp_fc_with_act.bin"), "rb") as f:
    struct.unpack("IIII", f.read(16))
    f.read(B * 0)
    f.read(1 * nb * 4); f.read(1 * ns * 4); f.read(1 * ns * 4)
    act0 = np.frombuffer(f.read(K * 4), dtype=np.float32)

rng = np.random.default_rng(1234)
acts = np.zeros((B, K), dtype=np.float32)
acts[0] = act0
for i in range(1, B):
    acts[i] = rng.standard_normal(K).astype(np.float32)

# ---- numpy 参考实现 ----
kE2M1 = np.array([0,1,2,3,4,6,8,12, 0,-1,-2,-3,-4,-6,-8,-12], dtype=np.float32)
def gemv_ref(prow, srow, brow, arow):
    total = 0.0
    for b in range(ns):
        wsum = 0.0
        for j in range(4):
            w = int(prow[b*4 + j])
            for k in range(8):
                wsum += float(kE2M1[(w >> (4*k)) & 0xF]) * float(arow[b*32 + j*8 + k])
        total += (wsum + float(brow[b])) * float(srow[b])
    return total

ref = np.array([gemv_ref(packed[i], scl[i], bias[i], acts[i]) for i in range(B)], dtype=np.float32)

# ---- 协议 ----
szPPer, szSPer, szAPer, szBPer, gblPer = nb*4, ns*4, K*4, ns*4, 4
cmd = ("MULTI_GEMV %d %d %d %d %d %d %d\n" % (B, K, szPPer, szSPer, szAPer, szBPer, gblPer)).encode()
body = b"".join(
    packed[i].tobytes() + scl[i].tobytes() + acts[i].tobytes() + bias[i].tobytes() + struct.pack("<f", 1.0)
    for i in range(B)
)
print(f"B={B} K={K} body={len(body)} bytes")

proc = subprocess.Popen([exe], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE, cwd=base, bufsize=0)
import threading
def drain():
    while True:
        l = proc.stderr.readline()
        if not l: break
print_stderr_lines = []
def read_stderr():
    while True:
        l = proc.stderr.readline()
        if not l: break
        print_stderr_lines.append(l.decode(errors="replace").rstrip())
t = threading.Thread(target=read_stderr, daemon=True); t.start()
time.sleep(2.0)

proc.stdin.write(cmd); proc.stdin.write(body); proc.stdin.flush()
sz = struct.unpack("<I", proc.stdout.read(4))[0]
outv = np.frombuffer(proc.stdout.read(sz), dtype=np.float32).copy()
proc.stdin.write(b"QUIT\n"); proc.stdin.flush()
time.sleep(1.0)
try: proc.kill()
except: pass

print(f"response sz={sz} ({sz//4} floats)")
print()
print(f"{'item':>4} {'GPU输出':>14} {'参考值':>14} {'误差':>10} 判定")
all_ok = True
for i in range(B):
    diff = abs(float(outv[i]) - float(ref[i]))
    ok = diff < 1e-4 and not np.isnan(outv[i])
    if not ok: all_ok = False
    tag = "OK" if ok else "FAIL"
    extra = "  (item0=已知-1.7111)" if i == 0 else ""
    print(f"{i:>4} {outv[i]:>14.7f} {ref[i]:>14.7f} {diff:>10.2e} {tag}{extra}")
print()
print("RESULT:", "ALL PASS" if all_ok else "FAILED")
