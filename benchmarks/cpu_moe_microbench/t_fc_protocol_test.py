"""FC_LOAD/FC_CALL 安全测试: 先 M=16 验证协议, 再 M=2048 全量.
只清理自己的子进程 (PID), 绝不 taskkill 全局 python.exe."""
import subprocess, struct, time, threading, sys
import numpy as np
import json, safetensors.torch

bench = r"E:\FreeToken\benchmarks\cpu_moe_microbench"
exe = bench + r"\t_mxfp4_gemv_v3_server.exe"
mdl = r"E:\models\Qwen3.6-35B-A3B-MXFP4-MTP"

with open(mdl + "/model.safetensors.index.json") as f: idx = json.load(f)
st = safetensors.torch.load_file(mdl + "/" + idx["weight_map"]["mtp.fc.weight"])
fc_w_all = st["mtp.fc.weight"].cpu().numpy()                     # (2048, 512)
fc_b_all = st["mtp.fc.biases"].cpu().numpy().astype(np.float32)  # (2048, 128)
fc_s_all = st["mtp.fc.scales"].cpu().numpy().astype(np.float32)  # (2048, 128)
K = 4096; nb, ns = K // 8, K // 32

kE2M1 = np.array([0,1,2,3,4,6,8,12, 0,-1,-2,-3,-4,-6,-8,-12], dtype=np.float32)
def ref_gemv(packed, scl, bias, act):
    """向量化 numpy 参考: packed (M,nb) scl/bias (M,ns) act (K,)"""
    M = packed.shape[0]
    w = packed[:, :, None]
    shifts = (np.arange(8, dtype=np.uint32) * 4)[None, None, :]
    nibs = ((w >> shifts) & 0xF).astype(np.int64)
    vals = kE2M1[nibs].reshape(M, K)
    prod = (vals * act[None, :]).reshape(M, ns, 32)
    return ((prod.sum(axis=2) + bias) * scl).sum(axis=1).astype(np.float32)

def read_exact(fp, n, what):
    b = fp.read(n)
    if len(b) < n:
        raise RuntimeError(f"short read on {what}: got {len(b)}/{n} (server died?)")
    return b

proc = None
try:
    proc = subprocess.Popen([exe], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, cwd=bench, bufsize=0)
    errs = []
    def rd():
        while True:
            l = proc.stderr.readline()
            if not l: break
            errs.append(l.decode(errors="replace").rstrip())
    threading.Thread(target=rd, daemon=True).start()
    time.sleep(2.0)
    if proc.poll() is not None:
        raise RuntimeError(f"server died at startup: {errs[-5:]}")

    def fc_load(M, fc_w, fc_s, fc_b):
        szP, szS, szB = M*nb*4, M*ns*4, M*ns*4
        proc.stdin.write(f"FC_LOAD {M} {K} {szP} {szS} {szB}\n".encode())
        proc.stdin.write(fc_w.tobytes() + fc_s.tobytes() + fc_b.tobytes())
        proc.stdin.flush()
        ok = read_exact(proc.stdout, 3, "FC_LOAD ack")
        if ok != b"OK\n": raise RuntimeError(f"FC_LOAD bad ack: {ok!r}")

    def fc_call(act):
        proc.stdin.write(f"FC_CALL {K*4}\n".encode())
        proc.stdin.write(act.tobytes())
        proc.stdin.flush()
        rl = read_exact(proc.stdout, 4, "FC_CALL len")
        sz = struct.unpack("<I", rl)[0]
        return np.frombuffer(read_exact(proc.stdout, sz, "FC_CALL data"), dtype=np.float32).copy()

    rng = np.random.default_rng(42)
    act = rng.standard_normal(K).astype(np.float32)

    # ===== 阶段 1: M=16 协议+数值验证 =====
    print("===== Stage 1: M=16 =====")
    M1 = 16
    fc_load(M1, fc_w_all[:M1], fc_s_all[:M1], fc_b_all[:M1])
    ref1 = ref_gemv(fc_w_all[:M1], fc_s_all[:M1], fc_b_all[:M1], act)
    out1 = fc_call(act)
    d1 = np.abs(out1 - ref1)
    print(f"  out[0..3]={out1[:4]}")
    print(f"  ref [0..3]={ref1[:4]}")
    print(f"  max|diff|={d1.max():.3e}  {'PASS' if d1.max() < 1e-3 else 'FAIL'}")
    if d1.max() >= 1e-3 or np.isnan(out1).any():
        print("Stage1 FAIL — 停止"); sys.exit(1)
    # 第二次调用 (验证状态机复用)
    out1b = fc_call(act)
    print(f"  2nd call max|diff|={np.abs(out1b - ref1).max():.3e}  {'PASS' if np.abs(out1b - ref1).max() < 1e-3 else 'FAIL'}")

    # ===== 阶段 2: M=2048 全量 =====
    print("\n===== Stage 2: M=2048 full =====")
    M2 = 2048
    t0 = time.time()
    fc_load(M2, fc_w_all, fc_s_all, fc_b_all)
    print(f"  FC_LOAD: {time.time()-t0:.2f}s (含 {M2*nb*4/1048576:.1f}MB 权重上传)")
    ref2 = ref_gemv(fc_w_all, fc_s_all, fc_b_all, act)
    out2 = fc_call(act)
    d2 = np.abs(out2 - ref2)
    nan_cnt = int(np.isnan(out2).sum())
    print(f"  out[0..4]={out2[:5]}")
    print(f"  ref [0..4]={ref2[:5]}")
    print(f"  max|diff|={d2.max():.3e} mean={d2.mean():.3e} NaN数={nan_cnt}")
    print(f"  判定: {'PASS' if d2.max() < 1e-3 and nan_cnt == 0 else 'FAIL'}")

    # 延迟测试 (30 次)
    ts = []
    for _ in range(30):
        t0 = time.time()
        fc_call(act)
        ts.append((time.time()-t0)*1000)
    print(f"\n  连续 30 次 FC_CALL(M=2048): median={np.median(ts):.2f}ms p90={np.percentile(ts,90):.2f}ms max={max(ts):.2f}ms")

    proc.stdin.write(b"QUIT\n"); proc.stdin.flush()
    time.sleep(0.5)
    print("\nserver 日志 (最后 8 行):")
    for l in errs[-8:]: print("  ", l)
    print("\n=== ALL DONE ===")
finally:
    if proc is not None:
        try:
            if proc.poll() is None:
                proc.kill()   # 只杀自己的子进程 (含 server, 同一进程组)
        except Exception:
            pass
