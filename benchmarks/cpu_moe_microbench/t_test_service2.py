
import subprocess, struct, numpy as np
M, K = 2048, 4096
NB = K // 16
rng = np.random.default_rng(42)
packed = rng.integers(0, 256, M * NB * 8, dtype=np.uint8)
scl = rng.integers(0, 128, M * NB, dtype=np.uint32)
act = rng.integers(-127, 128, NB * 16, dtype=np.int32)
asb = np.zeros(NB, dtype=np.float32)
gbl = np.ones(M, dtype=np.float32)
payload = struct.pack("<II", M, K) + packed.tobytes() + scl.tobytes() + act.tobytes() + asb.tobytes() + gbl.tobytes()
p = subprocess.Popen([r"t_d3d12_service.exe"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
raw = b""
try:
    p.stdin.write(payload); p.stdin.flush()
    raw = p.stdout.read(M * 4)
except BrokenPipeError:
    pass
p.stdin.close()
err = p.stderr.read().decode(errors="replace")
p.wait()
print("exit:", p.returncode, "stdout bytes:", len(raw))
print("stderr:", err)
