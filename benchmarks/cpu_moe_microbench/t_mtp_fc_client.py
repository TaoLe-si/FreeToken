"""Persistent D3D12 MTP fc client - benchmarks per-call latency."""
import subprocess
import struct
import time
import os
import sys
import numpy as np

DIR = "E:\\FreeToken\\benchmarks\\cpu_moe_microbench"
SERVER = os.path.join(DIR, "t_mtp_fc_server.exe")
WEIGHTS = os.path.join(DIR, "t_mtp_fc_weights.bin")

K = 4096
M = 1

# Start server
print("Starting server...")
proc = subprocess.Popen(
    [SERVER, WEIGHTS, str(M)],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    cwd=DIR
)
import threading
def drain_stderr():
    while True:
        line = proc.stderr.readline()
        if not line: break
        print("server:", line.decode().strip(), file=sys.stderr)
t = threading.Thread(target=drain_stderr, daemon=True)
t.start()

# Wait for "Ready." message
time.sleep(2.0)
print(f"server PID: {proc.pid}")

# Send a few test acts and time them
rng = np.random.default_rng(42)
for i in range(5):
    act = rng.normal(0, 1, size=K).astype(np.float32)
    act_bytes = act.tobytes()
    t0 = time.perf_counter()
    proc.stdin.write(struct.pack("<I", len(act_bytes)))
    proc.stdin.write(act_bytes)
    proc.stdin.flush()
    # Read response
    resp_len_bytes = proc.stdout.read(4)
    if len(resp_len_bytes) < 4:
        print(f"  iter {i}: server closed")
        break
    resp_len = struct.unpack("<I", resp_len_bytes)[0]
    resp = proc.stdout.read(resp_len)
    outv = np.frombuffer(resp, dtype=np.float32)
    t1 = time.perf_counter()
    print(f"  iter {i}: roundtrip {(t1-t0)*1000:.2f}ms outv={outv.tolist()}")

proc.terminate()
proc.wait(timeout=5)
print("done")
