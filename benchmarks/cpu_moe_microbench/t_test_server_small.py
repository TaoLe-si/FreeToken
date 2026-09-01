"""Test iGPU server with small M=1 to debug protocol."""
import subprocess, struct, time, sys, os, torch, safetensors.torch, numpy as np

DIR = r"E:\\FreeToken\\benchmarks\\cpu_moe_microbench"
SERVER = os.path.join(DIR, "t_mxfp4_gemv_server.exe")
MODEL_DIR = r"E:\\models\\Qwen3.6-35B-A3B-MXFP4-MTP"

if __name__ == "__main__":
    # Use 1 expert, 1 K/8 row only, so send small data
    # Just test with FC K=4096, M=1 to verify protocol
    M = 1
    K = 4096
    # Load FC row 0
    state = safetensors.torch.load_file(os.path.join(MODEL_DIR, "model-00022-of-00023.safetensors"))
    fc_packed = state["mtp.fc.weight"][0:1].contiguous()  # [1, 512]
    fc_scales = state["mtp.fc.scales"][0:1].contiguous()  # [1, 128]
    fc_biases = state["mtp.fc.biases"][0:1].contiguous()  # [1, 128]
    act = torch.randn(K, dtype=torch.float32) * 0.1

    print(f"sending M={M}, K={K}, sizes: p={fc_packed.numel()*4}, s={fc_scales.numel()*2}, b={fc_biases.numel()*2}, a={K*4}")

    proc = subprocess.Popen([SERVER], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=DIR)
    import threading
    def drain():
        while True:
            l = proc.stderr.readline()
            if not l: break
            print("S:", l.decode().strip(), file=sys.stderr)
    threading.Thread(target=drain, daemon=True).start()
    time.sleep(2.0)

    sz_packed = fc_packed.numel() * 4
    sz_scales = fc_scales.numel() * 2
    sz_biases = fc_biases.numel() * 2
    sz_act = K * 4
    hdr = struct.pack("<IIIII", M, K, sz_packed, sz_scales, sz_biases)
    print(f"hdr: {hdr.hex()}")
    print(f"writing hdr ({len(hdr)} bytes)")
    proc.stdin.write(hdr); proc.stdin.flush()
    print(f"writing packed ({sz_packed} bytes)")
    proc.stdin.write(fc_packed.view(torch.uint8).contiguous().numpy().tobytes()); proc.stdin.flush()
    print(f"writing scales ({sz_scales} bytes)")
    proc.stdin.write(fc_scales.view(torch.uint8).contiguous().numpy().tobytes()); proc.stdin.flush()
    print(f"writing biases ({sz_biases} bytes)")
    proc.stdin.write(fc_biases.view(torch.uint8).contiguous().numpy().tobytes()); proc.stdin.flush()
    print(f"writing act ({sz_act} bytes)")
    proc.stdin.write(act.numpy().tobytes()); proc.stdin.flush()
    print(f"all sent, waiting for response...")
    try:
        resp_len_bytes = proc.stdout.read(4)
        print(f"got {len(resp_len_bytes)} bytes for response length")
        if len(resp_len_bytes) < 4:
            print("FAIL: no response")
        else:
            resp_len = struct.unpack("<I", resp_len_bytes)[0]
            print(f"response length: {resp_len} (expected {M*4})")
            resp = proc.stdout.read(resp_len)
            print(f"got {len(resp)} bytes for response data")
            if len(resp) == resp_len:
                outv = np.frombuffer(resp, dtype=np.float32).copy()
                print(f"GPU outv: {outv}")
    except Exception as e:
        print(f"Exception: {e}")
    proc.terminate()
    proc.wait(timeout=5)