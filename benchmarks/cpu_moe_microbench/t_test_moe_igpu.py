"""Test generic iGPU MXFP4 GEMV server: MoE gate."""
import subprocess, struct, time, sys, os, torch, safetensors.torch, numpy as np

DIR = r"E:\\FreeToken\\benchmarks\\cpu_moe_microbench"
SERVER = os.path.join(DIR, "t_mxfp4_gemv_server.exe")
MODEL_DIR = r"E:\\models\\Qwen3.6-35B-A3B-MXFP4-MTP"

def call_gpu_server(packed, scales, biases, act, M, K):
    proc = subprocess.Popen([SERVER], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=DIR)
    import threading
    def drain():
        while True:
            l = proc.stderr.readline()
            if not l: break
            print("S:", l.decode().strip(), file=sys.stderr)
    threading.Thread(target=drain, daemon=True).start()
    time.sleep(2.0)
    sz_packed = packed.numel() * 4
    sz_scales = scales.numel() * 2
    sz_biases = biases.numel() * 2
    hdr = struct.pack("<IIIII", M, K, sz_packed, sz_scales, sz_biases)
    proc.stdin.write(hdr)
    proc.stdin.write(packed.view(torch.uint8).contiguous().numpy().tobytes())
    proc.stdin.write(scales.view(torch.uint8).contiguous().numpy().tobytes())
    proc.stdin.write(biases.view(torch.uint8).contiguous().numpy().tobytes())
    proc.stdin.write(act.numpy().tobytes())
    proc.stdin.flush()
    try:
        resp_len_bytes = proc.stdout.read(4)
        if len(resp_len_bytes) < 4:
            print(f"  GPU server: read returned {len(resp_len_bytes)} bytes")
            proc.terminate()
            return None
        resp_len = struct.unpack("<I", resp_len_bytes)[0]
        resp = proc.stdout.read(resp_len)
        outv = np.frombuffer(resp, dtype=np.float32).copy()
        proc.terminate()
        proc.wait(timeout=5)
        return outv
    except Exception as e:
        print(f"  GPU server exception: {e}")
        proc.terminate()
        return None

if __name__ == "__main__":
    base = MODEL_DIR
    state = safetensors.torch.load_file(os.path.join(base, "model-00022-of-00023.safetensors"))
    gate_packed = state["mtp.layers.0.mlp.switch_mlp.gate_proj.weight"]
    gate_scales = state["mtp.layers.0.mlp.switch_mlp.gate_proj.scales"]
    gate_biases = state["mtp.layers.0.mlp.switch_mlp.gate_proj.biases"]

    # Use 8 active experts
    M = 8
    K = 2048
    packed = gate_packed[:M].contiguous()
    scales = gate_scales[:M].contiguous()
    biases = gate_biases[:M].contiguous()
    act = torch.randn(K, dtype=torch.float32) * 0.1

    # Reshape for the kernel: 2D [M, K/8]
    packed_2d = packed.reshape(M, -1).contiguous()
    scales_2d = scales.reshape(M, -1).contiguous()
    biases_2d = biases.reshape(M, -1).contiguous()

    print(f"packed_2d: {tuple(packed_2d.shape)}, scales_2d: {tuple(scales_2d.shape)}, biases_2d: {tuple(biases_2d.shape)}")
    print(f"sz_packed={packed_2d.numel() * 4}, sz_scales={scales_2d.numel() * 2}, sz_biases={biases_2d.numel() * 2}")

    outv = call_gpu_server(packed_2d, scales_2d, biases_2d, act, M, K)
    if outv is not None:
        print(f"GPU result shape: {outv.shape}, norm: {np.linalg.norm(outv):.4f}")
    else:
        print("GPU server returned no result")