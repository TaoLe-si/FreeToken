# 780M vs 4070 via DirectML MatMul (B = weight-style big tensor in DRAM)
import numpy as np, time, sys
import onnxruntime as ort
from onnx import helper, TensorProto

K, N = 4096, 65536  # B: [K,N] fp16 = 512MB
A = np.random.rand(1, K).astype(np.float16)
B = np.random.rand(K, N).astype(np.float16)
Y = np.empty((1, N), dtype=np.float16)

node = helper.make_node("MatMul", ["A", "B"], ["Y"])
graph = helper.make_graph([node], "mm",
    [helper.make_tensor_value_info("A", TensorProto.FLOAT16, [1, K]),
     helper.make_tensor_value_info("B", TensorProto.FLOAT16, [K, N])],
    [helper.make_tensor_value_info("Y", TensorProto.FLOAT16, [1, N])])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
model.ir_version = 9
fname = "igpu_mm.onnx"
onnx_bytes = model.SerializeToString()
open(fname, "wb").write(onnx_bytes)

def bench(device_id):
    try:
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession(fname, so, providers=[("DmlExecutionProvider", {"device_id": device_id})])
    except Exception as e:
        return None, f"init failed: {str(e)[:150]}"
    # warmup
    sess.run(None, {"A": A, "B": B})
    best = float("inf")
    for _ in range(5):
        t0 = time.perf_counter()
        out = sess.run(None, {"A": A, "B": B})
        best = min(best, time.perf_counter() - t0)
    gbs = (K * N * 2) / best / 1e9
    return gbs, "ok"

for did in range(4):
    gbs, msg = bench(did)
    if gbs is None:
        print(f"device_id={did}: {msg}")
    else:
        print(f"device_id={did}: MatMul 512MB-B @ {gbs:6.1f} GB/s (B-read effective)")
