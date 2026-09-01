# iGPU zero-copy read bandwidth: DML IOBinding (B bound once, reused)
import numpy as np, time
import onnxruntime as ort
from onnx import helper, TensorProto

K, N = 4096, 65536  # B fp16 [K,N] = 512MB
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
open(fname, "wb").write(model.SerializeToString())

def bench(device_id):
    try:
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        sess = ort.InferenceSession(fname, so, providers=[("DmlExecutionProvider", {"device_id": device_id})])
    except Exception as e:
        return None, f"init: {str(e)[:120]}"
    io = sess.io_binding()
    try:
        b_ort = ort.OrtValue.ortvalue_from_numpy(B, "DML")
        a_ort = ort.OrtValue.ortvalue_from_numpy(A, "DML")
        y_ort = ort.OrtValue.ortvalue_from_numpy(Y, "DML")
    except Exception as e:
        return None, f"ortvalue DML: {str(e)[:150]}"
    io.bind_ortvalue_input("A", a_ort)
    io.bind_ortvalue_input("B", b_ort)
    io.bind_ortvalue_output("Y", y_ort)
    sess.run_with_iobinding(io, None)  # warmup (first read may page in)
    best = float("inf")
    for _ in range(6):
        t0 = time.perf_counter()
        sess.run_with_iobinding(io, None)
        best = min(best, time.perf_counter() - t0)
    gbs = (K * N * 2) / best / 1e9
    return gbs, "ok"

for did in [0, 1]:
    gbs, msg = bench(did)
    print(f"device_id={did}: IOBinding zero-copy B-read @ {gbs if gbs else '--':>6} GB/s | {msg}")
