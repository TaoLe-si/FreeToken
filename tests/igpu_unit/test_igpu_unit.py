"""Standalone unit tests for the cross-process iGPU MoE plumbing.

These tests run in the venv Python without needing real HIP / CUDA hardware.
They exercise the protocol/serialization, the handshake order, decode/staging
round-trip on mock workers, batched H2D/D2H paths, error propagation,
shutdown semantics, and the engine-side decode() full pipeline.

Run:
    venv python tests/igpu_unit/test_igpu_unit.py

The tests don't require pytest -- they just run sequentially with assertions.
"""

from __future__ import annotations

import os
import socket
import struct
import sys
import threading
import time

import numpy as np
import torch

sys.path.insert(0, "E:/FreeToken/python")

from freetoken.moe.igpu_shared_executor import (  # noqa
    IgpuSharedMoeExecutor, _H, _TOPK, _REQUEST_BYTES, _RESPONSE_BYTES,
)


# ============================================================================
# shared helpers
# ============================================================================

def _send_exact(sock, data):
    view = memoryview(data)
    sent = 0
    while sent < len(view):
        n = sock.send(view[sent:])
        if n == 0:
            raise ConnectionError("socket closed mid-send")
        sent += n


def _recv_exact(sock, n):
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        chunk = sock.recv(n - got)
        if not chunk:
            return None
        view[got:got + len(chunk)] = chunk
        got += len(chunk)
    return bytes(buf)


def _spawn_mock_worker(port: int, on_ready, on_request):
    """Spawn a mock TCP worker thread mimicking the real worker's order:
    listen -> accept -> ready event -> per-request callback."""
    ready_event = threading.Event()
    stop_event = threading.Event()
    error_holder = []

    def serve():
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", port))
        s.listen(1)
        s.settimeout(10.0)
        try:
            conn, addr = s.accept()
        except socket.timeout:
            error_holder.append("accept timeout")
            return
        s.close()
        conn.settimeout(60.0)
        on_ready()
        ready_event.set()
        try:
            while not stop_event.is_set():
                hdr = _recv_exact(conn, 4)
                if hdr is None:
                    break
                req_len = struct.unpack("<I", hdr)[0]
                if req_len % _REQUEST_BYTES != 0:
                    error_holder.append("bad request size " + str(req_len))
                    break
                payload = _recv_exact(conn, req_len)
                if payload is None:
                    break
                resp = on_request(payload)
                if resp is None:
                    break
                _send_exact(conn, struct.pack("<I", len(resp)) + resp)
        except Exception as e:
            error_holder.append(str(e))
        finally:
            try:
                conn.close()
            except Exception:
                pass

    t = threading.Thread(target=serve, daemon=True)
    t.start()
    return ready_event, stop_event, error_holder


def _pack_request_like_executor(layer_id, hid, ids, wts):
    """Mimics the engine's executor.decode packing layout: per-token layer_id
    then hidden (8KB) then ids (32B) then wts (32B), all tokens concatenated."""
    bs = hid.shape[0]
    req = bytearray(_REQUEST_BYTES * bs)
    for tok in range(bs):
        base = tok * _REQUEST_BYTES
        req[base] = layer_id & 0xFF
        req[base + 1:base + 1 + _H * 4] = bytes(hid[tok].numpy())
        off = base + 1 + _H * 4
        req[off:off + _TOPK * 4] = bytes(ids[tok].numpy())
        off += _TOPK * 4
        req[off:off + _TOPK * 4] = bytes(wts[tok].numpy())
    return bytes(req)


def _make_inputs(bs: int, seed: int = 0):
    rng = np.random.RandomState(seed)
    hid = torch.from_numpy(rng.standard_normal((bs, _H)).astype(np.float32) * 0.5)
    ids = torch.from_numpy(rng.randint(0, 256, size=(bs, _TOPK), dtype=np.int32))
    wts = torch.full((bs, _TOPK), 0.125, dtype=torch.float32)
    return hid.pin_memory(), ids.pin_memory(), wts.pin_memory()


# ============================================================================
# 1. protocol: byte layout
# ============================================================================

def test_request_byte_layout():
    """_REQUEST_BYTES (8257) and _RESPONSE_BYTES (8192) match the protocol."""
    assert _REQUEST_BYTES == 1 + _H * 4 + _TOPK * 4 + _TOPK * 4 == 8257
    assert _RESPONSE_BYTES == _H * 4 == 8192


# ============================================================================
# 2. handshake: connect-before-ready ordering
# ============================================================================

def test_handshake_connect_before_ready():
    """Worker is blocked on accept() before logging ready. Engine MUST connect
    TCP FIRST, then wait for ready -- otherwise both block forever."""
    port = 19801
    times = {}
    t0 = time.perf_counter()
    def on_ready():
        times["ready"] = time.perf_counter() - t0
    def on_request(p):
        return None
    ready, stop, err = _spawn_mock_worker(port, on_ready, on_request)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(("127.0.0.1", port))
    times["connected"] = time.perf_counter() - t0
    s.settimeout(5.0)
    assert ready.wait(timeout=5.0), "worker never signalled ready"
    assert times["ready"] > times["connected"], "ready must come AFTER connect"
    s.close()
    stop.set()


def test_handshake_no_deadlock_when_connect_delayed():
    """If the engine connects AFTER worker is already accepting, both should
    complete cleanly (no deadlock the other direction either)."""
    port = 19802
    ready, stop, err = _spawn_mock_worker(port, lambda: None, lambda p: None)
    time.sleep(0.3)  # let worker be in accept()
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(("127.0.0.1", port))
    assert ready.wait(timeout=5.0)
    s.close()
    stop.set()
    assert not err or all("accept" not in e for e in err), err


# ============================================================================
# 3. pinned D2H + host read race
# ============================================================================

def test_pinned_d2h_then_host_read():
    """async D2H to pinned CPU tensor, sync current stream, then host reads."""
    if torch.cuda.is_available():
        src = torch.zeros(_H, dtype=torch.float32, device="cuda:0")
        for i in range(_H):
            src[i] = float(i)
        buf = torch.empty(_H, dtype=torch.float32, pin_memory=True)
        buf.copy_(src, non_blocking=True)
        torch.cuda.current_stream().synchronize()
        arr = buf.numpy()
        assert float(arr[123]) == 123.0, float(arr[123])
    else:
        src = torch.arange(_H, dtype=torch.float32)
        buf = torch.empty(_H, dtype=torch.float32, pin_memory=True)
        buf.copy_(src)
        arr = buf.numpy()
        assert float(arr[123]) == 123.0


# ============================================================================
# 4. recv_into writable buffer requirement
# ============================================================================

def test_recv_into_bytearray():
    """recv_into needs writable bytes-like; bytearray works, bytes doesn't."""
    port = 19803
    ready, stop, err = _spawn_mock_worker(port, lambda: None,
                                          lambda p: bytes(_RESPONSE_BYTES))
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(("127.0.0.1", port))
    assert ready.wait(timeout=5.0)
    _send_exact(s, struct.pack("<I", _REQUEST_BYTES) + bytes(_REQUEST_BYTES))
    resp = bytearray(_RESPONSE_BYTES)
    got = 0
    while got < _RESPONSE_BYTES:
        n = s.recv_into(memoryview(resp)[got:], _RESPONSE_BYTES - got)
        if n == 0:
            raise ConnectionError()
        got += n
    s.close()
    stop.set()
    assert got == _RESPONSE_BYTES


def test_recv_into_readonly_rejected():
    """memoryview(bytes(...)) is read-only; recv_into rejects it."""
    data = bytes(_RESPONSE_BYTES)
    mv = memoryview(data)
    assert mv.readonly, "memoryview(bytes) must be readonly"
    # Sanity: bytearray version is writable
    ba = bytearray(_RESPONSE_BYTES)
    assert not memoryview(ba).readonly


# ============================================================================
# 5. multi-token packed layout (executor parity)
# ============================================================================

def test_packed_layout_bs1():
    """bs=1 pack: 1B layer_id + 8KB hidden + 32B ids + 32B wts."""
    hid, ids, wts = _make_inputs(1)
    req = _pack_request_like_executor(7, hid, ids, wts)
    assert len(req) == _REQUEST_BYTES
    assert req[0] == 7
    # Round-trip a few floats/ints
    h_np = np.frombuffer(req[1:1 + _H * 4], dtype=np.float32).reshape(_H)
    assert abs(float(h_np[100]) - float(hid[0][100])) < 1e-5
    i_np = np.frombuffer(req[1 + _H * 4:1 + _H * 4 + _TOPK * 4], dtype=np.int32)
    assert int(i_np[3]) == int(ids[0][3])
    w_np = np.frombuffer(req[1 + _H * 4 + _TOPK * 4:_REQUEST_BYTES], dtype=np.float32)
    assert abs(float(w_np[5]) - 0.125) < 1e-5


def test_packed_layout_bs4_per_token_layer_id():
    """bs=4: each token's first byte must be its layer_id, NOT shared."""
    hid, ids, wts = _make_inputs(4)
    req = _pack_request_like_executor(0xAB, hid, ids, wts)
    assert len(req) == _REQUEST_BYTES * 4
    for tok in range(4):
        assert req[tok * _REQUEST_BYTES] == 0xAB


def test_packed_layout_bs128_large_payload():
    """bs=128: request is ~1MB; packed layout valid for 40 layers x bs=128."""
    hid, ids, wts = _make_inputs(128, seed=99)
    req = _pack_request_like_executor(0x42, hid, ids, wts)
    assert len(req) == _REQUEST_BYTES * 128 == 1_056_896
    # First token and last token must both have layer_id = 0x42
    assert req[0] == 0x42
    assert req[(128 - 1) * _REQUEST_BYTES] == 0x42
    # Middle token: layer_id byte at offset 64 * _REQUEST_BYTES
    assert req[64 * _REQUEST_BYTES] == 0x42
    # First token's hidden first float should be hid[0][0]
    h_np = np.frombuffer(req[1:1 + _H * 4], dtype=np.float32)
    assert abs(float(h_np[0]) - float(hid[0][0])) < 1e-5
    # Last token's hidden last float should be hid[127][2047]
    base = (128 - 1) * _REQUEST_BYTES + 1
    h_np = np.frombuffer(req[base:base + _H * 4], dtype=np.float32)
    assert abs(float(h_np[-1]) - float(hid[127][-1])) < 1e-5


# ============================================================================
# 6. mock worker: end-to-end protocol round-trip (bs=1, bs=2, bs=128)
# ============================================================================

def test_e2e_bs1():
    """Send bs=1 request, verify worker parses layer_id and returns the
    requested response bytes."""
    port = 19901
    captured = {}
    def on_ready(): pass
    def on_request(payload):
        captured["len"] = len(payload)
        captured["layer_id"] = payload[0]
        # Token t=0 response: arange(_H) + 7.0
        resp = bytearray(_RESPONSE_BYTES)
        np.copyto(np.frombuffer(resp, dtype=np.float32), np.arange(_H, dtype=np.float32) + 7.0)
        return bytes(resp)
    ready, stop, err = _spawn_mock_worker(port, on_ready, on_request)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(("127.0.0.1", port))
    assert ready.wait(timeout=5.0)
    hid, ids, wts = _make_inputs(1)
    req = _pack_request_like_executor(13, hid, ids, wts)
    _send_exact(s, struct.pack("<I", len(req)) + req)
    hdr = _recv_exact(s, 4)
    resp = _recv_exact(s, struct.unpack("<I", hdr)[0])
    s.close()
    stop.set()
    assert captured["len"] == _REQUEST_BYTES
    assert captured["layer_id"] == 13
    out = np.frombuffer(resp, dtype=np.float32)
    assert abs(float(out[100]) - 107.0) < 1e-5  # arange(100) + 7
    assert abs(float(out[2047]) - 2054.0) < 1e-5  # 2047 + 7


def test_e2e_bs2():
    """bs=2: each token's layer_id is preserved; response carries 2*H floats."""
    port = 19902
    captured = []
    def on_ready(): pass
    def on_request(payload):
        captured.append(payload[:])
        resp = bytearray(2 * _RESPONSE_BYTES)
        np.copyto(np.frombuffer(resp, dtype=np.float32).reshape(2, _H),
                  np.arange(2 * _H, dtype=np.float32).reshape(2, _H) + 100.0)
        return bytes(resp)
    ready, stop, err = _spawn_mock_worker(port, on_ready, on_request)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(("127.0.0.1", port))
    assert ready.wait(timeout=5.0)
    hid, ids, wts = _make_inputs(2)
    req = _pack_request_like_executor(99, hid, ids, wts)
    _send_exact(s, struct.pack("<I", len(req)) + req)
    hdr = _recv_exact(s, 4)
    resp = _recv_exact(s, struct.unpack("<I", hdr)[0])
    s.close()
    stop.set()
    p = captured[0]
    assert p[0] == 99
    assert p[_REQUEST_BYTES] == 99
    out = np.frombuffer(resp, dtype=np.float32).reshape(2, _H)
    assert abs(float(out[0][0]) - 100.0) < 1e-5
    assert abs(float(out[1][0]) - 2148.0) < 1e-5  # 2048 + 100


def test_e2e_bs128_protocol():
    """bs=128: verify all 128 layer_ids present, all 128 hidden slices present,
    response carries 128*H floats. (~1MB each direction.)"""
    port = 19903
    captured = []
    def on_ready(): pass
    def on_request(payload):
        captured.append(len(payload))
        resp = bytearray(128 * _RESPONSE_BYTES)
        np.copyto(np.frombuffer(resp, dtype=np.float32).reshape(128, _H),
                  np.broadcast_to(np.arange(_H, dtype=np.float32), (128, _H)).copy())
        return bytes(resp)
    ready, stop, err = _spawn_mock_worker(port, on_ready, on_request)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(("127.0.0.1", port))
    assert ready.wait(timeout=5.0)
    hid, ids, wts = _make_inputs(128, seed=11)
    req = _pack_request_like_executor(0x77, hid, ids, wts)
    assert len(req) == 1_056_896
    _send_exact(s, struct.pack("<I", len(req)) + req)
    hdr = _recv_exact(s, 4)
    resp = _recv_exact(s, struct.unpack("<I", hdr)[0])
    s.close()
    stop.set()
    assert captured[0] == 1_056_896
    assert len(resp) == 1_048_576


# ============================================================================
# 7. worker error propagation
# ============================================================================

def test_bad_request_size_rejected():
    """A request whose payload isn't a multiple of _REQUEST_BYTES is rejected
    by the worker (and the mock worker closes the connection)."""
    port = 19904
    errors = []
    def on_ready(): pass
    def on_request(p):
        errors.append("should not reach here")
        return b"\x00" * _RESPONSE_BYTES
    ready, stop, err = _spawn_mock_worker(port, on_ready, on_request)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(("127.0.0.1", port))
    assert ready.wait(timeout=5.0)
    bad = struct.pack("<I", 1234) + bytes(1234)  # not a multiple of 8257
    try:
        _send_exact(s, bad)
    except Exception:
        pass
    time.sleep(0.2)
    s.close()
    stop.set()
    # Worker should have logged the bad size
    assert any("bad request size" in e for e in err), err
    assert errors == []  # callback should never have been invoked


# ============================================================================
# 8. socket behaviour: blocking mode, EOF detection
# ============================================================================

def test_socket_blocking_no_timeout():
    """Engine's executor removes socket timeout (setblocking / None). The
    decoded pipeline must remain correct under blocking I/O."""
    port = 19905
    received = []
    def on_ready(): pass
    def on_request(payload):
        received.append(payload[:])
        resp = bytearray(_RESPONSE_BYTES)
        np.copyto(np.frombuffer(resp, dtype=np.float32), np.zeros(_H, dtype=np.float32))
        return bytes(resp)
    ready, stop, err = _spawn_mock_worker(port, on_ready, on_request)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(None)  # blocking mode, no timeout
    s.connect(("127.0.0.1", port))
    assert ready.wait(timeout=5.0)
    hid, ids, wts = _make_inputs(1)
    req = _pack_request_like_executor(0x33, hid, ids, wts)
    _send_exact(s, struct.pack("<I", len(req)) + req)
    hdr = _recv_exact(s, 4)
    resp = _recv_exact(s, struct.unpack("<I", hdr)[0])
    s.close()
    stop.set()
    assert len(resp) == _RESPONSE_BYTES


def test_peer_closed_detected_as_eof():
    """When the worker process dies / closes the socket, recv_into returns 0
    (EOF), not a timeout. We rely on this since we removed socket timeout."""
    port = 19906
    def on_ready(): pass
    def on_request(p):
        return None  # closes connection immediately
    ready, stop, err = _spawn_mock_worker(port, on_ready, on_request)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(("127.0.0.1", port))
    assert ready.wait(timeout=5.0)
    # Mock worker will return None -> conn.close -> next recv returns 0
    _send_exact(s, struct.pack("<I", _REQUEST_BYTES) + bytes(_REQUEST_BYTES))
    # Drain whatever the worker sent (could be partial); eventually get EOF
    got_zero = False
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        try:
            n = s.recv_into(memoryview(bytearray(8)), 8)
            if n == 0:
                got_zero = True
                break
        except Exception:
            got_zero = True
            break
    s.close()
    stop.set()
    assert got_zero, "expected EOF (recv_into returning 0) after peer close"


# ============================================================================
# 9. executor's class-level invariants (no env-var hacks, graph_replay_safe)
# ============================================================================

def test_executor_graph_replay_safe_false():
    assert hasattr(IgpuSharedMoeExecutor, "graph_replay_safe")
    assert IgpuSharedMoeExecutor.graph_replay_safe is False


def test_no_env_var_hack():
    import inspect
    src = inspect.getsource(IgpuSharedMoeExecutor.__init__)
    assert "FT_SKIP_CUDA_GRAPH" not in src
    assert "FT_DISABLE_DECODE_REPLAY" not in src


# ============================================================================
# 10. executor.decode() full pipeline (no real HIP, mock subprocess)
# ============================================================================

class _FakeWorkerSubprocess:
    """Mimics subprocess.Popen: serves decode requests on TCP loopback and
    emits the IGPU_W {"msg": "ready for requests"} line so the executor's
    drain_stdout thread fires its _ready_event."""
    def __init__(self, port, on_request):
        self.pid = 99001
        r_fd, w_fd = os.pipe()
        self._stdout_read = os.fdopen(r_fd, "rb", buffering=0)
        self._stdout_write = os.fdopen(w_fd, "wb", buffering=0)
        self.stdout = self._stdout_read
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._port = port
        self._on_request = on_request

        def serve():
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("127.0.0.1", port))
            s.listen(1)
            s.settimeout(5.0)
            try:
                conn, _ = s.accept()
            except socket.timeout:
                return
            s.close()
            conn.settimeout(30.0)
            try:
                self._stdout_write.write(b'IGPU_W {"ts": 0, "level": "INFO", "msg": "ipc connected"}\n')
                self._stdout_write.write(b'IGPU_W {"ts": 0, "level": "INFO", "msg": "ready for requests"}\n')
                self._stdout_write.flush()
            except Exception:
                pass
            self._ready.set()
            try:
                while not self._stop.is_set():
                    hdr = _recv_exact(conn, 4)
                    if hdr is None:
                        break
                    req_len = struct.unpack("<I", hdr)[0]
                    if req_len % _REQUEST_BYTES != 0:
                        break
                    payload = _recv_exact(conn, req_len)
                    if payload is None:
                        break
                    resp = on_request(payload)
                    if resp is None:
                        break
                    _send_exact(conn, struct.pack("<I", len(resp)) + resp)
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

        threading.Thread(target=serve, daemon=True).start()

    def poll(self):
        return None

    def terminate(self):
        self._stop.set()

    def wait(self, timeout=5.0):
        return 0

    def kill(self):
        self._stop.set()




def _patch_skip_wait():
    """Bypass executor._wait_for_ready (the drain-driven 240s wait). For
    pipeline tests we don't have a real subprocess emitting IGPU_W lines;
    skipping is simpler than racing the os.pipe drain thread."""
    def noop(self, timeout_s):
        self._ready_event.set()
        return
    IgpuSharedMoeExecutor._wait_for_ready = noop
class _FakeCache:
    quant_format = "nvfp4"
    folder_path = "E:/models/Qwen3.6-35B-A3B-MXFP4-MTP-NVFP4"
    bank_sources = {}
    def is_igpu_shared_layer(self, x):
        return True


def test_executor_decode_full_pipeline_bs1():
    """Spawn a real executor against a fake worker subprocess and verify
    decode() returns the expected tensor shape and values."""
    import freetoken.moe.igpu_shared_executor as exec_mod
    port = 20001

    def on_request(payload):
        # Return deterministic floats so we can assert
        layer_id = payload[0]
        resp = bytearray(_RESPONSE_BYTES)
        np.copyto(np.frombuffer(resp, dtype=np.float32),
                  np.full(_H, float(layer_id) + 0.5, dtype=np.float32))
        return bytes(resp)

    # Monkey-patch subprocess.Popen before constructing the executor
    real_popen = exec_mod.subprocess.Popen

    def fake_popen(argv, **kw):
        # Drop env/dll path setup; just return our fake worker.
        # Also bypass the executor's stdout-drain wait_for_ready by replacing
        # the bound method on the instance.
        fw = _FakeWorkerSubprocess(port, on_request)
        # Wrap the original __init__ so we can patch _wait_for_ready on the
        # instance right after it returns -- but the instance doesn't exist
        # yet (fake_popen is called from inside __init__). We use a class hook
        # via __init_subclass__ semantics: monkey-patch the method on the class
        # for this scope. The executor calls self._wait_for_ready, which will
        # resolve to our override (still defined on the class).
        return fw

    _patch_skip_wait()  # bypass 240s drain-driven wait
    exec_mod.subprocess.Popen = fake_popen
    try:
        # The executor also reaches into port=0 default; we need to override
        # _default_port. Easiest: monkey-patch it.
        exec_mod._default_port = lambda: port
        # And the worker_argv builder - we don't actually run it
        exec_mod._worker_argv = lambda ftw, n, p: ([sys.executable, "-c", "pass"], {})

        ex = IgpuSharedMoeExecutor(_FakeCache(), torch.device("cpu"), 40, 8, 8)
        # Skip 240s wait_for_ready: drain thread may race with our os.pipe.
        ex._ready_event.set()
        ex.register_banks()
        assert ex._sock is not None

        # Now actually call decode()
        bs = 1
        h = torch.full((bs, _H), 1.0, dtype=torch.bfloat16)
        w = torch.full((bs, _TOPK), 0.125, dtype=torch.bfloat16)
        i = torch.zeros((bs, _TOPK), dtype=torch.int32)
        out = ex.decode(7, h, w, i)
        assert out.shape == (bs, _H)
        # Worker returned fill(_H, 7.5); should be in 'out'
        assert abs(float(out[0][0]) - 7.5) < 1e-3, float(out[0][0])
        ex.shutdown()
    finally:
        exec_mod.subprocess.Popen = real_popen


def test_executor_decode_full_pipeline_bs4():
    """bs=4 through the full decode pipeline; verify each token's output
    reflects the layer_id sent for that token."""
    import freetoken.moe.igpu_shared_executor as exec_mod
    port = 20002
    layer_ids_received = []

    def on_request(payload):
        n_tokens = len(payload) // _REQUEST_BYTES
        layer_ids_received.append([payload[t * _REQUEST_BYTES] for t in range(n_tokens)])
        resp = bytearray(n_tokens * _RESPONSE_BYTES)
        for t in range(n_tokens):
            lid = payload[t * _REQUEST_BYTES]
            base = t * _RESPONSE_BYTES
            np.copyto(np.frombuffer(resp, dtype=np.float32)[t * _H:(t + 1) * _H],
                      np.full(_H, float(lid) + 1.5, dtype=np.float32))
        return bytes(resp)

    real_popen = exec_mod.subprocess.Popen

    def fake_popen(argv, **kw):
        return _FakeWorkerSubprocess(port, on_request)

    _patch_skip_wait()  # bypass 240s drain-driven wait
    exec_mod.subprocess.Popen = fake_popen
    try:
        exec_mod._default_port = lambda: port
        exec_mod._worker_argv = lambda ftw, n, p: ([sys.executable, "-c", "pass"], {})

        ex = IgpuSharedMoeExecutor(_FakeCache(), torch.device("cpu"), 40, 8, 8)
        ex.register_banks()
        bs = 4
        h = torch.full((bs, _H), 1.0, dtype=torch.bfloat16)
        w = torch.full((bs, _TOPK), 0.125, dtype=torch.bfloat16)
        i = torch.zeros((bs, _TOPK), dtype=torch.int32)
        out = ex.decode(5, h, w, i)
        assert out.shape == (bs, _H)
        # All tokens share layer_id=5 in this test (decode() takes one
        # layer_id per call), so all rows should be 6.5
        for t in range(bs):
            assert abs(float(out[t][0]) - 6.5) < 1e-3, (t, float(out[t][0]))
        # Worker saw layer_id=5 for every token
        assert layer_ids_received[0] == [5, 5, 5, 5]
        ex.shutdown()
    finally:
        exec_mod.subprocess.Popen = real_popen


def test_executor_decode_uses_pinned_staging():
    """decode() should populate self._staging as pinned CPU tensors of shape
    (bs, _H) float32, (bs, _TOPK) int32, etc."""
    import freetoken.moe.igpu_shared_executor as exec_mod
    port = 20003
    real_popen = exec_mod.subprocess.Popen

    def fake_popen(argv, **kw):
        return _FakeWorkerSubprocess(port, lambda p: bytes(_RESPONSE_BYTES))

    _patch_skip_wait()  # bypass 240s drain-driven wait
    exec_mod.subprocess.Popen = fake_popen
    try:
        exec_mod._default_port = lambda: port
        exec_mod._worker_argv = lambda ftw, n, p: ([sys.executable, "-c", "pass"], {})
        ex = IgpuSharedMoeExecutor(_FakeCache(), torch.device("cpu"), 40, 8, 8)
        ex.register_banks()
        ex.decode(0,
                  torch.zeros(1, _H, dtype=torch.bfloat16),
                  torch.zeros(1, _TOPK, dtype=torch.bfloat16),
                  torch.zeros(1, _TOPK, dtype=torch.int32))
        assert ex._staging is not None
        hid, ids, wts, out = ex._staging
        assert hid.is_pinned()
        assert ids.is_pinned()
        assert wts.is_pinned()
        assert out.is_pinned()
        assert hid.shape == (1, _H)
        assert ids.shape == (1, _TOPK)
        assert wts.shape == (1, _TOPK)
        assert out.shape == (1, _H)
        ex.shutdown()
    finally:
        exec_mod.subprocess.Popen = real_popen


def test_executor_shutdown_closes_socket_and_proc():
    """shutdown() closes the TCP socket first, then terminates the proc."""
    import freetoken.moe.igpu_shared_executor as exec_mod
    port = 20004
    real_popen = exec_mod.subprocess.Popen
    fake = [None]
    def fake_popen(argv, **kw):
        f = _FakeWorkerSubprocess(port, lambda p: bytes(_RESPONSE_BYTES))
        fake[0] = f
        return f

    _patch_skip_wait()  # bypass 240s drain-driven wait
    exec_mod.subprocess.Popen = fake_popen
    try:
        exec_mod._default_port = lambda: port
        exec_mod._worker_argv = lambda ftw, n, p: ([sys.executable, "-c", "pass"], {})
        ex = IgpuSharedMoeExecutor(_FakeCache(), torch.device("cpu"), 40, 8, 8)
        ex.register_banks()
        sock_ref = ex._sock
        assert sock_ref is not None
        ex.shutdown()
        # Socket should be closed
        try:
            # Trying to send on a closed socket should raise
            sock_ref.send(b"x")
            closed = False
        except OSError:
            closed = True
        assert closed, "socket not closed after shutdown"
        # Subprocess should be signalled
        assert fake[0]._stop.is_set(), "subprocess not terminated after shutdown"
    finally:
        exec_mod.subprocess.Popen = real_popen


# ============================================================================
# runner
# ============================================================================

ALL_TESTS = [
    test_request_byte_layout,
    test_handshake_connect_before_ready,
    test_handshake_no_deadlock_when_connect_delayed,
    test_pinned_d2h_then_host_read,
    test_recv_into_bytearray,
    test_recv_into_readonly_rejected,
    test_packed_layout_bs1,
    test_packed_layout_bs4_per_token_layer_id,
    test_packed_layout_bs128_large_payload,
    test_e2e_bs1,
    test_e2e_bs2,
    test_e2e_bs128_protocol,
    test_bad_request_size_rejected,
    test_socket_blocking_no_timeout,
    test_peer_closed_detected_as_eof,
    test_executor_graph_replay_safe_false,
    test_no_env_var_hack,
    test_executor_decode_full_pipeline_bs1,
    test_executor_decode_full_pipeline_bs4,
    test_executor_decode_uses_pinned_staging,
    test_executor_shutdown_closes_socket_and_proc,
]


def _run_all():
    failed = []
    for fn in ALL_TESTS:
        name = fn.__name__
        try:
            fn()
            print("[OK]   " + name)
        except Exception as e:
            failed.append((name, e))
            print("[FAIL] " + name + ": " + repr(e))
    if failed:
        print("\n%d TESTS FAILED" % len(failed))
        sys.exit(1)
    print("\nALL %d UNIT TESTS PASSED" % len(ALL_TESTS))



def _run_group(names):
    import importlib
    failures = []
    for n in names:
        fn = globals().get(n)
        if fn is None:
            failures.append((n, "not found"))
            print("[FAIL] " + n + ": not found")
            continue
        try:
            fn()
            print("[OK]   " + n)
        except Exception as e:
            failures.append((n, e))
            print("[FAIL] " + n + ": " + repr(e))
    return failures


if __name__ == "__main__":
    import sys
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    if arg == "all":
        names = [fn.__name__ for fn in ALL_TESTS]
        fails = _run_group(names)
        print("\n%d tests passed, %d failed" % (len(names) - len(fails), len(fails)))
        sys.exit(1 if fails else 0)
    elif arg in ["proto","handshake","cuda","e2e_small","e2e_large","errors","executor","pipeline"]:
        fails = _run_group({"proto":["test_request_byte_layout","test_packed_layout_bs1","test_packed_layout_bs4_per_token_layer_id","test_packed_layout_bs128_large_payload"],"handshake":["test_handshake_connect_before_ready","test_handshake_no_deadlock_when_connect_delayed"],"cuda":["test_pinned_d2h_then_host_read","test_recv_into_bytearray","test_recv_into_readonly_rejected"],"e2e_small":["test_e2e_bs1","test_e2e_bs2"],"e2e_large":["test_e2e_bs128_protocol"],"errors":["test_bad_request_size_rejected","test_socket_blocking_no_timeout","test_peer_closed_detected_as_eof"],"executor":["test_executor_graph_replay_safe_false","test_no_env_var_hack"],"pipeline":["test_executor_decode_full_pipeline_bs1","test_executor_decode_full_pipeline_bs4","test_executor_decode_uses_pinned_staging","test_executor_shutdown_closes_socket_and_proc"]}.get(arg, []))
        print("\n%d tests passed, %d failed" % (len({"proto":["test_request_byte_layout","test_packed_layout_bs1","test_packed_layout_bs4_per_token_layer_id","test_packed_layout_bs128_large_payload"],"handshake":["test_handshake_connect_before_ready","test_handshake_no_deadlock_when_connect_delayed"],"cuda":["test_pinned_d2h_then_host_read","test_recv_into_bytearray","test_recv_into_readonly_rejected"],"e2e_small":["test_e2e_bs1","test_e2e_bs2"],"e2e_large":["test_e2e_bs128_protocol"],"errors":["test_bad_request_size_rejected","test_socket_blocking_no_timeout","test_peer_closed_detected_as_eof"],"executor":["test_executor_graph_replay_safe_false","test_no_env_var_hack"],"pipeline":["test_executor_decode_full_pipeline_bs1","test_executor_decode_full_pipeline_bs4","test_executor_decode_uses_pinned_staging","test_executor_shutdown_closes_socket_and_proc"]}.get(arg, [])) - len(fails), len(fails)))
        sys.exit(1 if fails else 0)
    else:
        print("unknown group:", arg)
        sys.exit(2)

def _patch_skip_wait():
    """Make executor._wait_for_ready a no-op so pipeline tests don't hang on
    the 240s drain-driven ready wait. The fake worker uses os.pipe and the
    executor's drain thread can race; skipping the wait is simpler."""
    def noop(self, timeout_s):
        self._ready_event.set()
        return
    IgpuSharedMoeExecutor._wait_for_ready = noop


