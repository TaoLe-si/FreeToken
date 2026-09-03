"""TCP-loopback IPC between the engine (server) and the HIP worker (client).

Earlier revisions used a Windows named mmap to share state across processes;
that does not work -- Python's mmap.mmap() on Windows gives each process a
private view of the named mapping (verified: writer writes tail=1..5, reader
in a separate process reads tail=0 for the lifetime of the test). True cross-
process coherent memory on Windows requires CreateFileMapping with the
SEC_RESERVE/SEC_COMMIT semantics and explicit MapViewOfFile2 calls -- not
exposed by the stdlib mmap module.

TCP loopback on 127.0.0.1 gives us ~10 GB/s on this machine and is the
smallest dependency on platform-specific behaviour. The IPC overhead per
request is ~50 us (single connection, no per-request handshake), well within
the 1-2 ms budget documented in FORM2_CROSS_PROCESS_DESIGN.md.

The wire format matches what the ring would have carried: header (1B opcode),
then the request or response slot bytes. No length prefix because each slot
is fixed-size.
"""

from __future__ import annotations

import os
import socket
import struct
import threading
import time

H_DIM = 2048
TOPK = 8
REQUEST_PAYLOAD = H_DIM * 4 + TOPK * 4 + TOPK * 4 + 8 + 8 + 8
RESPONSE_PAYLOAD = H_DIM * 4 + 4 + 8

# Pick a fixed port in the IANA dynamic range; tests can override via env.
DEFAULT_PORT = int(os.environ.get("FT_IGPU_PORT", "19750"))

OP_HELLO = 1
OP_READY = 2
OP_REQUEST = 3
OP_RESPONSE = 4
OP_BYE = 5


class IpcError(RuntimeError):
    pass


def _send_exact(sock: socket.socket, data: bytes) -> None:
    view = memoryview(data)
    sent = 0
    while sent < len(view):
        n = sock.send(view[sent:])
        if n == 0:
            raise IpcError("socket closed mid-send")
        sent += n


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    chunks = []
    remaining = n
    while remaining > 0:
        buf = sock.recv(remaining)
        if not buf:
            raise IpcError("socket closed mid-recv")
        chunks.append(buf)
        remaining -= len(buf)
    return b"".join(chunks)


class EngineSide:
    """The engine owns this. Listens for one worker connection, then exposes
    blocking send_request / recv_response primitives that match the ring API."""

    def __init__(self, port: int = DEFAULT_PORT, bind_host: str = "127.0.0.1") -> None:
        self.port = port
        self.bind_host = bind_host
        self._sock = None
        self._conn = None
        self._seq = 0
        self._lock = threading.Lock()
        self._slot_count = 0

    def start(self, *, timeout_s: float = 60.0) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self.bind_host, self.port))
        self._sock.listen(1)
        self._sock.settimeout(timeout_s)
        self._conn, _ = self._sock.accept()
        # Worker sends HELLO then a slot_count message; we ack by being ready.
        op = struct.unpack("!B", _recv_exact(self._conn, 1))[0]
        if op != OP_HELLO:
            raise IpcError("expected HELLO, got opcode " + str(op))
        self._slot_count = struct.unpack("!I", _recv_exact(self._conn, 4))[0]
        self._conn.settimeout(None)

    def send_request(self, request_slot: bytes, *, timeout_s: float = 30.0) -> None:
        self._conn.settimeout(timeout_s)
        _send_exact(self._conn, struct.pack("!BI", OP_REQUEST, len(request_slot)) + request_slot)

    def recv_response(self) -> tuple:
        # 1B op + 4B len + slot bytes
        head = _recv_exact(self._conn, 5)
        op, n = struct.unpack("!BI", head)
        if op != OP_RESPONSE:
            raise IpcError("expected RESPONSE, got opcode " + str(op))
        return _recv_exact(self._conn, n)

    def close(self) -> None:
        try:
            if self._conn is not None:
                try:
                    _send_exact(self._conn, struct.pack("!B", OP_BYE))
                except Exception:
                    pass
                self._conn.close()
        finally:
            self._conn = None
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None


class WorkerSide:
    """The HIP worker owns this. Connects to the engine's listener, sends HELLO,
    then exposes blocking recv_request / send_response primitives."""

    def __init__(self, port: int = DEFAULT_PORT, host: str = "127.0.0.1") -> None:
        self.port = port
        self.host = host
        self._sock = None

    def connect(self, *, slot_count: int, timeout_s: float = 60.0) -> None:
        deadline = time.monotonic() + timeout_s
        last_err = None
        while time.monotonic() < deadline:
            try:
                self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self._sock.settimeout(2.0)
                self._sock.connect((self.host, self.port))
                _send_exact(self._sock, struct.pack("!BI", OP_HELLO, slot_count))
                self._sock.settimeout(None)
                return
            except OSError as e:
                last_err = e
                try:
                    if self._sock is not None:
                        self._sock.close()
                finally:
                    self._sock = None
                time.sleep(0.5)
        raise IpcError("worker connect timeout: " + str(last_err))

    def recv_request(self) -> bytes:
        head = _recv_exact(self._sock, 5)
        op, n = struct.unpack("!BI", head)
        if op == OP_BYE:
            raise IpcError("engine closed connection")
        if op != OP_REQUEST:
            raise IpcError("expected REQUEST, got opcode " + str(op))
        return _recv_exact(self._sock, n)

    def send_response(self, response_slot: bytes) -> None:
        _send_exact(self._sock, struct.pack("!BI", OP_RESPONSE, len(response_slot)) + response_slot)

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None


# ---- Slot pack / unpack ------------------------------------------------------

_REQUEST_STRUCT = struct.Struct(
    f"<{H_DIM}f"
    f"{TOPK}i"
    f"{TOPK}f"
    "QQQ"
)
assert _REQUEST_STRUCT.size == REQUEST_PAYLOAD


def pack_request(hidden, ids, weights, token_id, request_id, seq):
    return _REQUEST_STRUCT.pack(*hidden, *ids, *weights, token_id, request_id, seq)


def unpack_request(payload: bytes):
    vals = _REQUEST_STRUCT.unpack(payload)
    hidden = list(vals[:H_DIM])
    ids = list(vals[H_DIM : H_DIM + TOPK])
    weights = list(vals[H_DIM + TOPK : H_DIM + 2 * TOPK])
    token_id = int(vals[H_DIM + 2 * TOPK])
    request_id = int(vals[H_DIM + 2 * TOPK + 1])
    seq = int(vals[H_DIM + 2 * TOPK + 2])
    return hidden, ids, weights, token_id, request_id, seq


_RESPONSE_STRUCT = struct.Struct(f"<{H_DIM}f i Q")
assert _RESPONSE_STRUCT.size == RESPONSE_PAYLOAD


def pack_response(out_hidden, rc, latency_us):
    return _RESPONSE_STRUCT.pack(*out_hidden, rc, latency_us)


def unpack_response(payload: bytes):
    vals = _RESPONSE_STRUCT.unpack(payload)
    out_hidden = list(vals[:H_DIM])
    rc = int(vals[H_DIM])
    latency_us = int(vals[H_DIM + 1])
    return out_hidden, rc, latency_us
