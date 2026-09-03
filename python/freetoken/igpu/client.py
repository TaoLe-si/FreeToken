"""Engine-side HIP worker client.

Manages the lifecycle of the HIP worker subprocess and the TCP-loopback IPC
that carries decode requests/responses. One client per engine.
"""

from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
import threading
import time

import numpy as np

from .protocol import (
    DEFAULT_PORT,
    EngineSide,
    H_DIM,
    IpcError,
    REQUEST_PAYLOAD,
    RESPONSE_PAYLOAD,
    TOPK,
    pack_request,
    unpack_response,
)


def _worker_executable() -> list:
    """Return the argv to spawn the worker: same Python interpreter as the engine,
    with PYTHONPATH pointing at the FreeToken python/ tree so that
    'freetoken.igpu.worker' is importable."""
    py = sys.executable
    pyhome = os.path.dirname(os.path.dirname(py))  # .../venv
    env_pypath = os.environ.get("PYTHONPATH", "")
    pkg_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # .../python
    pypath = os.pathsep.join(x for x in (env_pypath, pkg_root) if x)
    env = {
        "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
        "TEMP": os.environ.get("TEMP", ""),
        "TMP": os.environ.get("TMP", ""),
        "LOCALAPPDATA": os.environ.get("LOCALAPPDATA", ""),
        "PYTHONPATH": pypath,
        "PATH": os.environ.get("PATH", ""),
    }
    return [py, "-u", "-m", "freetoken.igpu.worker", "--port", str(DEFAULT_PORT)], env


class IgpuClient:
    def __init__(
        self,
        ftw_path: str,
        num_layers: int = 40,
        port: int = DEFAULT_PORT,
        ready_timeout_s: float = 180.0,
    ) -> None:
        self.ftw_path = ftw_path
        self.num_layers = num_layers
        self.port = port
        self.ready_timeout_s = ready_timeout_s
        self._ipc: EngineSide | None = None
        self._proc: subprocess.Popen | None = None
        self._stdout_thread: threading.Thread | None = None
        self._ready_event = threading.Event()

    # ------------------------------------------------------------------ lifecycle

    def start(self) -> None:
        if self._proc is not None:
            raise IpcError("IgpuClient already started")
        argv, env = _worker_executable()
        argv += ["--ftw", self.ftw_path, "--num-layers", str(self.num_layers), "--port", str(self.port)]
        creationflags = 0
        if os.name == "nt":
            creationflags = subprocess.CREATE_NO_WINDOW
        self._proc = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            creationflags=creationflags,
        )
        # Drain stdout in a thread so the worker can't block on a full pipe.
        self._stdout_thread = threading.Thread(
            target=self._drain_stdout, args=(self._proc.stdout,), daemon=True,
        )
        self._stdout_thread.start()

        # Start the IPC listener BEFORE we wait for the worker -- the worker
        # connects to us, so the listener has to be ready first.
        self._ipc = EngineSide(port=self.port)
        self._ipc.start(timeout_s=self.ready_timeout_s)

        # Wait for the worker's READY signal (it logs it to stdout).
        if not self._wait_for_ready(self.ready_timeout_s):
            self.stop()
            raise IpcError("worker did not become ready within %.1fs" % self.ready_timeout_s)

    def stop(self) -> None:
        # Close the IPC first so the worker exits its recv loop.
        if self._ipc is not None:
            try:
                self._ipc.close()
            except Exception:
                pass
            self._ipc = None
        if self._proc is not None:
            try:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
                    self._proc.wait(timeout=5.0)
            except Exception:
                pass
            self._proc = None

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    # ------------------------------------------------------------------ decode

    def decode(
        self,
        hidden: np.ndarray,
        ids: np.ndarray,
        weights: np.ndarray,
        timeout_s: float = 60.0,
    ) -> np.ndarray:
        if self._ipc is None:
            raise IpcError("client not started")
        # Pack the request into a fixed-size payload.
        # The struct format is little-endian; we just call the helper.
        payload = pack_request(
            hidden.astype(np.float32, copy=False).tolist(),
            ids.astype(np.int32, copy=False).tolist(),
            weights.astype(np.float32, copy=False).tolist(),
            token_id=0,
            request_id=0,
            seq=0,
        )
        assert len(payload) == REQUEST_PAYLOAD, (len(payload), REQUEST_PAYLOAD)
        t0 = time.perf_counter()
        self._ipc.send_request(payload, timeout_s=timeout_s)
        resp = self._ipc.recv_response()
        dt = time.perf_counter() - t0
        if len(resp) != RESPONSE_PAYLOAD:
            raise IpcError("short response: %d bytes" % len(resp))
        out_list, rc, latency_us = unpack_response(resp)
        if rc != 0:
            raise IpcError("worker rc=%d (latency_us=%d, dt_ms=%.2f)" % (rc, latency_us, dt * 1000))
        return np.asarray(out_list, dtype=np.float32)

    # ------------------------------------------------------------------ helpers

    def _drain_stdout(self, stream) -> None:
        try:
            for line in iter(stream.readline, b""):
                if not line:
                    break
                try:
                    text = line.decode("utf-8", errors="replace").rstrip()
                except Exception:
                    continue
                if text.startswith("IGPU_W "):
                    try:
                        obj = json.loads(text[len("IGPU_W "):])
                        if obj.get("msg") == "ready for requests":
                            self._ready_event.set()
                    except Exception:
                        pass
                    sys.stderr.write(text + "\n")
                else:
                    sys.stderr.write(text + "\n")
        except Exception:
            pass

    def _wait_for_ready(self, timeout_s: float) -> bool:
        return self._ready_event.wait(timeout=timeout_s)
