"""MyToken engine boot shim.

The installed Windows wheel computes deterministic loopback ports for the engine's
ZMQ channels (24000 + idx*10 + pid%2000). Vendor resident software (ASUS manager,
Tencent store helpers...) sometimes listens exactly there; connecting to them yields
a silent no-answer timeout and kills the detokenizer mid-load. We bind-probe the
candidate and walk forward while keeping every component's derivation consistent.
This runs BEFORE freetoken.cli so the compiled module picks up the patched resolver.
"""
import socket as _socket
import sys as _sys

# zmq.asyncio requires the selector loop on Windows; uvicorn defaults to the
# Proactor loop, which explodes on the first tokenizer<->scheduler message.
if _sys.platform == "win32":
    import asyncio as _asyncio
    _asyncio.set_event_loop_policy(_asyncio.WindowsSelectorEventLoopPolicy())


def _free_port(start: int) -> int:
    for offset in range(500):
        candidate = start + offset
        try:
            with _socket.socket() as _sk:
                _sk.bind(("127.0.0.1", candidate))
        except OSError:
            continue
        return candidate
    return start


def _patched_zmq_transport_addr(idx: int, suffix: str) -> str:
    import os
    if os.name == "nt":
        seed = 0
        for ch in suffix:
            if ch.isdigit():
                seed = seed * 10 + int(ch)
        start = 34000 + idx * 10 + (seed % 400)
        return "tcp://127.0.0.1:%d" % _free_port(start)
    import tempfile
    return "ipc://%s/freetoken_%d%s" % (tempfile.gettempdir(), idx, suffix)


def _install_patch() -> None:
    try:
        import freetoken.server.args as _args
        _orig = getattr(_args, "_zmq_transport_addr", None)
        # 仅当编译版仍是朴素公式（无探测）时打补丁，避免重复包装
        probe = getattr(_args, "_zmq_transport_addr_patched", False)
        if not probe and _orig is not None:
            _args._zmq_transport_addr = _patched_zmq_transport_addr
            _args._zmq_transport_addr_patched = True
    except Exception:
        pass


def main() -> int:
    _install_patch()
    sys.argv = ["ft"] + sys.argv[1:]
    from freetoken.cli import main as cli_main
    return cli_main()


if __name__ == "__main__":
    import sys
    raise SystemExit(main())
