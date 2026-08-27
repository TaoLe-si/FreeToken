from __future__ import annotations

from dataclasses import dataclass, field

from freetoken.engine import EngineConfig


def _zmq_transport_addr(idx: int, suffix: str) -> str:
    """Windows lacks the ZMQ ipc transport; use loopback TCP, deterministic port.

    厂商常驻软件（华硕管家/应用商店等）会恰好监听在公式窗口内——连接它们将得到
    无限期的无应答（10060），detokenizer 因此在装载期静默死亡。此处对候选端口做
    bind 探测，被占用即向上避让。

    [ft-zmq-env-sync] 探测结果经环境变量跨进程同步：主进程（前端）先分配并导出，
    spawn 子进程继承环境后必须原样复用。否则子进程自己的 bind 探测会把前端真实
    绑定视为"被占用"而避让，把回复推到无人监听的端口上。"""
    import os
    _k = (idx, suffix)
    _c = globals().get('_ZMQ_ADDR_CACHE')
    if _c is not None and _k in _c:
        return _c[_k]
    env_key = "FREETOKEN_ZMQ_ADDR_%d" % idx
    inherited = os.environ.get(env_key)
    if inherited:
        if _c is not None:
            _c[_k] = inherited
        return inherited
    addr = None
    if os.name == "nt":
        import socket as _socket
        seed = 0
        for ch in suffix:
            if ch.isdigit():
                seed = seed * 10 + int(ch)
        start = 34000 + idx * 10 + (seed % 400)
        for offset in range(0, 500):
            candidate = start + offset
            try:
                with _socket.socket() as _sk:
                    _sk.bind(("127.0.0.1", candidate))
            except OSError:
                continue
            addr = "tcp://127.0.0.1:%d" % candidate
            break
        if addr is None:
            addr = "tcp://127.0.0.1:%d" % start
    else:
        addr = "ipc:///tmp/freetoken_%d%s" % (idx, suffix)
    if _c is not None:
        _c[_k] = addr
    os.environ[env_key] = addr
    return addr





def _get_pid_suffix() -> str:
    import os

    return f".pid={os.getpid()}"


@dataclass(frozen=True)
class SchedulerConfig(EngineConfig):
    max_extend_tokens: int = 8192
    cache_type: str = "radix"
    offline_mode: bool = False
    decode_log_interval: int = 40
    special_token_ckpt: bool = False

    # networking config
    _unique_suffix: str = field(default_factory=_get_pid_suffix)

    @property
    def zmq_backend_addr(self) -> str:
        return _zmq_transport_addr(0, self._unique_suffix)

    @property
    def zmq_detokenizer_addr(self) -> str:
        return _zmq_transport_addr(1, self._unique_suffix)

    @property
    def zmq_scheduler_broadcast_addr(self) -> str:
        return _zmq_transport_addr(2, self._unique_suffix)

    @property
    def max_forward_len(self) -> int:
        return self.max_extend_tokens

    @property
    def backend_create_detokenizer_link(self) -> bool:
        return True
