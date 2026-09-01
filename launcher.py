"""FreeToken 集成桌面壳

PyInstaller 入口：检测 venv、启动 dev daemon、打开 pywebview 窗口。
双击 exe 即可使用。
"""

import os
import sys
import subprocess
import time
import json
import urllib.request
import urllib.error
import threading
import atexit

# ── 路径 ─────────────────────────────────────────────────────────────────
VENV_PYTHON = os.path.join(os.environ.get("LOCALAPPDATA", ""), "freeToken", "venv", "Scripts", "python.exe")
if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    DEV_PYTHON_DIR = sys._MEIPASS  # 单一权威：exe 冻结态只用打包副本，杜绝混源
else:
    DEV_PYTHON_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "freetoken")
DAEMON_HOST = "127.0.0.1"
DAEMON_PORT = 1900
DAEMON_URL = f"http://{DAEMON_HOST}:{DAEMON_PORT}"

_daemon_proc = None

# ── 窗口尺寸记忆 ─────────────────────────────────────────
WIN_CFG = os.path.join(os.environ.get("LOCALAPPDATA", ""), "freeToken", "window.json")

def load_win_size():
    """读取上次关闭时的窗口尺寸；默认更紧凑的 1020x680。"""
    try:
        with open(WIN_CFG, "r", encoding="utf-8") as fh:
            d = json.load(fh)
            w = int(d.get("w", 1020)); h = int(d.get("h", 680))
    except Exception:
        w, h = 1020, 680
    return max(760, min(w, 3840)), max(520, min(h, 2160))

def save_win_size(win):
    try:
        os.makedirs(os.path.dirname(WIN_CFG), exist_ok=True)
        with open(WIN_CFG, "w", encoding="utf-8") as fh:
            json.dump({"w": int(win.width), "h": int(win.height)}, fh)
    except Exception:
        pass

def find_venv() -> str | None:
    """Find a usable Python interpreter with freetoken installed."""
    if os.path.isfile(VENV_PYTHON):
        return VENV_PYTHON
    # 也检查 venv 的备选路径
    for p in [
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "freeToken", "venv", "Scripts", "python.exe"),
        os.path.join(os.environ.get("USERPROFILE", ""), "AppData", "Local", "freeToken", "venv", "Scripts", "python.exe"),
    ]:
        if os.path.isfile(p):
            return p
    return None

def _ensure_port_free() -> None:
    """Kill any process holding DAEMON_PORT TCP that is NOT a live daemon."""
    try:
        req = urllib.request.Request(f"{DAEMON_URL}/health")
        with urllib.request.urlopen(req, timeout=2) as resp:
            if resp.status == 200:
                # 健康不等于自家人：旧版安装守护(无 /engine/ready)也回 200。
                # 只有应答了新路由才放行，否则视为外来实例，落到下方清场。
                try:
                    urllib.request.urlopen(f"{DAEMON_URL}/engine/ready", timeout=2)
                    return
                except urllib.error.HTTPError as e2:
                    if e2.code != 404:
                        return
                except Exception:
                    return
    except Exception:
        pass
    import subprocess as _sp
    try:
        port = DAEMON_PORT
        _sp.run(
            "for /f " + chr(0x22) + "tokens=5" + chr(0x22) + " %a in ("
            + chr(0x27) + "netstat -ano ^| findstr " + chr(0x22) + ":" + str(port) + ".*LISTENING" + chr(0x22) + chr(0x27)
            + ") do @taskkill /F /PID %a",
            shell=True, timeout=10,
        )
        time.sleep(1.0)
    except Exception:
        pass



def _pid_alive_win(pid: int) -> bool:
    """Windows 进程存活探测（无需 psutil）。"""
    import ctypes
    k32 = ctypes.windll.kernel32
    h = k32.OpenProcess(0x1000, False, int(pid))  # PROCESS_QUERY_LIMITED_INFORMATION
    if h:
        k32.CloseHandle(h)
        return True
    return False


def _clear_stale_pidfile() -> None:
    """Remove the daemon pidfile when its holder is no longer alive."""
    try:
        base = os.path.join(os.environ.get("USERPROFILE", ""), ".freetoken", "daemon")
        path = os.path.join(base, "daemon.pid")
        if not os.path.isfile(path):
            return
        raw = open(path, "r", encoding="utf-8", errors="replace").read().strip()
        pid = None
        try:
            import json as _json
            pid = int(_json.loads(raw).get("pid"))
        except Exception:
            digits = "".join(ch for ch in raw if ch.isdigit())
            pid = int(digits) if digits else None
        def _IsFreTokenDaemon(p):
            try:
                out = subprocess.check_output(
                    ["powershell", "-NoProfile", "-Command",
                     "(Get-CimInstance Win32_Process -Filter \"ProcessId=%d\").CommandLine" % p],
                    text=True, timeout=8)
                low = (out or "").lower()
                return ("freetoken.cli daemon" in low) or ("ft.exe daemon" in low)
            except Exception:
                return False
        if pid is not None and _pid_alive_win(pid) and _IsFreTokenDaemon(pid):
            return  # 真守护进程健在：交回其自身互斥逻辑
        os.remove(path)  # 死锁 / PID 复用误判：一律清障
        print(f"已清理僵尸 daemon.pid (pid={pid})")
    except Exception as exc:
        print(f"daemon.pid 预检跳过: {exc}")


def start_daemon() -> subprocess.Popen | None:
    """Start the dev freetoken daemon, return the process handle."""
    _ensure_port_free()
    python = find_venv()
    if not python:
        print("未找到 freetoken 运行时环境，请先安装 FreeToken 桌面版")
        return None
    # 检查 dev 版 freetoken 目录
    dev_dir = DEV_PYTHON_DIR
    if not os.path.isdir(os.path.join(dev_dir, "freetoken")):
        print(f"未找到打包的 freetoken 副本: {dev_dir}")
        return None
    if not os.path.isdir(os.path.join(dev_dir, "freetoken")):
        print(f"未找到 freetoken 开发代码目录: {dev_dir}")
        return None
    env = os.environ.copy()
    env["PYTHONPATH"] = dev_dir
    # venv 的 kernel-cache wheel 与 dev/bundle 代码版本戳来自不同构建（已验证兼容）；
    # 不关掉版本戳校验引擎 load_jit 启动即 RuntimeError（serve_manager 也有同款兜底）。
    env.setdefault("FREETOKEN_DISABLE_KERNEL_CACHE_VERSION_CHECK", "1")
    # 探索器环境可能残留的 Python 配置会破坏 venv 解释器启动
    for _k in ("PYTHONHOME", "PYTHONSTARTUP", "PYTHONEXECUTABLE"):
        env.pop(_k, None)
    cmd = [python, "-u", "-m", "freetoken.cli", "daemon",
           "--host", DAEMON_HOST, "--port", str(DAEMON_PORT),
           "--log-level", "warning",
           "--stop-serve-on-exit"]
    CREATE_NO_WINDOW = 0x08000000
    proc = subprocess.Popen(cmd, env=env,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            creationflags=CREATE_NO_WINDOW)
    global _daemon_proc
    _daemon_proc = proc
    return proc

def wait_for_daemon(timeout: float = 30.0) -> bool:
    """Wait for the daemon health endpoint to respond."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            req = urllib.request.Request(f"{DAEMON_URL}/health")
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        time.sleep(0.5)
    return False

def stop_daemon() -> None:
    """Stop engine + daemon fully. Closing the UI must leave nothing behind."""
    global _daemon_proc
    # 1) 请求优雅关闭（daemon 会先停引擎再退出）
    try:
        req = urllib.request.Request(f"{DAEMON_URL}/shutdown",
                                      data=b"{}",
                                      headers={"Content-Type": "application/json"},
                                      method="POST")
        urllib.request.urlopen(req, timeout=5)
    except Exception:
        pass
    # 2) 给足时间让 daemon 完成“停引擎→退出”全流程（大模型收尾可达十几秒）
    if _daemon_proc and _daemon_proc.poll() is None:
        try:
            _daemon_proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass
    # 3) 兜底：整树强杀（覆盖引擎未被优雅停止的情况）
    if _daemon_proc and _daemon_proc.poll() is None:
        try:
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(_daemon_proc.pid)],
                           capture_output=True, timeout=10)
        except Exception:
            try:
                _daemon_proc.kill()
            except Exception:
                pass
    # 3.5) FreeToken 残留清扫：引擎孤儿调度进程 / checkpoint 子进程
    try:
        subprocess.run(["powershell", "-NoProfile", "-Command",
            "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
            "Where-Object { $_.CommandLine -match 'freetoken|multiprocessing.spawn' } | "
            "ForEach-Object { Stop-Process -Id $($_.ProcessId) -Force }"],
            capture_output=True, timeout=25)
    except Exception:
        pass
    # 4) 双保险：端口兜底清理（防止任何残留持有者）
    try:
        subprocess.run(["taskkill", "/T", "/F", "/PID",
                        _port_holder_pid(1919)], capture_output=True, timeout=10)
    except Exception:
        pass

def _port_holder_pid(port):
    try:
        out = subprocess.check_output(
            ["netstat", "-ano"], text=True, timeout=10)
        for line in out.splitlines():
            parts = line.split()
            if len(parts) >= 5 and parts[3] == "LISTENING" and parts[1].endswith(":" + str(port)):
                return int(parts[4])
    except Exception:
        pass
    return -1

class Bridge:
    """Exposed to panel.js as window.pywebview.api"""

    # 面板设置的本地持久化文件 (localStorage 在某些 webview2 配置下跨 session 不持久,
    # 这里用 Python 端直接落 JSON 文件, 跨应用/跨重启都稳定)
    _SETTINGS_FILE = os.path.join(
        os.environ.get("LOCALAPPDATA", ""), "freeToken", "panel_settings.json"
    )

    def _read_settings(self):
        try:
            with open(self._SETTINGS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _write_settings(self, obj):
        try:
            os.makedirs(os.path.dirname(self._SETTINGS_FILE), exist_ok=True)
            with open(self._SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=1)
            return True
        except Exception:
            return False

    def get_settings(self):
        """从本地文件读取设置（pywebview JS API 必须返回可序列化值）"""
        return self._read_settings()

    def set_settings(self, obj):
        """写入设置到本地文件"""
        if not isinstance(obj, dict):
            return False
        return self._write_settings(obj)

    def select_folder(self):
        import webview
        try:
            result = webview.windows[0].create_file_dialog(
                webview.FOLDER_DIALOG, allow_multiple=False)
            if isinstance(result, (list, tuple)):
                return result[0] if len(result) else None
            return result or None
        except Exception:
            pass
        return None


def main():
    # ── 单实例锁：重复启动时聚焦已有窗口并退出 ──
    import ctypes
    _k32 = ctypes.windll.kernel32
    _k32.CreateMutexW(None, False, "Global\\MyTokenSingleInstance")
    if _k32.GetLastError() == 183:  # ERROR_ALREADY_EXISTS
        try:
            _u = ctypes.windll.user32
            _hwnd = _u.FindWindowW(None, "MyToken - 本地 LLM 引擎")
            if _hwnd:
                _u.ShowWindow(_hwnd, 9)          # SW_RESTORE
                _u.SetForegroundWindow(_hwnd)
        except Exception:
            pass
        sys.exit(0)
    """PyInstaller entry point."""
    # 启动 daemon（失败自动清场重试一次）
    print("正在启动 MyToken 引擎...")
    proc = start_daemon()
    ok = bool(proc) and wait_for_daemon(30)
    if not ok:
        print("首次启动未就绪，清理残留后重试…")
        try: stop_daemon()
        except Exception: pass
        _clear_stale_pidfile()
        _ensure_port_free()
        time.sleep(1.0)
        proc = start_daemon()
        ok = bool(proc) and wait_for_daemon(30)
    if not ok:
        _err = "MyToken 服务启动失败\n\n可能原因：安全软件拦截 / 端口 1900 被占用 / 运行时损坏。\n详情见 %LOCALAPPDATA%\\freeToken\\launcher-last-error.txt"
        try:
            import os as _os
            _p = _os.path.join(_os.environ.get("LOCALAPPDATA",""), "freeToken")
            _os.makedirs(_p, exist_ok=True)
            open(_os.path.join(_p, "launcher-last-error.txt"), "w", encoding="utf-8").write(
                "daemon failed to become healthy; pidfile=%s" % (proc and proc.pid or "-"))
        except Exception: pass
        try:
            ctypes.windll.user32.MessageBoxW(None, _err, "MyToken", 0x10)
        except Exception:
            input("启动失败，按回车退出")
        sys.exit(1)
    print(f"引擎已就绪: {DAEMON_URL}")
    # 注册退出清理
    atexit.register(stop_daemon)
    # 打开 pywebview 窗口
    try:
        import webview
        # 先开一个启动提示窗口
        w0, h0 = load_win_size()
        # 持久化由 Bridge.get_settings/set_settings 走 %LOCALAPPDATA%/freeToken/panel_settings.json
        # （pywebview 6.2 不支持 storage_path 参数, 改走 js_api 落 JSON 文件更可靠）
        window = webview.create_window(
            title="MyToken - 本地 LLM 引擎",
            url=f"{DAEMON_URL}/panel",
            width=w0,
            height=h0,
            resizable=True,
            min_size=(760, 520),
            js_api=Bridge(),
        )
        def _on_closed():
            save_win_size(window)
        try:
            window.events.closed += _on_closed
        except Exception:
            pass
        webview.start(debug=False)
    except ImportError:
        # 没有 pywebview 时退到浏览器
        import webbrowser
        webbrowser.open(f"{DAEMON_URL}/panel")
        print(f"打开浏览器: {DAEMON_URL}/panel")
        print("按 Ctrl+C 退出")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    finally:
        stop_daemon()

if __name__ == "__main__":
    main()