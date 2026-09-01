
import sys
sys.path.insert(0, r"E:\FreeToken\python")
try:
    from freetoken.daemon.serve_manager import ServeManager
    sm = ServeManager(log_dir=r"C:\Users\Administrator\AppData\Local\freeToken\daemon\logs", pidfile=r"C:\Users\Administrator\AppData\Local\freeToken\daemon\state.pid")
    r = sm.start("Qwen/Qwen2.5-0.5B-Instruct", 19100, ["--memory-ratio","0.9","--disable-pynccl","--dense-ffn-engine","cpu"])
    print("OK:", r)
except Exception as e:
    import traceback
    traceback.print_exc()
    print("ERR:", e)
