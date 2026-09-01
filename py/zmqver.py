import zmq, zmq.asyncio, inspect
print("pyzmq:", zmq.__version__)
from zmq.asyncio import Socket
src = inspect.getsource(Socket.__init__) if hasattr(Socket,"__init__") else "(no init)"
print(src[:600])
# 找 _get_loop/_loop 相关
import zmq.asyncio as z
for name in dir(z):
    if "oop" in name: print("attr:", name)
cls = z._AsyncSocket if hasattr(z,"_AsyncSocket") else None
if cls:
    for m in ("_get_loop","_loop","_init_loop"):
        if hasattr(cls,m):
            try: print(m, ":", inspect.getsource(getattr(cls,m))[:400])
            except Exception as e: print(m, "err", e)
