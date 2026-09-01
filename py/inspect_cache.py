import sys
# drop source tree so site-packages (compiled) wins
sys.path = [p for p in sys.path if "E:\\FreeToken\\python" not in p]
from freetoken.moe.offload_cache import OffloadMoeCache
meths = [m for m in dir(OffloadMoeCache) if not m.startswith("_")]
print("方法:", meths)
# 关键字段
import inspect as _i
try:
    sig = _i.signature(OffloadMoeCache.__init__)
    print("__init__:", sig)
except Exception as e:
    print("sig err:", e)
