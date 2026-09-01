
import sys
sys.path.insert(0, r"E:\FreeToken\python")
from freetoken.daemon.serve_manager import spawn_serve, build_serve_command
print("spawn_serve imported OK")
# 不能直接调（需要 self），但可以确认 import 不报错
