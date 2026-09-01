import asyncio, os, sys
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
sys.path.insert(0, r"E:\FreeToken\python")
os.chdir(r"E:\FreeToken\python")
from freetoken.server.launch import launch_server

if __name__ == "__main__":
    launch_server()
