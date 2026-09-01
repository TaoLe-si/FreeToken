"""Run pyinstaller to build FreeToken.exe."""
import os
import subprocess
import sys

PYINSTALLER = r"C:\Users\Administrator\AppData\Local\freeToken\venv\Scripts\pyinstaller.exe"
spec = r"E:\FreeToken\FreeTokenDbg.spec"
cmd = [PYINSTALLER, "--clean", "--noconfirm", spec]
print(f"Running: {' '.join(cmd)}", flush=True)
os.chdir(r"E:\FreeToken")
result = subprocess.run(cmd)
print(f"\nExit code: {result.returncode}", flush=True)
sys.exit(result.returncode)
