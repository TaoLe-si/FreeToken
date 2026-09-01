import subprocess, sys
r = subprocess.run([sys.executable, r"E:\FreeToken\benchmarks\cpu_moe_microbench\_leaf.py"], capture_output=True, text=True, timeout=240)
print(r.stdout[-500:])
print("rc:", r.returncode)