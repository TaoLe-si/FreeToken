import sys, os, subprocess
print('CUDA_HOME:', os.environ.get('CUDA_HOME', 'unset'))
print('CUDA_PATH:', os.environ.get('CUDA_PATH', 'unset'))
# 用 v12.6 编译一段 hello world 风格的 cu 看是否 OK
test = r'''#include <stdio.h>\n__global__ void k(){printf("hi\\n");}\nint main(){k<<<1,1>>>();cudaDeviceSynchronize();return 0;}'''
import tempfile
with tempfile.NamedTemporaryFile(suffix='.cu', delete=False, mode='w') as f: f.write(test); cu=f.name
p = subprocess.run([r'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.6\\bin\\nvcc.exe', cu, '-o', cu+'.exe'], capture_output=True, text=True, timeout=60)
print('v12.6 nvcc RC:', p.returncode)
if p.stderr: print('stderr:', p.stderr[:300])