@echo off
rem Build hip_moe_dll.dll with hipcc (ROCm 6.4)
setlocal
set ROCM=C:\Program Files\AMD\ROCm\6.4
set PATH=%ROCM%\bin;%PATH%
set HIP_PLATFORM=amd
cd /d E:\FreeToken\benchmarks\cpu_moe_microbench
hipcc -O2 --offload-arch=gfx1103 -shared -std=c++17 hip_moe_dll.hip -o hip_moe_dll.dll -I "%ROCM%\include" -L "%ROCM%\lib" -lamdhip64 || (echo BUILD FAILED & exit /b 1)
echo BUILD OK
