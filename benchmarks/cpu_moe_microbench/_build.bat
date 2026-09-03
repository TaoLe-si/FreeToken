@echo off
setlocal
call "C:\Program Files\AMD\ROCm\6.4\bin\hipcc.bat" --target=hip -O3 -shared -std=c++17 -o hip_moe_dll.dll hip_moe_dll.hip
