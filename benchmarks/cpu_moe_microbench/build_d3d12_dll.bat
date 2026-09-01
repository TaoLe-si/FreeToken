@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\Common7\Tools\VsDevCmd.bat" -arch=x64 >nul 2>&1
cl /nologo /O2 /LD d3d12_gemv_dll.cpp /Fe:d3d12_gemv.dll /link /DLL
