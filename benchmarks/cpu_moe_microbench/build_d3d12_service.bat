@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\Common7\Tools\VsDevCmd.bat" -arch=x64 >nul 2>&1
cl /nologo /EHsc /O2 t_d3d12_service.cpp /Fe:t_d3d12_service.exe
