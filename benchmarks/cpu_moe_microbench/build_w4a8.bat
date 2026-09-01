@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\Common7\Tools\VsDevCmd.bat" -arch=x64 >nul 2>&1
cl /O2 /arch:AVX512 /std:c++17 /EHsc /nologo bench_w4a8.cpp /Fe:bench_w4a8.exe
