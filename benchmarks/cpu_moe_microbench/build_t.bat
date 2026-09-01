@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\Common7\Tools\VsDevCmd.bat" -arch=x64 >nul 2>&1
cl /O2 /arch:AVX512 /std:c++17 /EHsc /nologo t_avx512.cpp /Fe:t_avx512.exe
