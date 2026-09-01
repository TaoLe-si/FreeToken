@echo off
rem Build the CPU MoE microbenchmark with VS2026 (MSVC).
rem Usage: build.bat          -> builds both bench_avx2.exe and bench_avx512.exe
rem        build.bat avx512   -> only the AVX-512 build
setlocal
set VSDIR=C:\Program Files\Microsoft Visual Studio\18\Enterprise
call "%VSDIR%\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 >nul
set CL=%VSDIR%\VC\Tools\MSVC\14.51.36231\bin\Hostx64\x64\cl.exe
if "%1"=="avx2" goto avx2
if "%1"=="avx512" goto avx512
"%CL%" /nologo /O2 /arch:AVX2 /std:c++20 /EHsc cpu_moe_microbench.cpp /Fe:bench_avx2.exe || exit /b 1
:avx512
"%CL%" /nologo /O2 /arch:AVX512 /std:c++20 /EHsc cpu_moe_microbench.cpp /Fe:bench_avx512.exe || exit /b 1
:avx2
"%CL%" /nologo /O2 /arch:AVX2 /std:c++20 /EHsc cpu_moe_microbench.cpp /Fe:bench_avx2.exe || exit /b 1
echo done.
