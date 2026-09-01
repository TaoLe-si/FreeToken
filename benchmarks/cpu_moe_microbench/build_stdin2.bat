@echo off
cd /d E:\\FreeToken\\benchmarks\\cpu_moe_microbench
setlocal
set MSVC=C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Tools\MSVC\14.51.36231
set WINSDK=C:\Program Files (x86)\Windows Kits\10
"%MSVC%\bin\Hostx64\x64\cl.exe" /nologo /EHsc /O2 /std:c++17 ^
  /I"%MSVC%\include" /I"%WINSDK%\Include\10.0.26100.0\ucrt" /I"%WINSDK%\Include\10.0.26100.0\um" /I"%WINSDK%\Include\10.0.26100.0\shared" ^
  t_stdin_test2.cpp /Fe:t_stdin_test2.exe /link /LIBPATH:"%MSVC%\lib\x64" /LIBPATH:"%WINSDK%\Lib\10.0.26100.0\ucrt\x64" /LIBPATH:"%WINSDK%\Lib\10.0.26100.0\um\x64"
