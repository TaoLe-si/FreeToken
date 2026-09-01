@echo off
REM Build t_mtp_moe_server.exe (Phase 2.3 MoE fused HLSL server, 2026-08-29).
REM Prerequisites: dxc must be in PATH (Windows SDK or DirectXShaderCompiler).
REM   Or copy dxc.exe from Visual Studio's DXC distribution.
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\Common7\Tools\VsDevCmd.bat" -arch=x64 >nul 2>&1
setlocal
set DXC=C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Tools\Llvm\x64\bin\dxc.exe
if not exist "%DXC%" (
  echo DXC not found at %DXC%. Edit build_t_mtp_moe.bat to point at your dxc.exe
  exit /b 1
)
cd /d E:\FreeToken\benchmarks\cpu_moe_microbench

REM 1) Compile HLSL to DXIL.
"%DXC%" -T cs_6_6 -E moe_route    -Zi -Qstrip_debug -Fo t_mtp_moe_route.dxil    t_mtp_moe_fused.hlsl
"%DXC%" -T cs_6_6 -E moe_expert_8x -Zi -Qstrip_debug -Fo t_mtp_moe_expert.dxil  t_mtp_moe_fused.hlsl
"%DXC%" -T cs_6_6 -E moe_shared   -Zi -Qstrip_debug -Fo t_mtp_moe_shared.dxil   t_mtp_moe_fused.hlsl
"%DXC%" -T cs_6_6 -E moe_combine  -Zi -Qstrip_debug -Fo t_mtp_moe_combine.dxil  t_mtp_moe_fused.hlsl

REM 2) Compile C++ server.
cl /nologo /EHsc /O2 /std:c++17 /I . t_mtp_moe_server.cpp /Fe:t_mtp_moe_server.exe /link d3d12.lib dxgi.lib

endlocal
