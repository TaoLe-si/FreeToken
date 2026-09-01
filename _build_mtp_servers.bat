@echo off
setlocal
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\Common7\Tools\VsDevCmd.bat" -arch=x64 >nul 2>&1
cd /d E:\FreeToken\benchmarks\cpu_moe_microbench

set DXC=E:\dxc_unzip\bin\x64\dxc.exe
if not exist "%DXC%" (
  echo DXC not found at %DXC%. Install DirectXShaderCompiler first.
  exit /b 1
)

REM --- D3D12 / DirectX path ---
"%DXC%" -T cs_6_5 -E main -Fo t_mtp_moe_route.dxil    t_mtp_moe_route.hlsl
"%DXC%" -T cs_6_5 -E main -Fo t_mtp_moe_expert.dxil  t_mtp_moe_expert.hlsl
"%DXC%" -T cs_6_5 -E main -Fo t_mtp_moe_shared.dxil   t_mtp_moe_shared.hlsl
"%DXC%" -T cs_6_5 -E main -Fo t_mtp_moe_combine.dxil  t_mtp_moe_combine.hlsl
"%DXC%" -T cs_6_5 -E main -Fo t_mtp_attn_qkv.dxil    t_mtp_attn_qkv.hlsl
"%DXC%" -T cs_6_5 -E main -Fo t_mtp_attn_rope.dxil    t_mtp_attn_rope.hlsl
"%DXC%" -T cs_6_5 -E main -Fo t_mtp_attn_attn.dxil    t_mtp_attn_attn.hlsl
"%DXC%" -T cs_6_5 -E main -Fo t_mtp_attn_oproj.dxil   t_mtp_attn_oproj.hlsl

cl /nologo /EHsc /O2 /std:c++17 /I . t_mtp_moe_server.cpp /Fe:t_mtp_moe_server.exe /link d3d12.lib dxgi.lib
cl /nologo /EHsc /O2 /std:c++17 /I . t_mtp_attn_server.cpp /Fe:t_mtp_attn_server.exe /link d3d12.lib dxgi.lib
cl /nologo /EHsc /O2 /std:c++17 /I . t_mtp_moe_route_server.cpp /Fe:t_mtp_moe_route_server.exe /link d3d12.lib dxgi.lib

REM --- ROCm / HIP path (AMD Radeon 780M, gfx1103) ---
set ROCM=C:\Program Files\AMD\ROCm\6.4
set HIPCC=%ROCM%\bin\hipcc.bat
set PATH=%ROCM%\bin;%PATH%

REM IMPORTANT: -O0 -fno-lto to prevent LTO from stripping the kernel symbol,
REM which would cause "invalid device function" at runtime.
"%HIPCC%" --offload-arch=gfx1103 -O0 -std=c++17 -fno-lto t_mtp_moe_route_hip.hip t_mtp_moe_route_hip_server.cpp -o t_mtp_moe_route_hip_server.exe 2>nul
"%HIPCC%" --offload-arch=gfx1103 -O0 -std=c++17 -fno-lto t_mxfp4_gemv_sk.hip t_mxfp4_gemv_v3_hip_server.cpp -o t_mxfp4_gemv_v3_hip_server.exe 2>nul

REM --- Copy HIP server exes to dist/bin/ for FreeToken.exe bundling ---
copy /Y t_mtp_moe_route_hip_server.exe "%~dp0dist\bin\" >nul 2>&1
copy /Y t_mxfp4_gemv_v3_hip_server.exe "%~dp0dist\bin\" >nul 2>&1

echo === MTP server build complete ===
