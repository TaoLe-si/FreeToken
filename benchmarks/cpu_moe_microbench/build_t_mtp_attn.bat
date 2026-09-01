@echo off
REM Build t_mtp_attn_server.exe (Phase 2.4 attn fused HLSL server, 2026-08-29).
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\Common7\Tools\VsDevCmd.bat" -arch=x64 >nul 2>&1
setlocal
set DXC=C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Tools\Llvm\x64\bin\dxc.exe
if not exist "%DXC%" (
  echo DXC not found at %DXC%. Edit build_t_mtp_attn.bat to point at your dxc.exe
  exit /b 1
)
cd /d E:\FreeToken\benchmarks\cpu_moe_microbench

"%DXC%" -T cs_6_6 -E qkv_proj_norm -Zi -Qstrip_debug -Fo t_mtp_attn_qkv.dxil    t_mtp_attn_fused.hlsl
"%DXC%" -T cs_6_6 -E rope_kvappend -Zi -Qstrip_debug -Fo t_mtp_attn_rope.dxil    t_mtp_attn_fused.hlsl
"%DXC%" -T cs_6_6 -E attn_gqa_gate -Zi -Qstrip_debug -Fo t_mtp_attn_attn.dxil    t_mtp_attn_fused.hlsl
"%DXC%" -T cs_6_6 -E o_proj        -Zi -Qstrip_debug -Fo t_mtp_attn_oproj.dxil   t_mtp_attn_fused.hlsl

cl /nologo /EHsc /O2 /std:c++17 /I . t_mtp_attn_server.cpp /Fe:t_mtp_attn_server.exe /link d3d12.lib dxgi.lib

endlocal
