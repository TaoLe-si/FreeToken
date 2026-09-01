$env:PATH = "C:\Program Files\AMD\ROCm\6.4\bin;" + $env:PATH
Set-Location "E:\FreeToken\benchmarks\cpu_moe_microbench"
& hipcc --offload-arch=gfx1103 hip_nvfp4_gemv.hip -o hip_nvfp4_gemv.exe 2>&1 | Out-String
if ($LASTEXITCODE -ne 0 -or -not (Test-Path hip_nvfp4_gemv.exe)) { "BUILD FAILED" } else { "BUILD OK " + (Get-Item hip_nvfp4_gemv.exe).Length + " bytes " + (Get-Item hip_nvfp4_gemv.exe).LastWriteTime }
