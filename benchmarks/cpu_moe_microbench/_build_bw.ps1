$env:PATH = "C:\\Program Files\\AMD\\ROCm\\6.4\bin;" + $env:PATH
Set-Location "E:\\FreeToken\\benchmarks\\cpu_moe_microbench"
& hipcc --offload-arch=gfx1103 hip_shared_bw.hip -o hip_shared_bw.exe 2>&1 | Out-String
if (Test-Path hip_shared_bw.exe) { "BUILD OK " + (Get-Item hip_shared_bw.exe).Length + " bytes" } else { "BUILD FAILED" }
