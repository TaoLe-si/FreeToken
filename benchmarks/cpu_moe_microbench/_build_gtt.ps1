$env:PATH = "C:\Program Files\AMD\ROCm\6.4\bin;" + $env:PATH
Set-Location E:\FreeToken\benchmarks\cpu_moe_microbench
& hipcc --offload-arch=gfx1103 -O3 -o hip_gtt_bw.exe hip_gtt_bw.hip 2>&1 | Out-String
if (Test-Path .\hip_gtt_bw.exe) { Write-Output "BUILD_OK" } else { Write-Output "BUILD_FAILED" }