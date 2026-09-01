$env:PATH = "C:\Program Files\AMD\ROCm\6.4\bin;" + $env:PATH
Set-Location E:\FreeToken\benchmarks\cpu_moe_microbench
& hipcc --offload-arch=gfx1103 -shared -o hip_moe_dll.dll hip_moe_dll.hip *>&1 | ForEach-Object { "$_" } | Select-String -Pattern "error" -Context 0,2 | Out-String