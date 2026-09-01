$env:PATH = "C:\Program Files\AMD\ROCm\6.4\bin;" + $env:PATH
Set-Location E:\FreeToken\benchmarks\cpu_moe_microbench
& .\hip_gtt_bw.exe 2>&1 | Out-String