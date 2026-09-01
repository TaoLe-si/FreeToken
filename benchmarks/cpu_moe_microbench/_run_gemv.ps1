$env:PATH = "C:\Program Files\AMD\ROCm\6.4\bin;" + $env:PATH
& "E:\FreeToken\benchmarks\cpu_moe_microbench\hip_nvfp4_gemv.exe" 2>&1 | Out-String
