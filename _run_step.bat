@echo off
setlocal
set PATH=C:\Program Files\AMD\ROCm\6.4\bin;%PATH%
cd /d "E:\FreeToken\benchmarks\cpu_moe_microbench"
bench_mtp_step.exe 2>&1
