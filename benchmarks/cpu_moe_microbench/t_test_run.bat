@echo off
cd /d E:\\FreeToken\\benchmarks\\cpu_moe_microbench
type t_test_input.bin | t_mxfp4_gemv_server.exe > t_test_output.bin 2> t_test_stderr.txt
