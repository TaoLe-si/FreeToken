@echo off
cd /d E:\\FreeToken\\benchmarks\\cpu_moe_microbench
echo aaaaaaaaaaaaaaaaaaaa > t_test_input2.txt
type t_test_input2.txt | t_stdin_test2.exe 1> t_out2.txt 2> t_err2.txt
type t_err2.txt
type t_out2.txt
