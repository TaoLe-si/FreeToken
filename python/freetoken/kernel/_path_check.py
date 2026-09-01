import os
p = os.path.dirname(os.path.abspath(__file__))
print('__file__ dir:', p)
cand = os.path.join(p, "..", "..", "..", "..", "benchmarks", "cpu_moe_microbench", "t_mxfp4_gemv_server.exe")
print('candidate:', os.path.abspath(cand))
print('exists:', os.path.exists(os.path.abspath(cand)))
