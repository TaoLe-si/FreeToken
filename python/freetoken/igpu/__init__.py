"""iGPU HIP worker subprocess for cross-process MoE decode.

See FORM2_CROSS_PROCESS_DESIGN.md for the architecture and HIP_WORKER_PITFALLS.md
for the in-engine failure modes that forced the cross-process split.

The package is split into five modules:
- protocol: shared-memory ring buffer layout + atomics
- worker:    torch-free HIP process (called as `python -m freetoken.igpu.worker`)
- client:    engine-side handle that posts requests to the worker
- supervisor: daemon-side spawn / kill / health-watch for the worker
"""
