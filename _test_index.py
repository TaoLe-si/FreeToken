import sys, traceback
sys.path.insert(0, r'E:\FreeToken\python')
try:
    import torch
    from freetoken.kernel.index import indexing
    print('indexing imported OK')
    w = torch.randn(1024, 256, dtype=torch.float16, device='cuda')
    idx = torch.randint(0, 1024, (8,), device='cuda', dtype=torch.long)
    out = torch.empty(8, 256, dtype=torch.float16, device='cuda')
    print('calling indexing...')
    result = indexing(w, idx, output=out)
    print('SUCCESS, result shape:', result.shape)
except Exception as e:
    print('EXCEPTION CAUGHT:', type(e).__name__, str(e)[:200])
    traceback.print_exc()