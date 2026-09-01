import zmq.asyncio as z, inspect
cls = z._AsyncSocket
for m in ("_get_loop", "_loop"):
    a = getattr(cls, m, None)
    if a is not None:
        try:
            print("=== "+m+" ===")
            print(inspect.getsource(a.fget if isinstance(a,property) else a)[:800])
        except Exception as e:
            print(m, "err", e)
