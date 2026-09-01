import sys, traceback
sys.path.insert(0, r'E:\FreeToken\python')

# Monkey-patch load_jit 和 indexing 来追踪
import freetoken.kernel.utils as ku
orig_load_jit = ku.load_jit
def patched_load_jit(*a, **kw):
    print('[TRACE] load_jit called:', a[:2], list(a[2:5]), file=sys.stderr)
    traceback.print_stack(file=sys.stderr)
    return orig_load_jit(*a, **kw)
ku.load_jit = patched_load_jit

import freetoken.kernel.index as ki
orig_indexing = ki.indexing
def patched_indexing(*a, **kw):
    print('[TRACE] indexing called', file=sys.stderr)
    return orig_indexing(*a, **kw)
ki.indexing = patched_indexing
ki._jit_index_module = ki.indexing.__wrapped__ if hasattr(ki.indexing, '__wrapped__') else ki._jit_index_module

# Patch before any other imports
from freetoken.server.launch import launch_server
launch_server()