import sys, os
sys.path.insert(0, r'E:\FreeToken\python')
import safetensors
from safetensors import safe_open
root = r'E:\models\Qwen3.6-27B-FP8'
f = safe_open(os.path.join(root,'layers-0.safetensors'), framework='pt')
ks = list(f.keys())
print('layers-0 键数:', len(ks))
for k in ks:
    t = f.get_tensor(k)
    print(' ', k, t.shape, t.dtype, '|', round((t.numel()*t.element_size())/1024**2,1),'MB')
# 看 mtp.safetensors
import os.path as op
for fn in ['mtp.safetensors','outside.safetensors']:
    p = op.join(root,fn)
    if not op.exists(p): continue
    f2 = safe_open(p, framework='pt')
    print('\n===', fn,'===')
    for k in list(f2.keys()):
        t = f2.get_tensor(k)
        print(' ', k, t.shape, t.dtype, '|', round((t.numel()*t.element_size())/1024**2,1),'MB')