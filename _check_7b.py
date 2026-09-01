import sys, os, json
sys.path.insert(0, r'E:\FreeToken\python')
from huggingface_hub import HfApi
api = HfApi()
try:
    info = api.model_info('Qwen/Qwen2.5-7B-Instruct')
    for s in info.siblings[:8]: print(s.rfilename, getattr(s,'size',None))
except Exception as e:
    print('err:',e)