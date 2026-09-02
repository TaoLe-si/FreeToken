
import json,struct,subprocess,sys,time,os
import numpy as np, torch
sys.path.insert(0,'E:/FreeToken/python')
from freetoken.models.qwen3_5_moe.mtp import _dequant_mxfp4_affine

d='E:/models/Qwen3.6-35B-A3B-MXFP4-MTP-NVFP4/mtp.safetensors'
from safetensors.torch import load_file
st=load_file(d)
W=st['mtp.fc.weight']; S=st['mtp.fc.scales']; B=st['mtp.fc.biases']
print('W',tuple(W.shape),W.dtype,'S',tuple(S.shape),'B',tuple(B.shape))
M,K=2048,4096
Wn=W.numpy().astype('uint32'); Sn=S.numpy().astype('float32'); Bn=B.numpy().astype('float32')

rng=np.random.default_rng(0)
x=(rng.standard_normal(K).astype('float32')*0.5)

with torch.no_grad():
    Wdq=_dequant_mxfp4_affine(torch.from_numpy(Wn.astype('int64')), torch.from_numpy(Sn), torch.from_numpy(Bn))
    y_ref=(Wdq @ torch.from_numpy(x)).numpy()
print('y_ref norm',float(np.linalg.norm(y_ref)))

exe='E:/FreeToken/benchmarks/cpu_moe_microbench/t_mxfp4_gemv_v3_hip_server.exe.new'
env=dict(os.environ)
p=subprocess.Popen([exe],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env)
szP=M*(K//8)*4; ns=K//32; szS=M*ns*4; szBi=M*ns*4
body=Wn.tobytes()+Sn.tobytes()+Bn.tobytes()
p.stdin.write(f'FC_LOAD {M} {K} {szP} {szS} {szBi}\n'.encode()); p.stdin.write(body); p.stdin.flush()
ack=p.stdout.read(3)
print('FC_LOAD ack:',ack)
t0=time.time()
p.stdin.write(f'FC_CALL {K*4}\n'.encode()); p.stdin.write(x.tobytes()); p.stdin.flush()
rl=p.stdout.read(4); sz=struct.unpack('<I',rl)[0]; out=p.stdout.read(sz)
y_srv=np.frombuffer(out,dtype=np.float32).copy()
print('srv %.2f ms'%((time.time()-t0)*1e3),'y_srv norm',float(np.linalg.norm(y_srv)))
cos=float(np.dot(y_ref,y_srv)/(np.linalg.norm(y_ref)*np.linalg.norm(y_srv)+1e-9))
print('COS(y_ref, y_srv) =',cos)
p.kill()
