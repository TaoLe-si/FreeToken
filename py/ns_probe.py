import json, time, urllib.request
t0=time.time()
body=json.dumps({"model":"q","messages":[{"role":"user","content":"hi"}],
                 "max_tokens":8,"stream":False}).encode()
req=urllib.request.Request("http://127.0.0.1:1919/v1/chat/completions",data=body,
                           headers={"Content-Type":"application/json"})
out={}
try:
    r=urllib.request.urlopen(req,timeout=90)
    out["body"]=r.read().decode("utf-8","replace")[:800]
    out["ok"]=True
except Exception as e:
    body_txt=""
    if hasattr(e,"read"):
        try: body_txt=e.read().decode("utf-8","replace")[:400]
        except Exception: pass
    out["err"]=str(e)[:120]; out["err_body"]=body_txt
out["t"]=round(time.time()-t0,1)
with open(r"E:\FreeToken\py\ns_result.json","w") as f: json.dump(out,f)
