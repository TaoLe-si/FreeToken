import json, time, urllib.request
t0 = time.time()
body = json.dumps({"model":"q","messages":[{"role":"user","content":"数到10，用中文"}],
                   "max_tokens":24,"stream":True}).encode()
req = urllib.request.Request("http://127.0.0.1:1919/v1/chat/completions", data=body,
                             headers={"Content-Type":"application/json"})
out = {"first": None, "chunks": 0, "text": "", "err": None, "total": None}
try:
    resp = urllib.request.urlopen(req, timeout=150)
    buf = b""
    while True:
        chunk = resp.read(4096)
        if not chunk: break
        buf += chunk
        txt = buf.decode("utf-8", "replace")
        n = txt.count('"content":"')
        if out["first"] is None and n > 0: out["first"] = round(time.time()-t0, 2)
        out["chunks"] = n
    out["text"] = buf.decode("utf-8","replace")[-500:]
    out["total"] = round(time.time()-t0, 1)
except Exception as e:
    out["err"] = str(e)[:200]
    out["total"] = round(time.time()-t0, 1)
out["ts"] = time.time()
with open(r"E:\FreeToken\py\decode_result.json", "w") as f:
    json.dump(out, f)
