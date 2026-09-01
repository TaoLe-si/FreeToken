import urllib.request, json, time, sys
body = json.dumps({"model": "Qwen3.6-35B-A3B-MXFP4-MTP-NVFP4", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 10, "stream": False, "temperature": 0}).encode()
req = urllib.request.Request("http://127.0.0.1:1919/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
t0 = time.time()
print(f"  sending request at t=0", flush=True)
try:
    r = urllib.request.urlopen(req, timeout=600)
    print(f"  status: {r.status} at t={(time.time()-t0):.1f}s", flush=True)
    content = r.read()
    print(f"  read at t={(time.time()-t0):.1f}s, len={len(content)}", flush=True)
    print(content.decode()[:500])
except Exception as e:
    print(f"  FAIL at t={(time.time()-t0):.1f}s: {e}", flush=True)
