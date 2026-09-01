
import urllib.request
import json
import time

body = json.dumps({
    "model": "Qwen3.6-35B-A3B-MXFP4-MTP-NVFP4",
    "messages": [{"role": "user", "content": "Hi"}],
    "max_tokens": 5,
    "stream": False
}).encode()
req = urllib.request.Request("http://127.0.0.1:1900/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
print("sending chat...")
t0 = time.time()
try:
    r = urllib.request.urlopen(req, timeout=180)
    print(f"chat ok in {time.time()-t0:.1f}s")
    print(r.read().decode()[:1000])
except urllib.error.HTTPError as e:
    print(f"chat error {e.code}:", e.read().decode()[:500])
except Exception as e:
    print(f"chat failed after {time.time()-t0:.1f}s:", e)
