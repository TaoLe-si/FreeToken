import urllib.request, json, time, sys
prompt = "Write a 200-word essay on the history of the internet."
body = json.dumps({"model": "Qwen3.6-35B-A3B-MXFP4-MTP-NVFP4", "messages": [{"role": "user", "content": prompt}], "max_tokens": 200, "stream": True}).encode()
req = urllib.request.Request("http://127.0.0.1:1919/v1/chat/completions", data=body, headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
t0 = time.time()
try:
    r = urllib.request.urlopen(req, timeout=600)
    last_t = t0
    tokens = []
    chunk_count = 0
    last_content = ""
    for raw in r:
        line = raw.decode('utf-8', errors='replace').strip()
        if not line.startswith('data: '):
            continue
        chunk_count += 1
        cur_t = time.time()
        dt = (cur_t - last_t) * 1000
        last_t = cur_t
        try:
            obj = json.loads(line[6:])
            delta = obj.get('choices', [{}])[0].get('delta', {})
            content = delta.get('content', '')
            if content:
                tokens.append(content)
                last_content += content
                if chunk_count <= 50:
                    print(f"  t={dt:6.1f}ms chunk={chunk_count}: '{content}' (total={len(tokens)} chars)", flush=True)
        except:
            pass
    elapsed = time.time() - t0
    print(f"=== DONE: {chunk_count} chunks, {elapsed:.1f}s, {len(tokens)} content chunks, total chars: {sum(len(t) for t in tokens)}")
    print(f"Stream output:\n{''.join(tokens)}")
except Exception as e:
    print(f"FAIL: {e}")
