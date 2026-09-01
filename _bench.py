import http.client, json, time, sys

DAEMON, ENGINE = 1900, 1919
MODEL = "E:/models/Qwen3.6-35B-A3B-MXFP4-MTP-NVFP4"
PROMPT = "写一个Python函数计算斐波那契数列前n项,并给出示例"
MAXTOK = 256

def req(port, path, body=None, timeout=300):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        m = "POST" if body is not None else "GET"
        c.request(m, path, json.dumps(body) if body else None,
                  {"Content-Type": "application/json", "Connection": "close"})
        r = c.getresponse()
        data = r.read().decode("utf-8", "replace")
        return r.status, data
    finally:
        c.close()

def wait_ready(limit=240):
    t0 = time.time()
    while time.time() - t0 < limit:
        try:
            s, d = req(ENGINE, "/v1/chat/completions",
                       {"model": "qwen", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1})
            if s != 503:
                return True, round(time.time() - t0)
        except Exception:
            pass
        time.sleep(5)
    return False, round(time.time() - t0)

def bench(tag, args, runs=2):
    req(DAEMON, "/engine/stop", {"force": True})
    time.sleep(6)
    s, d = req(DAEMON, "/engine/start", {"model": MODEL, "port": ENGINE, "args": args})
    if s != 200:
        print(f"[{tag}] start FAIL {s} {d[:120]}")
        return
    ok, t = wait_ready()
    if not ok:
        print(f"[{tag}] NOT READY after {t}s")
        return
    rates = []
    for i in range(runs):
        t0 = time.time()
        s, d = req(ENGINE, "/v1/chat/completions",
                   {"model": "qwen", "messages": [{"role": "user", "content": PROMPT}],
                    "max_tokens": MAXTOK, "temperature": 0})
        dt = time.time() - t0
        if s != 200:
            print(f"[{tag}] run{i} HTTP {s}: {d[:150]}")
            return
        j = json.loads(d)
        n = j["usage"]["completion_tokens"]
        rates.append(n / dt)
        if i == 0:
            msg = j["choices"][0]["message"]
            head = (msg.get("reasoning_content") or msg.get("content") or "")[:70]
            print(f"[{tag}] load {t}s | tokens {n} | text: {head!r}")
    print(f"[{tag}] RESULT: " + " / ".join(f"{r:.1f}" for r in rates) + " tok/s  args={' '.join(args)}")

if __name__ == "__main__":
    cfg = json.load(open(sys.argv[1]))
    for tag, args in cfg:
        bench(tag, args)
