import asyncio, threading, time, sys
sys.path.insert(0, r"E:\FreeToken\python")
import zmq, zmq.asyncio

ADDR = "tcp://127.0.0.1:15555"

def sender():
    ctx = zmq.Context()
    s = ctx.socket(zmq.PUSH)
    s.connect(ADDR)
    time.sleep(0.8)
    for i in range(3):
        s.send(("msg-%d" % i).encode())
        time.sleep(0.1)

async def main():
    ctx = zmq.asyncio.Context()
    pull = ctx.socket(zmq.PULL)
    pull.bind(ADDR)
    got = []
    async def reader():
        while len(got) < 3:
            m = await pull.recv()
            got.append(m.decode())
    t = threading.Thread(target=sender, daemon=True)
    t.start()
    try:
        await asyncio.wait_for(reader(), timeout=5)
        print("RECV OK:", got)
    except asyncio.TimeoutError:
        print("TIMEOUT, got:", got)

loop = asyncio.SelectorEventLoop()
asyncio.set_event_loop(loop)
loop.run_until_complete(main())
