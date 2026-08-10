"""
One-off diagnostic: subscribes to l2_data (the REAL channel name -- Coinbase's
docs incorrectly show "level2" in the request example, but the server only
recognizes "l2_data", matching the name used in its own response messages)
WITH JWT auth, using the already-validated Ed25519 signing code.

Run manually (not as a service):
    python3 debug_l2data_auth.py
"""
import asyncio
import json
import time

import websockets

from jwt_auth import build_ws_jwt

ADVANCED_TRADE_WS = "wss://advanced-trade-ws.coinbase.com"


async def main():
    print(f"Connecting to {ADVANCED_TRADE_WS} ...")
    try:
        async with websockets.connect(
            ADVANCED_TRADE_WS,
            ping_interval=20,
            ping_timeout=20,
            max_size=None,  # disable the default 1MB message size limit --
                             # level2's initial snapshot can be a large payload
        ) as ws:
            print("Connected. Building JWT...")
            token = build_ws_jwt()
            print(f"JWT built, length={len(token)}")

            subscribe_msg = {
                "type": "subscribe",
                "product_ids": ["BTC-USD"],
                "channel": "level2",
                "jwt": token,
            }
            print(f"Sending: {json.dumps({**subscribe_msg, 'jwt': '<redacted>'})}")
            await ws.send(json.dumps(subscribe_msg))

            start = time.time()
            msg_num = 0
            try:
                while time.time() - start < 10:
                    remaining = 10 - (time.time() - start)
                    raw = await asyncio.wait_for(ws.recv(), timeout=max(remaining, 0.1))
                    msg_num += 1
                    print(f"\n--- Message #{msg_num} (t+{time.time()-start:.2f}s) ---")
                    print(raw[:1000])
                print(f"\n10 seconds elapsed with {msg_num} messages received.")
            except asyncio.TimeoutError:
                print(f"\n10 seconds elapsed with {msg_num} messages received -- connection open but silent.")
            except websockets.ConnectionClosed as e:
                print(f"\n*** CONNECTION CLOSED *** code={e.code} reason={e.reason!r}")
                print(f"Received {msg_num} messages before closure, over {time.time()-start:.2f}s")

    except Exception as e:
        print(f"Connection-level exception: {type(e).__name__}: {e}")


if __name__ == "__main__":
    asyncio.run(main())
