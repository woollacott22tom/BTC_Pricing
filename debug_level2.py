"""
One-off diagnostic: subscribes ONLY to the level2 channel (no market_trades)
and prints every message + the exact close reason, to isolate why level2
triggers an immediate disconnect while market_trades works fine.

Run manually (not as a service):
    python3 debug_level2.py
"""
import asyncio
import json
import time

import websockets

from jwt_auth import build_ws_jwt

ADVANCED_TRADE_WS = "wss://advanced-trade-ws.coinbase.com"
PRODUCT_ID = "BTC-USD"


async def main():
    print(f"Connecting to {ADVANCED_TRADE_WS} ...")
    try:
        async with websockets.connect(ADVANCED_TRADE_WS, ping_interval=20, ping_timeout=20) as ws:
            print("Connected. Building JWT...")
            token = build_ws_jwt()
            print(f"JWT built, length={len(token)}, first 40 chars: {token[:40]}...")

            subscribe_msg = {
                "type": "subscribe",
                "product_ids": [PRODUCT_ID],
                "channel": "level2",
                "jwt": token,
            }
            print(f"Sending subscribe: {json.dumps({**subscribe_msg, 'jwt': '<redacted>'})}")
            await ws.send(json.dumps(subscribe_msg))

            start = time.time()
            msg_num = 0
            try:
                async for raw in ws:
                    msg_num += 1
                    print(f"\n--- Message #{msg_num} (t+{time.time()-start:.2f}s) ---")
                    print(raw[:1000])
                    if time.time() - start > 8:
                        print("\n8 seconds elapsed with no disconnect -- looks stable, stopping test.")
                        break
            except websockets.ConnectionClosed as e:
                print(f"\n*** CONNECTION CLOSED *** code={e.code} reason={e.reason!r}")
                print(f"Received {msg_num} messages before closure, over {time.time()-start:.2f}s")

    except Exception as e:
        print(f"Connection-level exception: {type(e).__name__}: {e}")


if __name__ == "__main__":
    asyncio.run(main())
