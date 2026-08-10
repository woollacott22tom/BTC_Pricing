"""
One-off diagnostic: subscribes ONLY to level2, with no auth, matching
Coinbase's documented "Sending Messages without API Keys" example exactly.
Prints every message and the precise close code/reason.

Run manually (not as a service):
    python3 debug_level2_noauth.py
"""
import asyncio
import json
import time

import websockets

ADVANCED_TRADE_WS = "wss://advanced-trade-ws.coinbase.com"


async def main():
    print(f"Connecting to {ADVANCED_TRADE_WS} ...")
    try:
        async with websockets.connect(
            ADVANCED_TRADE_WS,
            ping_interval=20,
            ping_timeout=20,
            user_agent_header="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            origin="https://www.coinbase.com",
        ) as ws:
            print("Connected.")

            subscribe_msg = {
                "type": "subscribe",
                "product_ids": ["BTC-USD"],
                "channel": "level2",
            }
            print(f"Sending: {json.dumps(subscribe_msg)}")
            await ws.send(json.dumps(subscribe_msg))

            start = time.time()
            msg_num = 0
            try:
                async for raw in ws:
                    msg_num += 1
                    print(f"\n--- Message #{msg_num} (t+{time.time()-start:.2f}s) ---")
                    print(raw[:1000])
                    if time.time() - start > 10:
                        print("\n10 seconds elapsed, stable -- stopping test.")
                        break
            except websockets.ConnectionClosed as e:
                print(f"\n*** CONNECTION CLOSED *** code={e.code} reason={e.reason!r}")
                print(f"Received {msg_num} messages before closure, over {time.time()-start:.2f}s")

    except Exception as e:
        print(f"Connection-level exception: {type(e).__name__}: {e}")


if __name__ == "__main__":
    asyncio.run(main())
