"""
Tracks a local order book from Coinbase Advanced Trade's authenticated
"level2" WebSocket channel. Message shape differs from the old Exchange
product's level2 feed:

  {
    "channel": "l2_data",
    "events": [
      {
        "type": "snapshot" | "update",
        "product_id": "BTC-USD",
        "updates": [
          {"side": "bid" | "offer", "price_level": "65000.00", "new_quantity": "1.5"},
          ...
        ]
      }
    ]
  }

Note Advanced Trade uses "offer" where the old Exchange feed used "sell"/"ask".
A "snapshot" event's updates list represents the full initial book; "update"
events are incremental deltas after that. new_quantity of "0" means remove
that price level.
"""
from __future__ import annotations


class OrderBook:
    def __init__(self):
        self.bids: dict[float, float] = {}
        self.asks: dict[float, float] = {}
        self.ready = False

    def apply_event(self, event: dict):
        updates = event.get("updates", [])
        for u in updates:
            side = u.get("side")
            try:
                price = float(u["price_level"])
                size = float(u["new_quantity"])
            except (KeyError, TypeError, ValueError):
                continue
            book_side = self.bids if side == "bid" else self.asks
            if size == 0.0:
                book_side.pop(price, None)
            else:
                book_side[price] = size
        if event.get("type") == "snapshot":
            self.ready = True

    def best_bid(self) -> float | None:
        return max(self.bids) if self.bids else None

    def best_ask(self) -> float | None:
        return min(self.asks) if self.asks else None

    def top10_bid_depth(self) -> float | None:
        if not self.bids:
            return None
        top_prices = sorted(self.bids, reverse=True)[:10]
        return sum(self.bids[p] for p in top_prices)

    def top10_ask_depth(self) -> float | None:
        if not self.asks:
            return None
        top_prices = sorted(self.asks)[:10]
        return sum(self.asks[p] for p in top_prices)
