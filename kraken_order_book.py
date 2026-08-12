"""
Tracks a local order book from Kraken's public WebSocket v2 "book" channel.
No auth required (unlike Kraken's level3/order-level channel, which
explicitly needs a token -- confirmed from Kraken's own docs).

Message shapes:

  Snapshot:
  {
    "channel": "book", "type": "snapshot",
    "data": [{
      "symbol": "BTC/USD",
      "bids": [{"price": 65000.0, "qty": 1.5}, ...],
      "asks": [{"price": 65001.0, "qty": 2.0}, ...],
      "checksum": 123456789
    }]
  }

  Update (same bids/asks shape, but deltas -- qty=0 means remove that
  price level, same convention used for Coinbase):
  {
    "channel": "book", "type": "update",
    "data": [{"symbol": "BTC/USD", "bids": [...], "asks": [...], "checksum": ...}]
  }

Note Kraken sends numeric price/qty directly (floats), not strings like
Coinbase does -- handled defensively either way.
"""
from __future__ import annotations


class KrakenOrderBook:
    def __init__(self):
        self.bids: dict[float, float] = {}
        self.asks: dict[float, float] = {}
        self.ready = False

    def apply_message(self, msg_type: str, data_entry: dict):
        for level in data_entry.get("bids", []):
            self._apply_level(self.bids, level)
        for level in data_entry.get("asks", []):
            self._apply_level(self.asks, level)
        if msg_type == "snapshot":
            self.ready = True

    @staticmethod
    def _apply_level(book_side: dict[float, float], level: dict):
        try:
            price = float(level["price"])
            qty = float(level["qty"])
        except (KeyError, TypeError, ValueError):
            return
        if qty == 0.0:
            book_side.pop(price, None)
        else:
            book_side[price] = qty

    def best_bid(self) -> float | None:
        return max(self.bids) if self.bids else None

    def best_ask(self) -> float | None:
        return min(self.asks) if self.asks else None

    def bid_depth(self, n: int) -> float | None:
        if not self.bids:
            return None
        top_prices = sorted(self.bids, reverse=True)[:n]
        return sum(self.bids[p] for p in top_prices)

    def ask_depth(self, n: int) -> float | None:
        if not self.asks:
            return None
        top_prices = sorted(self.asks)[:n]
        return sum(self.asks[p] for p in top_prices)
