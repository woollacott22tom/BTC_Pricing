"""
Tracks a local order book from Coinbase's public "level2" WebSocket channel,
which sends one full snapshot on subscribe, then incremental diffs
(l2update messages) after that. This replaces Binance's ready-made
top-20-levels-per-message depth20 stream, which isn't available here since
Coinbase's public feed works differently.

Usage:
    book = OrderBook()
    book.apply_snapshot(bids, asks)   # once, from the "snapshot" message
    book.apply_update(changes)         # repeatedly, from "l2update" messages
    book.top10_bid_depth(), book.top10_ask_depth(), book.best_bid(), book.best_ask()
"""
from __future__ import annotations


class OrderBook:
    def __init__(self):
        self.bids: dict[float, float] = {}  # price -> size
        self.asks: dict[float, float] = {}
        self.ready = False

    def apply_snapshot(self, bids: list[list[str]], asks: list[list[str]]):
        self.bids = {float(p): float(s) for p, s in bids}
        self.asks = {float(p): float(s) for p, s in asks}
        self.ready = True

    def apply_update(self, changes: list[list[str]]):
        for side, price_str, size_str in changes:
            price, size = float(price_str), float(size_str)
            book_side = self.bids if side == "buy" else self.asks
            if size == 0.0:
                book_side.pop(price, None)
            else:
                book_side[price] = size

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
