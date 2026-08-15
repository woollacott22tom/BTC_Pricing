"""
Tracks a local order book from Crypto.com Exchange's public "book" channel.

IMPORTANT CAVEAT: unlike order_book.py (Coinbase) and kraken_order_book.py,
this was built WITHOUT being able to verify against Crypto.com's actual
live message shape -- their docs site is JS-rendered and didn't return
real content when fetched. Only the channel naming convention
("book.{instrument_name}.{depth}") was confirmed from search results.

This is written defensively: it tries several plausible field-name
variants for each price level (Crypto.com's docs have used different
conventions across API versions -- [price, qty] pairs, [price, qty,
num_orders] triples, and named p/q vs price/quantity keys have all
appeared in different doc snippets found). If the live shape doesn't
match any of these, apply_message() logs nothing silently -- pair this
with the raw-message debug logging in crypto_ingestion.py to see the
actual shape and correct this file quickly, the same way Kraken's
integration was validated and fixed on first real deployment.
"""
from __future__ import annotations
import logging

log = logging.getLogger("crypto_order_book")


def _parse_level(level) -> tuple[float, float] | None:
    """Tries multiple plausible level formats defensively:
      - [price, qty] or [price, qty, num_orders] (list/tuple form)
      - {"price": ..., "qty": ...} or {"p": ..., "q": ...} (dict form)
    Returns (price, qty) or None if nothing recognizable matched."""
    try:
        if isinstance(level, (list, tuple)) and len(level) >= 2:
            return float(level[0]), float(level[1])
        if isinstance(level, dict):
            price = level.get("price", level.get("p"))
            qty = level.get("qty", level.get("q", level.get("quantity")))
            if price is not None and qty is not None:
                return float(price), float(qty)
    except (TypeError, ValueError):
        pass
    return None


class CryptoComOrderBook:
    def __init__(self):
        self.bids: dict[float, float] = {}
        self.asks: dict[float, float] = {}
        self.ready = False

    def apply_message(self, data_entry: dict, is_snapshot: bool = True):
        """IMPORTANT: Crypto.com's book channel, confirmed from live data,
        pushes a FULL replacement snapshot roughly every 500ms -- NOT
        sparse incremental deltas like Coinbase/Kraken. is_snapshot
        defaults to True and should stay True on every call; treating
        pushes as incremental merges (the original design here) would
        let stale price levels accumulate forever, since a level that
        silently disappears between snapshots never gets an explicit
        qty=0 to signal its removal."""
        bids_raw = data_entry.get("bids", [])
        asks_raw = data_entry.get("asks", [])

        if is_snapshot:
            self.bids = {}
            self.asks = {}

        parsed_any = False
        for level in bids_raw:
            parsed = _parse_level(level)
            if parsed is None:
                continue
            parsed_any = True
            price, qty = parsed
            if qty == 0.0:
                self.bids.pop(price, None)
            else:
                self.bids[price] = qty

        for level in asks_raw:
            parsed = _parse_level(level)
            if parsed is None:
                continue
            parsed_any = True
            price, qty = parsed
            if qty == 0.0:
                self.asks.pop(price, None)
            else:
                self.asks[price] = qty

        if not parsed_any and (bids_raw or asks_raw):
            log.warning(f"Could not parse any levels from book message -- "
                        f"unrecognized shape. Sample bid: {bids_raw[0] if bids_raw else None}")

        if is_snapshot:
            self.ready = True

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
