"""In-memory exchange with fault injection, for tests and dry runs."""
from __future__ import annotations

from carry.types import OrderResult


class FakeExchange:
    def __init__(self, prices: dict[str, float] | None = None,
                 funding: dict[str, list[float]] | None = None,
                 spot_fee: float = 0.001, mgn: float | None = 10.0) -> None:
        self.prices = prices or {"BTC": 60000.0, "ETH": 3000.0}
        self.funding = funding or {}
        self.spot_fee = spot_fee
        self.mgn = mgn
        self.spot: dict[str, float] = {}
        self.swap: dict[str, float] = {}
        self.adl: dict[str, int] = {}
        self.calls: list[tuple[str, str, float]] = []
        self._fail: dict[str, int] = {}
        self._partial: dict[str, float] = {}

    # --- fault injection -------------------------------------------------
    def fail_next(self, op: str, n: int = 1) -> None:
        self._fail[op] = n

    def partial_next(self, op: str, fraction: float) -> None:
        self._partial[op] = fraction

    def trigger_adl(self, coin: str, fraction: float) -> None:
        self.swap[coin] = round(self.swap.get(coin, 0.0) * (1.0 - fraction), 10)

    def _gate(self, op: str, coin: str, qty: float) -> OrderResult | None:
        self.calls.append((op, coin, qty))
        if self._fail.get(op, 0) > 0:
            self._fail[op] -= 1
            return OrderResult(False, error=f"injected {op} failure")
        return None

    def _frac(self, op: str) -> float:
        return self._partial.pop(op, 1.0)

    # --- market data / account -------------------------------------------
    def price(self, coin: str) -> float:
        return self.prices[coin]

    def funding_history(self, coin: str, n: int) -> list[float]:
        return list(self.funding.get(coin, []))[-n:]

    def spot_balance(self, coin: str) -> float:
        return self.spot.get(coin, 0.0)

    def swap_contracts(self, coin: str) -> float:
        return self.swap.get(coin, 0.0)

    def swap_adl_rank(self, coin: str) -> int:
        return self.adl.get(coin, 0)

    def mgn_ratio(self) -> float | None:
        return self.mgn

    # --- orders (market, fill at current price) --------------------------
    def sell_swap(self, coin: str, contracts: float) -> OrderResult:
        bad = self._gate("sell_swap", coin, contracts)
        if bad:
            return bad
        q = contracts * self._frac("sell_swap")
        self.swap[coin] = self.swap.get(coin, 0.0) + q
        return OrderResult(True, q, self.prices[coin])

    def buy_swap(self, coin: str, contracts: float) -> OrderResult:
        bad = self._gate("buy_swap", coin, contracts)
        if bad:
            return bad
        q = min(contracts * self._frac("buy_swap"), self.swap.get(coin, 0.0))
        self.swap[coin] = self.swap.get(coin, 0.0) - q
        return OrderResult(True, q, self.prices[coin])

    def buy_spot(self, coin: str, qty: float) -> OrderResult:
        bad = self._gate("buy_spot", coin, qty)
        if bad:
            return bad
        q = qty * self._frac("buy_spot")
        self.spot[coin] = self.spot.get(coin, 0.0) + q * (1.0 - self.spot_fee)
        return OrderResult(True, q, self.prices[coin])

    def sell_spot(self, coin: str, qty: float) -> OrderResult:
        bad = self._gate("sell_spot", coin, qty)
        if bad:
            return bad
        q = min(qty * self._frac("sell_spot"), self.spot.get(coin, 0.0))
        self.spot[coin] = self.spot.get(coin, 0.0) - q
        return OrderResult(True, q, self.prices[coin])
