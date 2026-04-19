from __future__ import annotations
from typing import TYPE_CHECKING
from backtest.models import Bar, Position

if TYPE_CHECKING:
    from backtest.exchange import SimExchange


class BaseStrategy:
    def __init__(self, exchange: SimExchange, symbol: str):
        self._exchange = exchange
        self._symbol = symbol
        self._bar_history: list[Bar] = []

    def _push_bar(self, bar: Bar) -> None:
        self._bar_history.append(bar)
        self.on_bar(bar)

    def on_init(self) -> None:
        pass

    def on_bar(self, bar: Bar) -> None:
        pass

    def buy(self, quantity: float, price: float | None = None) -> None:
        type_ = "limit" if price is not None else "market"
        self._exchange.submit_order(self._symbol, "buy", type_, quantity, price)

    def sell(self, quantity: float, price: float | None = None) -> None:
        type_ = "limit" if price is not None else "market"
        self._exchange.submit_order(self._symbol, "sell", type_, quantity, price)

    def close(self) -> None:
        pos = self._exchange.get_position(self._symbol)
        if pos is None:
            return
        side = "sell" if pos.side == "long" else "buy"
        self._exchange.submit_order(self._symbol, side, "market", pos.quantity)

    @property
    def position(self) -> Position | None:
        return self._exchange.get_position(self._symbol)

    @property
    def balance(self) -> float:
        return self._exchange.balance

    @property
    def equity(self) -> float:
        return self._exchange.equity

    def history(self, n: int) -> list[Bar]:
        return self._bar_history[-n:]
