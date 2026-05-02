import math
import time
import uuid
from typing import Callable, TypeVar

from backtest.models import Order, Position

try:
    from binance.um_futures import UMFutures
except ImportError:
    UMFutures = None

_T = TypeVar("_T")


def _retry(fn: Callable[[], _T], attempts: int = 3, backoff: float = 2.0) -> _T:
    last_exc = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if i < attempts - 1:
                time.sleep(backoff * (2 ** i))
    raise last_exc


class LiveExchange:
    def __init__(self, client, symbol: str, leverage: int,
                 commission_rate: float, dry_run: bool = False):
        self._client = client
        self._symbol = symbol
        self._leverage = leverage
        self._commission_rate = commission_rate
        self._dry_run = dry_run
        self._balance: float = 0.0
        self._position: Position | None = None
        self._current_price: float = 0.0
        self._lot_step: float = 0.001
        self._pending_order_ids: list[str] = []
        self._fetch_lot_step()

    def _fetch_lot_step(self) -> None:
        info = _retry(lambda: self._client.exchange_info())
        for s in info["symbols"]:
            if s["symbol"] == self._symbol:
                for f in s["filters"]:
                    if f["filterType"] == "LOT_SIZE":
                        self._lot_step = float(f["stepSize"])
                        return
        raise ValueError(f"Symbol {self._symbol!r} not found in exchange_info")

    def _round_qty(self, qty: float) -> float:
        if self._lot_step <= 0:
            return qty
        precision = max(0, -int(round(math.log10(self._lot_step))))
        return round(qty, precision)

    def sync(self) -> None:
        balances = _retry(lambda: self._client.balance())
        for b in balances:
            if b["asset"] == "USDT":
                self._balance = float(b["availableBalance"])
                break

        positions = _retry(lambda: self._client.get_position_risk(symbol=self._symbol))
        self._position = None
        for p in positions:
            qty = float(p["positionAmt"])
            if abs(qty) < 1e-8:
                continue
            entry_price = float(p["entryPrice"])
            unrealized_pnl = float(p["unrealizedProfit"])
            notional = abs(qty) * entry_price
            margin = notional / self._leverage
            self._position = Position(
                symbol=self._symbol,
                side="long" if qty > 0 else "short",
                quantity=notional,
                entry_price=entry_price,
                leverage=self._leverage,
                unrealized_pnl=unrealized_pnl,
                margin=margin,
            )
            break

        premium = _retry(lambda: self._client.mark_price(symbol=self._symbol))
        self._current_price = float(premium["markPrice"])

    def get_position(self, symbol: str) -> Position | None:
        if self._position and self._position.symbol == symbol:
            return self._position
        return None

    @property
    def balance(self) -> float:
        return self._balance

    @property
    def equity(self) -> float:
        if self._position is None:
            return self._balance
        return self._balance + self._position.margin + self._position.unrealized_pnl

    def submit_order(self, symbol: str, side: str, type_: str,
                     quantity: float, price: float | None = None) -> Order:
        order_id = uuid.uuid4().hex[:8]

        if self._dry_run:
            print(f"[dry-run] {side.upper()} {type_} {quantity:.2f} USDT @ {price or 'market'}")
            return Order(
                id=order_id, symbol=symbol, side=side, type=type_,
                quantity=quantity, price=price, status="filled",
                filled_price=self._current_price, filled_at=int(time.time() * 1000),
                commission=quantity * self._commission_rate,
            )

        contract_qty = self._round_qty(quantity / self._current_price)
        if contract_qty <= 0:
            return Order(id=order_id, symbol=symbol, side=side, type=type_,
                         quantity=quantity, price=price, status="canceled")

        params: dict = {
            "symbol": symbol,
            "side": side.upper(),
            "type": "MARKET" if type_ == "market" else "LIMIT",
            "quantity": contract_qty,
        }
        if type_ == "limit" and price is not None:
            params["price"] = price
            params["timeInForce"] = "GTC"

        resp = _retry(lambda: self._client.new_order(**params))
        binance_id = str(resp["orderId"])
        self._pending_order_ids.append(binance_id)
        return Order(
            id=binance_id, symbol=symbol, side=side, type=type_,
            quantity=quantity, price=price, status="pending",
        )

    def wait_fills(self, timeout: float = 30.0) -> None:
        if not self._pending_order_ids:
            return
        deadline = time.time() + timeout
        remaining = list(self._pending_order_ids)
        while remaining and time.time() < deadline:
            still_pending = []
            for oid in remaining:
                resp = _retry(lambda oid=oid: self._client.query_order(
                    symbol=self._symbol, orderId=int(oid)
                ))
                if resp["status"] not in ("FILLED", "CANCELED", "EXPIRED", "REJECTED"):
                    still_pending.append(oid)
            remaining = still_pending
            if remaining:
                time.sleep(0.5)
        if remaining:
            print(f"[WARN] orders not confirmed within {timeout}s: {remaining}")
        self._pending_order_ids.clear()
