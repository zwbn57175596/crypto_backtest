# src/backtest/live_exchange.py
import math
import time
import uuid

from backtest.live_connector import BaseExchangeConnector
from backtest.live_history import LiveHistoryDB
from backtest.models import Order, Position


class LiveExchange:
    def __init__(self, connector: BaseExchangeConnector, history_db: LiveHistoryDB,
                 account_id: str, symbol: str, leverage: int,
                 commission_rate: float, dry_run: bool = False):
        self._connector = connector
        self._history_db = history_db
        self._account_id = account_id
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
        info = self._connector.exchange_info(self._symbol)
        for f in info.get("filters", []):
            if f["filterType"] == "LOT_SIZE":
                self._lot_step = float(f["stepSize"])
                return
        raise ValueError(f"LOT_SIZE filter not found for {self._symbol!r}")

    def _round_qty(self, qty: float) -> float:
        if self._lot_step <= 0:
            return qty
        precision = max(0, -int(round(math.log10(self._lot_step))))
        return round(qty, precision)

    def sync(self) -> None:
        if not self._dry_run:
            self._balance = self._connector.fetch_balance()
            pos_data = self._connector.fetch_position(self._symbol)
            if pos_data is None:
                self._position = None
            else:
                notional = pos_data["qty"] * pos_data["entry_price"]
                margin = notional / self._leverage
                self._position = Position(
                    symbol=self._symbol,
                    side=pos_data["side"],
                    quantity=notional,
                    entry_price=pos_data["entry_price"],
                    leverage=self._leverage,
                    unrealized_pnl=pos_data["unrealized_pnl"],
                    margin=margin,
                )
        self._current_price = self._connector.fetch_mark_price(self._symbol)
        self._history_db.record_position(
            self._account_id, self._connector.exchange_name,
            self._symbol, self._position, self._balance, self.equity,
            int(time.time() * 1000),
        )

    def get_position(self, symbol: str) -> Position | None:
        if self._position and self._position.symbol == symbol:
            return self._position
        return None

    @property
    def leverage(self) -> int:
        return self._leverage

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
        ts = int(time.time() * 1000)

        if self._dry_run:
            order_dict = {
                "order_id": order_id, "symbol": symbol, "side": side, "type": type_,
                "quantity": quantity, "price": price, "status": "filled",
                "filled_price": self._current_price,
                "filled_qty": quantity / self._current_price if self._current_price else 0,
                "commission": quantity * self._commission_rate,
                "ts": ts, "filled_at": ts,
            }
            self._history_db.upsert_order(self._account_id, self._connector.exchange_name, order_dict)
            print(f"[dry-run] {side.upper()} {type_} {quantity:.2f} USDT @ {price or 'market'}")
            return Order(
                id=order_id, symbol=symbol, side=side, type=type_,
                quantity=quantity, price=price, status="filled",
                filled_price=self._current_price, filled_at=ts,
                commission=quantity * self._commission_rate,
            )

        contract_qty = self._round_qty(quantity / self._current_price)
        if contract_qty <= 0:
            return Order(id=order_id, symbol=symbol, side=side, type=type_,
                         quantity=quantity, price=price, status="canceled")

        resp = self._connector.submit_order(symbol, side, type_, contract_qty, price)
        exchange_id = str(resp["orderId"])

        order_dict = {
            "order_id": exchange_id, "symbol": symbol, "side": side, "type": type_,
            "quantity": quantity, "price": price, "status": "pending",
            "filled_price": None, "filled_qty": None, "commission": None,
            "ts": ts, "filled_at": None,
        }
        self._history_db.upsert_order(self._account_id, self._connector.exchange_name, order_dict)
        self._pending_order_ids.append(exchange_id)
        return Order(id=exchange_id, symbol=symbol, side=side, type=type_,
                     quantity=quantity, price=price, status="pending")

    def wait_fills(self, timeout: float = 30.0) -> None:
        if not self._pending_order_ids:
            return
        deadline = time.time() + timeout
        remaining = list(self._pending_order_ids)
        while remaining and time.time() < deadline:
            still_pending = []
            for oid in remaining:
                order_dict = self._connector.query_order(self._symbol, oid)
                if order_dict["status"] not in ("filled", "canceled", "expired", "rejected"):
                    still_pending.append(oid)
                else:
                    self._history_db.upsert_order(
                        self._account_id, self._connector.exchange_name, order_dict
                    )
            remaining = still_pending
            if remaining:
                time.sleep(0.5)
        if remaining:
            print(f"[WARN] orders not confirmed within {timeout}s: {remaining}")
        self._pending_order_ids.clear()
