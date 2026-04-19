import uuid
from backtest.models import Bar, Order, Position, Trade

_FUNDING_HOURS = {0, 8, 16}


class SimExchange:
    def __init__(
        self,
        balance: float,
        leverage: int,
        commission_rate: float,
        funding_rate: float,
        maintenance_margin: float,
    ):
        self.initial_balance = balance
        self.balance = balance
        self.leverage = leverage
        self.commission_rate = commission_rate
        self.funding_rate = funding_rate
        self.maintenance_margin = maintenance_margin

        self._positions: dict[str, Position] = {}
        self._pending_orders: list[Order] = []
        self._trades: list[Trade] = []
        self._equity_curve: list[tuple[int, float]] = []

    def submit_order(
        self, symbol: str, side: str, type_: str, quantity: float, price: float | None = None
    ) -> Order:
        order = Order(
            id=uuid.uuid4().hex[:8],
            symbol=symbol,
            side=side,
            type=type_,
            quantity=quantity,
            price=price,
        )
        self._pending_orders.append(order)
        return order

    def on_new_bar(self, bar: Bar) -> None:
        self._settle_funding(bar)
        self._match_orders(bar)
        self._update_unrealized_pnl(bar)
        self._check_liquidation(bar)
        self._record_equity(bar)

    def _settle_funding(self, bar: Bar) -> None:
        from datetime import datetime, timezone
        dt = datetime.fromtimestamp(bar.timestamp / 1000, tz=timezone.utc)
        if dt.hour not in _FUNDING_HOURS or dt.minute != 0:
            return
        for pos in self._positions.values():
            payment = pos.quantity * self.funding_rate
            if pos.side == "long":
                self.balance -= payment
            else:
                self.balance += payment

    def _match_orders(self, bar: Bar) -> None:
        still_pending = []
        for order in self._pending_orders:
            if order.symbol != bar.symbol:
                still_pending.append(order)
                continue
            filled_price = self._try_fill(order, bar)
            if filled_price is not None:
                self._execute_fill(order, filled_price, bar)
            else:
                still_pending.append(order)
        self._pending_orders = still_pending

    def _try_fill(self, order: Order, bar: Bar) -> float | None:
        if order.type == "market":
            return bar.open
        if order.type == "limit":
            if order.side == "buy" and bar.low <= order.price:
                return order.price
            if order.side == "sell" and bar.high >= order.price:
                return order.price
        return None

    def _execute_fill(self, order: Order, price: float, bar: Bar) -> None:
        order.status = "filled"
        order.filled_price = price
        order.filled_at = bar.timestamp

        commission = order.quantity * self.commission_rate
        order.commission = commission
        self.balance -= commission

        pos = self._positions.get(order.symbol)
        pnl = 0.0

        if pos is None:
            side = "long" if order.side == "buy" else "short"
            margin = order.quantity / self.leverage
            self.balance -= margin
            self._positions[order.symbol] = Position(
                symbol=order.symbol,
                side=side,
                quantity=order.quantity,
                entry_price=price,
                leverage=self.leverage,
                margin=margin,
            )
        elif (pos.side == "long" and order.side == "sell") or (
            pos.side == "short" and order.side == "buy"
        ):
            close_qty = min(order.quantity, pos.quantity)
            if pos.side == "long":
                pnl = close_qty * (price - pos.entry_price) / pos.entry_price
            else:
                pnl = close_qty * (pos.entry_price - price) / pos.entry_price
            self.balance += pnl
            margin_returned = pos.margin * (close_qty / pos.quantity)
            self.balance += margin_returned
            pos.margin -= margin_returned

            remaining = pos.quantity - close_qty
            if remaining <= 1e-8:
                del self._positions[order.symbol]
            else:
                pos.quantity = remaining

            leftover = order.quantity - close_qty
            if leftover > 1e-8:
                side = "long" if order.side == "buy" else "short"
                margin = leftover / self.leverage
                self.balance -= margin
                self._positions[order.symbol] = Position(
                    symbol=order.symbol,
                    side=side,
                    quantity=leftover,
                    entry_price=price,
                    leverage=self.leverage,
                    margin=margin,
                )
        else:
            total_qty = pos.quantity + order.quantity
            pos.entry_price = (
                pos.entry_price * pos.quantity + price * order.quantity
            ) / total_qty
            pos.quantity = total_qty
            additional_margin = order.quantity / self.leverage
            self.balance -= additional_margin
            pos.margin += additional_margin

        trade = Trade(
            id=uuid.uuid4().hex[:8],
            order_id=order.id,
            symbol=order.symbol,
            side=order.side,
            price=price,
            quantity=order.quantity,
            pnl=pnl,
            commission=commission,
            timestamp=bar.timestamp,
        )
        self._trades.append(trade)

    def _update_unrealized_pnl(self, bar: Bar) -> None:
        pos = self._positions.get(bar.symbol)
        if pos is None:
            return
        if pos.side == "long":
            pos.unrealized_pnl = pos.quantity * (bar.close - pos.entry_price) / pos.entry_price
        else:
            pos.unrealized_pnl = pos.quantity * (pos.entry_price - bar.close) / pos.entry_price

    def _check_liquidation(self, bar: Bar) -> None:
        pos = self._positions.get(bar.symbol)
        if pos is None:
            return
        equity_in_position = pos.margin + pos.unrealized_pnl
        if equity_in_position <= 0 or (pos.margin / equity_in_position) >= (1 / self.maintenance_margin):
            self._trades.append(Trade(
                id=uuid.uuid4().hex[:8],
                order_id="liquidation",
                symbol=bar.symbol,
                side="sell" if pos.side == "long" else "buy",
                price=bar.close,
                quantity=pos.quantity,
                pnl=-pos.margin,
                commission=0.0,
                timestamp=bar.timestamp,
            ))
            self.balance -= min(pos.margin, self.balance)
            del self._positions[bar.symbol]

    def _record_equity(self, bar: Bar) -> None:
        equity = self.balance
        for pos in self._positions.values():
            equity += pos.margin + pos.unrealized_pnl
        self._equity_curve.append((bar.timestamp, equity))

    def get_position(self, symbol: str) -> Position | None:
        return self._positions.get(symbol)

    def get_trades(self) -> list[Trade]:
        return list(self._trades)

    def get_equity_curve(self) -> list[tuple[int, float]]:
        return list(self._equity_curve)

    @property
    def equity(self) -> float:
        total = self.balance
        for pos in self._positions.values():
            total += pos.margin + pos.unrealized_pnl
        return total
