from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone

from backtest.exchange import SimExchange
from backtest.models import Bar, Position, Trade
from backtest.multi_data_feed import MultiSymbolDataFeed


class PortfolioBacktestEngine:
    """
    Runs the consecutive-reverse strategy simultaneously across N symbols
    with a shared SimExchange (cross-margin style).

    Position sizing uses wallet balance = cash + sum(locked margins),
    matching BET live behaviour where sizing is decoupled from unrealised PnL.
    """

    def __init__(
        self,
        db_path: str,
        symbols: list[str],
        interval: str,
        exchange: str,
        balance: float = 10000.0,
        leverage: int = 10,
        commission_rate: float = 0.0004,
        funding_rate: float = 0.0001,
        maintenance_margin: float = 0.005,
        consecutive_threshold: int = 7,
        position_multiplier: float = 2.9,
        initial_position_pct: float = 0.05,
        profit_candle_threshold: int = 3,
        max_position_symbols: int = 10,
        margin_ratio_limit: float = 0.8,
        start: str | None = None,
        end: str | None = None,
    ):
        self.db_path = db_path
        self.symbols = symbols
        self.interval = interval
        self.exchange_name = exchange
        self.balance = balance
        self.leverage = leverage
        self.commission_rate = commission_rate
        self.funding_rate = funding_rate
        self.maintenance_margin = maintenance_margin
        self.consecutive_threshold = consecutive_threshold
        self.position_multiplier = position_multiplier
        self.initial_position_pct = initial_position_pct
        self.profit_candle_threshold = profit_candle_threshold
        self.max_position_symbols = max_position_symbols
        self.margin_ratio_limit = margin_ratio_limit
        self.start_ts = self._parse_time(start) if start else None
        self.end_ts = self._parse_time(end) if end else None

    @staticmethod
    def _parse_time(s: str) -> int:
        dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)

    def run(self) -> dict:
        sim = SimExchange(
            balance=self.balance,
            leverage=self.leverage,
            commission_rate=self.commission_rate,
            funding_rate=self.funding_rate,
            maintenance_margin=self.maintenance_margin,
            margin_mode="cross",
        )

        states: dict[str, dict] = {
            sym: {"consecutive_count": 0, "streak_direction": 0, "profit_candle_count": 0}
            for sym in self.symbols
        }

        feed = MultiSymbolDataFeed(
            db_path=self.db_path,
            symbols=self.symbols,
            interval=self.interval,
            exchange=self.exchange_name,
            start_ts=self.start_ts,
            end_ts=self.end_ts,
        )

        for _ts, bars in feed:
            for bar in bars.values():
                sim.on_new_bar(bar)
            newly_opened: set[str] = set()
            for sym, bar in bars.items():
                self._process_symbol(sym, bar, states[sym], sim, newly_opened)

        trades = sim.get_trades()

        # Deduplicate equity curve: on_new_bar records one entry per symbol per timestamp;
        # keep the last value per timestamp so Reporter sees one point per bar.
        seen: dict[int, float] = {}
        for ts, eq in sim.get_equity_curve():
            seen[ts] = eq
        equity_curve = sorted(seen.items())

        return {
            "trades": trades,
            "trades_count": len(trades),
            "equity_curve": equity_curve,
            "final_equity": sim.equity,
            "initial_balance": self.balance,
            "portfolio_symbols": self.symbols,
            "per_symbol_stats": self._per_symbol_stats(trades),
        }

    # ── Strategy logic ───────────────────────────────────────────

    def _process_symbol(
        self,
        sym: str,
        bar: Bar,
        state: dict,
        sim: SimExchange,
        newly_opened: set[str],
    ) -> None:
        direction = self._bar_direction(bar)
        if direction == 0:
            return

        self._update_streak(state, direction)

        pos: Position | None = sim._positions.get(sym)
        can_trigger = state["consecutive_count"] >= self.consecutive_threshold

        if pos is None:
            state["profit_candle_count"] = 0
            if can_trigger and self._account_allows_new_slot(sym, sim, newly_opened):
                self._submit_open(sym, state, direction, sim)
                newly_opened.add(sym)

        elif self._is_profit_candle(pos, direction):
            state["profit_candle_count"] += 1
            if state["profit_candle_count"] >= self.profit_candle_threshold:
                self._submit_close(sym, pos, sim)
                state["profit_candle_count"] = 0
                if can_trigger:
                    # Reopening same symbol — not a new slot
                    self._submit_open(sym, state, direction, sim)

        else:
            self._submit_close(sym, pos, sim)
            state["profit_candle_count"] = 0
            if can_trigger:
                # Reopening same symbol — not a new slot
                self._submit_open(sym, state, direction, sim)

    def _update_streak(self, state: dict, direction: int) -> None:
        if direction == state["streak_direction"]:
            state["consecutive_count"] += 1
        else:
            state["consecutive_count"] = 1
            state["streak_direction"] = direction

    @staticmethod
    def _bar_direction(bar: Bar) -> int:
        if bar.close > bar.open:
            return 1
        elif bar.close < bar.open:
            return -1
        return 0

    @staticmethod
    def _is_profit_candle(pos: Position, direction: int) -> bool:
        return (pos.side == "long" and direction == 1) or \
               (pos.side == "short" and direction == -1)

    # ── Account risk controls ────────────────────────────────────

    def _account_allows_new_slot(
        self, sym: str, sim: SimExchange, newly_opened: set[str]
    ) -> bool:
        # Count actual positions + any symbols opened earlier this same bar
        occupied = set(sim._positions.keys()) | newly_opened
        if len(occupied) >= self.max_position_symbols:
            return False
        equity = sim.equity
        if equity > 0:
            total_margin = sum(p.margin for p in sim._positions.values())
            if total_margin / equity > self.margin_ratio_limit:
                return False
        return True

    # ── Order helpers ────────────────────────────────────────────

    def _calc_quantity(self, state: dict, sim: SimExchange) -> float:
        wallet = sim.balance + sum(p.margin for p in sim._positions.values())
        n = state["consecutive_count"] - self.consecutive_threshold + 1
        multiplier = self.position_multiplier ** (n - 1)
        return wallet * self.initial_position_pct * multiplier

    def _submit_open(
        self, sym: str, state: dict, direction: int, sim: SimExchange
    ) -> None:
        qty = self._calc_quantity(state, sim)
        if qty <= 0:
            return
        side = "sell" if direction == 1 else "buy"  # reverse: up-streak → short
        sim.submit_order(sym, side, "market", qty)

    @staticmethod
    def _submit_close(sym: str, pos: Position, sim: SimExchange) -> None:
        side = "sell" if pos.side == "long" else "buy"
        sim.submit_order(sym, side, "market", pos.quantity)

    # ── Per-symbol statistics ────────────────────────────────────

    def _per_symbol_stats(self, trades: list[Trade]) -> dict:
        gross_pnl: dict[str, float] = defaultdict(float)
        commissions: dict[str, float] = defaultdict(float)
        closes: dict[str, list[float]] = defaultdict(list)

        for t in trades:
            commissions[t.symbol] += t.commission
            if t.pnl != 0.0:
                gross_pnl[t.symbol] += t.pnl
                closes[t.symbol].append(t.pnl)

        stats = {}
        for sym in self.symbols:
            pnls = closes[sym]
            total = len(pnls)
            wins = sum(1 for p in pnls if p > 0)
            stats[sym] = {
                "total_trades": total,
                "win_rate": wins / total if total > 0 else 0.0,
                "net_pnl": gross_pnl[sym] - commissions[sym],
            }
        return stats
