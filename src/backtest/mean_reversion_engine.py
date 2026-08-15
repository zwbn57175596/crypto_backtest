from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone

from backtest.exchange import SimExchange
from backtest.mean_reversion import (
    MRParams, bar_direction, bracket_prices, entry_price, fade_side,
    position_notional, update_streak,
)
from backtest.models import Bar, Trade
from backtest.multi_data_feed import MultiSymbolDataFeed

_INTERVAL_MS: dict[str, int] = {
    "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
    "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000, "6h": 21_600_000,
    "8h": 28_800_000, "12h": 43_200_000, "1d": 86_400_000,
}


class MeanReversionPortfolioEngine:
    """
    多币均值回归组合引擎：连续反向信号 + maker 限价入场 +
    TP/SL/时间止损 bracket 出场 + 固定风险仓位（无马丁）。

    与 PortfolioBacktestEngine 并列，不共享代码（后者是马丁复现用）。
    """

    def __init__(
        self,
        db_path: str,
        symbols: list[str],
        interval: str,
        exchange: str,
        params: MRParams,
        balance: float = 10000.0,
        leverage: int = 10,
        commission_rate: float = 0.0005,
        funding_rate: float = 0.0001,
        maintenance_margin: float = 0.005,
        maker_commission_rate: float = 0.0002,
        taker_commission_rate: float = 0.0005,
        max_position_symbols: int = 10,
        margin_ratio_limit: float = 0.8,
        start: str | None = None,
        end: str | None = None,
    ):
        self.db_path = db_path
        self.symbols = symbols
        self.interval = interval
        self.exchange_name = exchange
        self.params = params
        self.balance = balance
        self.leverage = leverage
        self.commission_rate = commission_rate
        self.funding_rate = funding_rate
        self.maintenance_margin = maintenance_margin
        self.maker_commission_rate = maker_commission_rate
        self.taker_commission_rate = taker_commission_rate
        self.max_position_symbols = max_position_symbols
        self.margin_ratio_limit = margin_ratio_limit
        self.start_ts = self._parse_time(start) if start else None
        self.end_ts = self._parse_time(end) if end else None
        if interval not in _INTERVAL_MS:
            raise ValueError(f"Unsupported interval: {interval!r}")
        self._interval_ms = _INTERVAL_MS[interval]
        self._entries_submitted = 0

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
            maker_commission_rate=self.maker_commission_rate,
            taker_commission_rate=self.taker_commission_rate,
        )

        states: dict[str, dict] = {
            sym: {"count": 0, "streak_dir": 0, "pending_expire_ts": 0}
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
            newly_submitted: set[str] = set()
            for sym, bar in bars.items():
                self._process_symbol(sym, bar, states[sym], sim, newly_submitted)

        trades = sim.get_trades()

        # 去重净值曲线：on_new_bar 每 symbol 记一条，保留同 ts 最后一条
        seen: dict[int, float] = {}
        for ts, eq in sim.get_equity_curve():
            seen[ts] = eq
        equity_curve = sorted(seen.items())

        entries_filled = sum(
            1 for t in trades
            if not t.order_id.startswith("bracket") and t.order_id != "liquidation" and t.pnl == 0.0
        )
        miss_rate = (
            1.0 - entries_filled / self._entries_submitted
            if self._entries_submitted > 0 else 0.0
        )

        return {
            "trades": trades,
            "trades_count": len(trades),
            "equity_curve": equity_curve,
            "final_equity": sim.equity,
            "initial_balance": self.balance,
            "portfolio_symbols": self.symbols,
            "per_symbol_stats": self._per_symbol_stats(trades),
            "entries_submitted": self._entries_submitted,
            "entries_filled": entries_filled,
            "miss_rate": miss_rate,
        }

    # ── 单币处理 ────────────────────────────────────────────────
    def _process_symbol(
        self, sym: str, bar: Bar, state: dict, sim: SimExchange, newly_submitted: set[str]
    ) -> None:
        direction = bar_direction(bar.open, bar.close)
        state["count"], state["streak_dir"] = update_streak(
            state["count"], state["streak_dir"], direction
        )

        pos = sim._positions.get(sym)

        # 1) 持仓存在但尚无 bracket（本 bar 刚成交）→ 补挂 bracket
        if pos is not None:
            if sym not in sim._brackets:
                tp, sl = bracket_prices(
                    pos.entry_price, pos.side, self.params.tp_pct, self.params.sl_pct
                )
                expire_ts = bar.timestamp + self.params.time_stop_bars * self._interval_ms
                sim.set_bracket(sym, tp, sl, expire_ts)
            return  # 一币一仓，不叠加

        # 2) 空仓但有仍在有效期内的入场挂单 → 等待
        if state["pending_expire_ts"] and bar.timestamp < state["pending_expire_ts"]:
            return
        state["pending_expire_ts"] = 0  # 挂单已成交或已被交易所撤销

        # 3) 新信号？
        if direction == 0 or state["count"] < self.params.threshold:
            return
        if not self._account_allows_new_slot(sim, newly_submitted):
            return

        # 4) 反手侧挂 maker 限价
        side = fade_side(state["streak_dir"])
        px = entry_price(bar.close, state["streak_dir"], self.params.entry_offset_pct)
        notional = position_notional(sim.equity, self.params.risk_pct, self.params.sl_pct)
        if notional <= 0:
            return
        expire_ts = bar.timestamp + self.params.entry_ttl_bars * self._interval_ms
        sim.submit_order(sym, side, "limit", notional, price=px, expire_ts=expire_ts)
        state["pending_expire_ts"] = expire_ts
        newly_submitted.add(sym)
        self._entries_submitted += 1

    def _account_allows_new_slot(self, sim: SimExchange, newly_submitted: set[str]) -> bool:
        occupied = (
            set(sim._positions.keys())
            | {o.symbol for o in sim._pending_orders}
            | newly_submitted
        )
        if len(occupied) >= self.max_position_symbols:
            return False
        equity = sim.equity
        if equity > 0:
            total_margin = sum(p.margin for p in sim._positions.values())
            if total_margin / equity > self.margin_ratio_limit:
                return False
        return True

    # ── 每币统计 ────────────────────────────────────────────────
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
