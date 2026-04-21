import math
from backtest.models import Trade


class Reporter:
    @staticmethod
    def generate(result: dict) -> dict:
        trades: list[Trade] = result["trades"]
        equity_curve: list[tuple[int, float]] = result["equity_curve"]
        initial = result["initial_balance"]
        final = result["final_equity"]

        net_return = (final - initial) / initial if initial else 0.0
        total_commission = sum(t.commission for t in trades)
        total_trades = len(trades)

        closing_trades = [t for t in trades if t.pnl != 0.0]
        wins = [t for t in closing_trades if t.pnl > 0]
        losses = [t for t in closing_trades if t.pnl < 0]
        win_rate = len(wins) / len(closing_trades) if closing_trades else 0.0
        total_profit = sum(t.pnl for t in wins)
        total_loss = abs(sum(t.pnl for t in losses))
        profit_factor = total_profit / total_loss if total_loss > 0 else float("inf")

        long_trades = sum(1 for t in trades if t.side == "buy")
        short_trades = sum(1 for t in trades if t.side == "sell")

        max_dd, max_dd_duration = Reporter._calc_drawdown(equity_curve)
        returns = Reporter._calc_returns(equity_curve)
        sharpe = Reporter._calc_sharpe(returns)
        sortino = Reporter._calc_sortino(returns)

        if len(equity_curve) >= 2:
            days = (equity_curve[-1][0] - equity_curve[0][0]) / (1000 * 86400)
            if days > 0 and (1 + net_return) > 0:
                annual_return = (1 + net_return) ** (365 / max(days, 1)) - 1
            else:
                annual_return = -1.0 if net_return <= -1 else 0.0
        else:
            annual_return = 0.0

        avg_hold_time = Reporter._calc_avg_hold_time(trades)

        return {
            "net_return": net_return, "annual_return": annual_return,
            "max_drawdown": max_dd, "max_dd_duration": max_dd_duration,
            "sharpe_ratio": sharpe, "sortino_ratio": sortino,
            "win_rate": win_rate, "profit_factor": profit_factor,
            "total_trades": total_trades, "long_trades": long_trades,
            "short_trades": short_trades, "avg_hold_time": avg_hold_time,
            "total_commission": total_commission, "total_funding": 0.0,
            "equity_curve": equity_curve,
            "trades": [
                {"id": t.id, "order_id": t.order_id, "symbol": t.symbol,
                 "side": t.side, "price": t.price, "quantity": t.quantity,
                 "pnl": t.pnl, "commission": t.commission, "timestamp": t.timestamp}
                for t in trades
            ],
        }

    @staticmethod
    def _calc_drawdown(curve: list[tuple[int, float]]) -> tuple[float, int]:
        if not curve:
            return 0.0, 0
        peak = curve[0][1]
        max_dd = 0.0
        dd_start = curve[0][0]
        max_dd_dur = 0
        for ts, eq in curve:
            if eq >= peak:
                peak = eq
                dd_start = ts
            dd = (peak - eq) / peak if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd
                max_dd_dur = ts - dd_start
        return max_dd, max_dd_dur

    @staticmethod
    def _calc_returns(curve: list[tuple[int, float]]) -> list[float]:
        if len(curve) < 2:
            return []
        return [(curve[i][1] - curve[i-1][1]) / curve[i-1][1]
                for i in range(1, len(curve)) if curve[i-1][1] > 0]

    @staticmethod
    def _calc_sharpe(returns: list[float], risk_free: float = 0.0) -> float:
        if len(returns) < 2:
            return 0.0
        mean = sum(returns) / len(returns)
        var = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
        std = math.sqrt(var)
        if std == 0:
            return 0.0
        return (mean - risk_free) * math.sqrt(365 * 24) / std

    @staticmethod
    def _calc_sortino(returns: list[float], risk_free: float = 0.0) -> float:
        if len(returns) < 2:
            return 0.0
        mean = sum(returns) / len(returns)
        downside = [r for r in returns if r < 0]
        if not downside:
            return float("inf") if mean > 0 else 0.0
        down_var = sum(r ** 2 for r in downside) / len(downside)
        down_std = math.sqrt(down_var)
        if down_std == 0:
            return 0.0
        return (mean - risk_free) * math.sqrt(365 * 24) / down_std

    @staticmethod
    def _calc_avg_hold_time(trades: list[Trade]) -> int:
        if len(trades) < 2:
            return 0
        pairs = []
        for i in range(0, len(trades) - 1, 2):
            pairs.append(trades[i + 1].timestamp - trades[i].timestamp)
        return int(sum(pairs) / len(pairs)) if pairs else 0
