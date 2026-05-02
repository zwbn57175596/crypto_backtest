"""
Numba JIT-compiled backtest simulation for ConsecutiveReverse strategy.

This module provides a pure-function implementation of the strategy + exchange
logic that can be JIT-compiled by Numba for 50-200x speedup over the Python
event-driven engine.

Usage:
    from backtest.numba_simulate import simulate, load_bars

    bars = load_bars(db_path, symbol, interval, exchange, start_ts, end_ts)
    metrics = simulate(bars, threshold=5, multiplier=1.1, initial_pct=0.01,
                       profit_threshold=1, leverage=50, commission_rate=0.0004,
                       funding_rate=0.0001, maintenance_margin=0.5,
                       initial_balance=1000.0)
"""

import math
import sqlite3

import numpy as np
from numba import njit


def load_bars(
    db_path: str,
    symbol: str,
    interval: str,
    exchange: str,
    start_ts: int,
    end_ts: int,
) -> np.ndarray:
    """Load klines from SQLite into a float64 ndarray of shape (N, 6).

    Columns: [timestamp_ms, open, high, low, close, volume]
    """
    conn = sqlite3.connect(db_path)
    query = (
        "SELECT timestamp, open, high, low, close, volume "
        "FROM klines WHERE symbol = ? AND interval = ? AND exchange = ? "
        "AND timestamp >= ? AND timestamp <= ? ORDER BY timestamp ASC"
    )
    cursor = conn.execute(query, (symbol, interval, exchange, start_ts, end_ts))
    rows = cursor.fetchall()
    conn.close()
    if not rows:
        return np.empty((0, 6), dtype=np.float64)
    return np.array(rows, dtype=np.float64)


# ---------------------------------------------------------------------------
# Numba-compiled simulation kernel
# ---------------------------------------------------------------------------

# Position side constants
_NO_POS = 0
_LONG = 1
_SHORT = 2

# Order side constants
_BUY = 1
_SELL = 2


@njit(cache=True)
def _fill_order(
    order_side, order_qty, fill_price,
    balance, pos_side, pos_qty, pos_entry, pos_margin,
    leverage, commission_rate,
    total_trades, wins, losses, total_profit, total_loss,
):
    """Process a single order fill. Returns updated state."""
    commission = order_qty * commission_rate
    balance -= commission

    pnl = 0.0

    if pos_side == _NO_POS:
        # Open new position
        margin = order_qty / leverage
        balance -= margin
        pos_side = _LONG if order_side == _BUY else _SHORT
        pos_qty = order_qty
        pos_entry = fill_price
        pos_margin = margin
        total_trades += 1
    elif (pos_side == _LONG and order_side == _SELL) or (
        pos_side == _SHORT and order_side == _BUY
    ):
        # Close position
        close_qty = min(order_qty, pos_qty)
        if pos_side == _LONG:
            pnl = close_qty * (fill_price - pos_entry) / pos_entry
        else:
            pnl = close_qty * (pos_entry - fill_price) / pos_entry
        balance += pnl
        margin_returned = pos_margin * (close_qty / pos_qty)
        balance += margin_returned

        total_trades += 1
        if pnl > 0:
            wins += 1
            total_profit += pnl
        elif pnl < 0:
            losses += 1
            total_loss += abs(pnl)

        remaining = pos_qty - close_qty
        leftover = order_qty - close_qty

        if remaining <= 1e-8:
            pos_side = _NO_POS
            pos_qty = 0.0
            pos_entry = 0.0
            pos_margin = 0.0
        else:
            pos_qty = remaining
            pos_margin -= margin_returned

        # If leftover, open new position in opposite direction
        if leftover > 1e-8:
            margin = leftover / leverage
            balance -= margin
            pos_side = _LONG if order_side == _BUY else _SHORT
            pos_qty = leftover
            pos_entry = fill_price
            pos_margin = margin
            total_trades += 1
    else:
        # Add to existing position (same side)
        total_qty = pos_qty + order_qty
        pos_entry = (pos_entry * pos_qty + fill_price * order_qty) / total_qty
        pos_qty = total_qty
        additional_margin = order_qty / leverage
        balance -= additional_margin
        pos_margin += additional_margin
        total_trades += 1

    return (balance, pos_side, pos_qty, pos_entry, pos_margin,
            total_trades, wins, losses, total_profit, total_loss)


@njit(cache=True)
def simulate_martingale(
    bars: np.ndarray,
    threshold: int,
    multiplier: float,
    initial_pct: float,
    profit_threshold: int,
    sizing_leverage: int,
    exchange_leverage: int,
    commission_rate: float,
    funding_rate: float,
    maintenance_margin: float,
    initial_balance: float,
) -> tuple:
    """Run ConsecutiveReverse Martingale strategy (add to position on loss candle).

    Parameters
    ----------
    bars : ndarray shape (N, 6) — [ts_ms, open, high, low, close, volume]
    threshold : CONSECUTIVE_THRESHOLD
    multiplier : POSITION_MULTIPLIER
    initial_pct : INITIAL_POSITION_PCT
    profit_threshold : PROFIT_CANDLE_THRESHOLD
    sizing_leverage : Strategy LEVERAGE (for position sizing in _calc_quantity)
    exchange_leverage : SimExchange leverage (for margin = qty / leverage)
    commission_rate : e.g. 0.0004
    funding_rate : e.g. 0.0001
    maintenance_margin : e.g. 0.5
    initial_balance : e.g. 1000.0

    Returns
    -------
    tuple of (net_return, annual_return, max_drawdown, sharpe_ratio,
              sortino_ratio, win_rate, profit_factor, total_trades)
    """
    n = bars.shape[0]
    if n == 0:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)

    # Exchange state
    balance = initial_balance
    pos_side = _NO_POS
    pos_qty = 0.0
    pos_entry = 0.0
    pos_margin = 0.0
    pos_unrealized_pnl = 0.0

    # Strategy state
    consecutive_count = 0
    streak_direction = 0  # +1 up, -1 down
    profit_candle_count = 0

    # Pending orders: up to 2 per bar (close + reopen)
    # Using arrays: pend_sides[0..1], pend_qtys[0..1]
    n_pending = 0
    pend_side_0 = 0
    pend_qty_0 = 0.0
    pend_side_1 = 0
    pend_qty_1 = 0.0

    # Metrics accumulators
    total_trades = 0
    wins = 0
    losses = 0
    total_profit = 0.0
    total_loss = 0.0

    # For Sharpe/Sortino: online returns
    prev_equity = initial_balance
    sum_ret = 0.0
    sum_ret_sq = 0.0
    sum_down_sq = 0.0
    n_returns = 0
    n_downside = 0

    # For max drawdown
    peak_equity = initial_balance
    max_dd = 0.0

    for i in range(n):
        ts = bars[i, 0]
        open_price = bars[i, 1]
        close = bars[i, 4]

        # === 1. Settle funding ===
        ts_sec = int(ts) // 1000
        hour = (ts_sec // 3600) % 24
        minute = (ts_sec % 3600) // 60
        if minute == 0 and (hour == 0 or hour == 8 or hour == 16):
            if pos_side != _NO_POS:
                payment = pos_qty * funding_rate
                if pos_side == _LONG:
                    balance -= payment
                else:
                    balance += payment

        # === 2. Match pending orders (all fill at this bar's open) ===
        if n_pending >= 1:
            (balance, pos_side, pos_qty, pos_entry, pos_margin,
             total_trades, wins, losses, total_profit, total_loss) = _fill_order(
                pend_side_0, pend_qty_0, open_price,
                balance, pos_side, pos_qty, pos_entry, pos_margin,
                exchange_leverage, commission_rate,
                total_trades, wins, losses, total_profit, total_loss)

        if n_pending >= 2:
            (balance, pos_side, pos_qty, pos_entry, pos_margin,
             total_trades, wins, losses, total_profit, total_loss) = _fill_order(
                pend_side_1, pend_qty_1, open_price,
                balance, pos_side, pos_qty, pos_entry, pos_margin,
                exchange_leverage, commission_rate,
                total_trades, wins, losses, total_profit, total_loss)

        # Clear pending
        n_pending = 0
        pend_side_0 = 0
        pend_qty_0 = 0.0
        pend_side_1 = 0
        pend_qty_1 = 0.0

        # === 3. Update unrealized PnL ===
        if pos_side == _LONG:
            pos_unrealized_pnl = pos_qty * (close - pos_entry) / pos_entry
        elif pos_side == _SHORT:
            pos_unrealized_pnl = pos_qty * (pos_entry - close) / pos_entry
        else:
            pos_unrealized_pnl = 0.0

        # === 4. Check liquidation ===
        if pos_side != _NO_POS:
            equity_in_pos = pos_margin + pos_unrealized_pnl
            if equity_in_pos <= 0 or (
                pos_margin / equity_in_pos >= 1.0 / maintenance_margin
            ):
                total_trades += 1
                losses += 1
                total_loss += pos_margin
                balance -= min(pos_margin, balance)
                pos_side = _NO_POS
                pos_qty = 0.0
                pos_entry = 0.0
                pos_margin = 0.0
                pos_unrealized_pnl = 0.0

        # === 5. Record equity & compute metrics ===
        equity = balance
        if pos_side != _NO_POS:
            equity += pos_margin + pos_unrealized_pnl

        if equity > peak_equity:
            peak_equity = equity
        if peak_equity > 0:
            dd = (peak_equity - equity) / peak_equity
            if dd > max_dd:
                max_dd = dd

        if prev_equity > 0 and equity > 0 and i > 0:
            ret = (equity - prev_equity) / prev_equity
            sum_ret += ret
            sum_ret_sq += ret * ret
            n_returns += 1
            if ret < 0:
                sum_down_sq += ret * ret
                n_downside += 1
        prev_equity = equity

        # === 6. Strategy logic ===
        if close > open_price:
            direction = 1
        elif close < open_price:
            direction = -1
        else:
            direction = 0

        if direction == 0:
            continue

        # Update streak
        if direction == streak_direction:
            consecutive_count += 1
        else:
            consecutive_count = 1
            streak_direction = direction

        # Strategy decision
        if pos_side == _NO_POS:
            profit_candle_count = 0  # reset stale count from external close (e.g. liquidation)
            # Try open
            qty = _calc_quantity(
                consecutive_count, threshold, balance, initial_pct, multiplier, sizing_leverage
            )
            if qty > 0:
                if direction == 1:
                    pend_side_0 = _SELL
                else:
                    pend_side_0 = _BUY
                pend_qty_0 = qty
                n_pending = 1
        else:
            # Check if profit candle
            is_profit = (pos_side == _LONG and direction == 1) or (
                pos_side == _SHORT and direction == -1
            )
            if is_profit:
                profit_candle_count += 1
                if profit_candle_count >= profit_threshold:
                    # Close order
                    if pos_side == _LONG:
                        pend_side_0 = _SELL
                    else:
                        pend_side_0 = _BUY
                    pend_qty_0 = pos_qty
                    n_pending = 1
                    profit_candle_count = 0

                    # Try reopen (after close fills, balance returns margin+pnl)
                    # Use current balance for sizing (matches original: strategy
                    # reads self.balance which hasn't changed yet since order is pending)
                    reopen_qty = _calc_quantity(
                        consecutive_count, threshold, balance, initial_pct, multiplier, sizing_leverage
                    )
                    if reopen_qty > 0:
                        if direction == 1:
                            pend_side_1 = _SELL
                        else:
                            pend_side_1 = _BUY
                        pend_qty_1 = reopen_qty
                        n_pending = 2
            else:
                # Loss candle - add to contrarian position up to current target size
                profit_candle_count = 0
                target_qty = _calc_quantity(
                    consecutive_count, threshold, balance, initial_pct, multiplier, sizing_leverage
                )
                add_qty = target_qty - pos_qty
                if add_qty > 0:
                    if pos_side == _LONG:
                        pend_side_0 = _BUY
                    else:
                        pend_side_0 = _SELL
                    pend_qty_0 = add_qty
                    n_pending = 1

    # === Compute final metrics ===
    final_equity = balance
    if pos_side != _NO_POS:
        final_equity += pos_margin + pos_unrealized_pnl

    net_return = (final_equity - initial_balance) / initial_balance if initial_balance > 0 else 0.0

    # Annual return
    annual_return = 0.0
    if n >= 2:
        days = (bars[n - 1, 0] - bars[0, 0]) / (1000.0 * 86400.0)
        if days > 0 and (1.0 + net_return) > 0:
            annual_return = (1.0 + net_return) ** (365.0 / days) - 1.0
        elif net_return <= -1.0:
            annual_return = -1.0

    # Sharpe ratio (annualized, hourly returns assumed)
    sharpe = 0.0
    if net_return <= -1.0:
        sharpe = -999.0
    elif n_returns >= 2:
        mean_ret = sum_ret / n_returns
        variance = (sum_ret_sq / n_returns) - (mean_ret * mean_ret)
        variance = variance * n_returns / (n_returns - 1)
        if variance > 0:
            std = math.sqrt(variance)
            sharpe = (mean_ret - 0.0) * math.sqrt(365.0 * 24.0) / std

    # Sortino ratio
    sortino = 0.0
    if net_return <= -1.0:
        sortino = -999.0
    elif n_returns >= 2:
        mean_ret = sum_ret / n_returns
        if n_downside > 0:
            down_std = math.sqrt(sum_down_sq / n_downside)
            if down_std > 0:
                sortino = (mean_ret - 0.0) * math.sqrt(365.0 * 24.0) / down_std
        elif mean_ret > 0:
            sortino = 1e10

    # Win rate
    closing_trades = wins + losses
    win_rate = wins / closing_trades if closing_trades > 0 else 0.0

    # Profit factor
    profit_factor = total_profit / total_loss if total_loss > 0 else 1e10

    return (
        net_return,
        annual_return,
        max_dd,
        sharpe,
        sortino,
        win_rate,
        profit_factor,
        total_trades,
    )


@njit(cache=True)
def simulate_close_reopen(
    bars: np.ndarray,
    threshold: int,
    multiplier: float,
    initial_pct: float,
    profit_threshold: int,
    sizing_leverage: int,
    exchange_leverage: int,
    commission_rate: float,
    funding_rate: float,
    maintenance_margin: float,
    initial_balance: float,
) -> tuple:
    """Run ConsecutiveReverse Close+Reopen strategy (close then reopen on loss candle).

    Identical to simulate_martingale except on loss candle:
    - Loss candle: Close position immediately + try to reopen (instead of adding to position)

    Parameters
    ----------
    Same as simulate_martingale

    Returns
    -------
    tuple of (net_return, annual_return, max_drawdown, sharpe_ratio,
              sortino_ratio, win_rate, profit_factor, total_trades)
    """
    n = bars.shape[0]
    if n == 0:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)

    # Exchange state
    balance = initial_balance
    pos_side = _NO_POS
    pos_qty = 0.0
    pos_entry = 0.0
    pos_margin = 0.0
    pos_unrealized_pnl = 0.0

    # Strategy state
    consecutive_count = 0
    streak_direction = 0  # +1 up, -1 down
    profit_candle_count = 0

    # Pending orders: up to 2 per bar (close + reopen)
    n_pending = 0
    pend_side_0 = 0
    pend_qty_0 = 0.0
    pend_side_1 = 0
    pend_qty_1 = 0.0

    # Metrics accumulators
    total_trades = 0
    wins = 0
    losses = 0
    total_profit = 0.0
    total_loss = 0.0

    # For Sharpe/Sortino: online returns
    prev_equity = initial_balance
    sum_ret = 0.0
    sum_ret_sq = 0.0
    sum_down_sq = 0.0
    n_returns = 0
    n_downside = 0

    # For max drawdown
    peak_equity = initial_balance
    max_dd = 0.0

    for i in range(n):
        ts = bars[i, 0]
        open_price = bars[i, 1]
        close = bars[i, 4]

        # === 1. Settle funding ===
        ts_sec = int(ts) // 1000
        hour = (ts_sec // 3600) % 24
        minute = (ts_sec % 3600) // 60
        if minute == 0 and (hour == 0 or hour == 8 or hour == 16):
            if pos_side != _NO_POS:
                payment = pos_qty * funding_rate
                if pos_side == _LONG:
                    balance -= payment
                else:
                    balance += payment

        # === 2. Match pending orders (all fill at this bar's open) ===
        if n_pending >= 1:
            (balance, pos_side, pos_qty, pos_entry, pos_margin,
             total_trades, wins, losses, total_profit, total_loss) = _fill_order(
                pend_side_0, pend_qty_0, open_price,
                balance, pos_side, pos_qty, pos_entry, pos_margin,
                exchange_leverage, commission_rate,
                total_trades, wins, losses, total_profit, total_loss)

        if n_pending >= 2:
            (balance, pos_side, pos_qty, pos_entry, pos_margin,
             total_trades, wins, losses, total_profit, total_loss) = _fill_order(
                pend_side_1, pend_qty_1, open_price,
                balance, pos_side, pos_qty, pos_entry, pos_margin,
                exchange_leverage, commission_rate,
                total_trades, wins, losses, total_profit, total_loss)

        # Clear pending
        n_pending = 0
        pend_side_0 = 0
        pend_qty_0 = 0.0
        pend_side_1 = 0
        pend_qty_1 = 0.0

        # === 3. Update unrealized PnL ===
        if pos_side == _LONG:
            pos_unrealized_pnl = pos_qty * (close - pos_entry) / pos_entry
        elif pos_side == _SHORT:
            pos_unrealized_pnl = pos_qty * (pos_entry - close) / pos_entry
        else:
            pos_unrealized_pnl = 0.0

        # === 4. Check liquidation ===
        if pos_side != _NO_POS:
            equity_in_pos = pos_margin + pos_unrealized_pnl
            if equity_in_pos <= 0 or (
                pos_margin / equity_in_pos >= 1.0 / maintenance_margin
            ):
                total_trades += 1
                losses += 1
                total_loss += pos_margin
                balance -= min(pos_margin, balance)
                pos_side = _NO_POS
                pos_qty = 0.0
                pos_entry = 0.0
                pos_margin = 0.0
                pos_unrealized_pnl = 0.0

        # === 5. Record equity & compute metrics ===
        equity = balance
        if pos_side != _NO_POS:
            equity += pos_margin + pos_unrealized_pnl

        if equity > peak_equity:
            peak_equity = equity
        if peak_equity > 0:
            dd = (peak_equity - equity) / peak_equity
            if dd > max_dd:
                max_dd = dd

        if prev_equity > 0 and equity > 0 and i > 0:
            ret = (equity - prev_equity) / prev_equity
            sum_ret += ret
            sum_ret_sq += ret * ret
            n_returns += 1
            if ret < 0:
                sum_down_sq += ret * ret
                n_downside += 1
        prev_equity = equity

        # === 6. Strategy logic ===
        if close > open_price:
            direction = 1
        elif close < open_price:
            direction = -1
        else:
            direction = 0

        if direction == 0:
            continue

        # Update streak
        if direction == streak_direction:
            consecutive_count += 1
        else:
            consecutive_count = 1
            streak_direction = direction

        # Strategy decision
        if pos_side == _NO_POS:
            profit_candle_count = 0  # reset stale count from external close (e.g. liquidation)
            # Try open
            qty = _calc_quantity(
                consecutive_count, threshold, balance, initial_pct, multiplier, sizing_leverage
            )
            if qty > 0:
                if direction == 1:
                    pend_side_0 = _SELL
                else:
                    pend_side_0 = _BUY
                pend_qty_0 = qty
                n_pending = 1
        else:
            # Check if profit candle
            is_profit = (pos_side == _LONG and direction == 1) or (
                pos_side == _SHORT and direction == -1
            )
            if is_profit:
                profit_candle_count += 1
                if profit_candle_count >= profit_threshold:
                    # Close order
                    if pos_side == _LONG:
                        pend_side_0 = _SELL
                    else:
                        pend_side_0 = _BUY
                    pend_qty_0 = pos_qty
                    n_pending = 1
                    profit_candle_count = 0

                    # Try reopen
                    reopen_qty = _calc_quantity(
                        consecutive_count, threshold, balance, initial_pct, multiplier, sizing_leverage
                    )
                    if reopen_qty > 0:
                        if direction == 1:
                            pend_side_1 = _SELL
                        else:
                            pend_side_1 = _BUY
                        pend_qty_1 = reopen_qty
                        n_pending = 2
            else:
                # Loss candle - close immediately + try reopen (not martingale add)
                profit_candle_count = 0
                if pos_side == _LONG:
                    pend_side_0 = _SELL
                else:
                    pend_side_0 = _BUY
                pend_qty_0 = pos_qty
                n_pending = 1

                # Try reopen contrarian
                reopen_qty = _calc_quantity(
                    consecutive_count, threshold, balance, initial_pct, multiplier, sizing_leverage
                )
                if reopen_qty > 0:
                    if direction == 1:
                        pend_side_1 = _SELL
                    else:
                        pend_side_1 = _BUY
                    pend_qty_1 = reopen_qty
                    n_pending = 2

    # === Compute final metrics (same as simulate_martingale) ===
    final_equity = balance
    if pos_side != _NO_POS:
        final_equity += pos_margin + pos_unrealized_pnl

    net_return = (final_equity - initial_balance) / initial_balance if initial_balance > 0 else 0.0

    # Annual return
    annual_return = 0.0
    if n >= 2:
        days = (bars[n - 1, 0] - bars[0, 0]) / (1000.0 * 86400.0)
        if days > 0 and (1.0 + net_return) > 0:
            annual_return = (1.0 + net_return) ** (365.0 / days) - 1.0
        elif net_return <= -1.0:
            annual_return = -1.0

    # Sharpe ratio (annualized, hourly returns assumed)
    sharpe = 0.0
    if net_return <= -1.0:
        sharpe = -999.0
    elif n_returns >= 2:
        mean_ret = sum_ret / n_returns
        variance = (sum_ret_sq / n_returns) - (mean_ret * mean_ret)
        variance = variance * n_returns / (n_returns - 1)
        if variance > 0:
            std = math.sqrt(variance)
            sharpe = (mean_ret - 0.0) * math.sqrt(365.0 * 24.0) / std

    # Sortino ratio
    sortino = 0.0
    if net_return <= -1.0:
        sortino = -999.0
    elif n_returns >= 2:
        mean_ret = sum_ret / n_returns
        if n_downside > 0:
            down_std = math.sqrt(sum_down_sq / n_downside)
            if down_std > 0:
                sortino = (mean_ret - 0.0) * math.sqrt(365.0 * 24.0) / down_std
        elif mean_ret > 0:
            sortino = 1e10

    # Win rate
    closing_trades = wins + losses
    win_rate = wins / closing_trades if closing_trades > 0 else 0.0

    # Profit factor
    profit_factor = total_profit / total_loss if total_loss > 0 else 1e10

    return (
        net_return,
        annual_return,
        max_dd,
        sharpe,
        sortino,
        win_rate,
        profit_factor,
        total_trades,
    )


# Backwards compatibility: alias simulate to simulate_martingale
simulate = simulate_martingale


@njit(cache=True)
def _calc_quantity(
    consecutive_count: int,
    threshold: int,
    balance: float,
    initial_pct: float,
    multiplier: float,
    leverage: int,
) -> float:
    """Calculate position quantity (USDT notional)."""
    if consecutive_count < threshold:
        return 0.0
    base = balance * initial_pct
    n = consecutive_count - threshold + 1
    mult = multiplier ** (n - 1)
    return base * mult * leverage
