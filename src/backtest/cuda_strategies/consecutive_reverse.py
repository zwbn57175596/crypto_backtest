"""ConsecutiveReverse CUDA kernel for GPU-accelerated backtesting.

This module implements the ConsecutiveReverse strategy as a CUDA kernel that can
run thousands of parameter combinations in parallel on GPU.

The kernel logic is ported from the CPU numba version in numba_simulate.py and
uses device functions from cuda_exchange.py for order filling and quantity calculation.
"""

import math

try:
    from numba import cuda
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False

if HAS_CUDA:
    from backtest.cuda_exchange import (
        NO_POS, LONG, SHORT, BUY, SELL,
        device_fill_order, device_calc_quantity
    )

    @cuda.jit
    def consecutive_reverse_kernel(
        bars, params, results, n_bars, n_combos,
        exchange_leverage, commission_rate, funding_rate,
        maintenance_margin, initial_balance
    ):
        """CUDA kernel for ConsecutiveReverse strategy backtesting.

        Each GPU thread runs one parameter combination through all bars.

        Parameters
        ----------
        bars : float64[:, 6]
            [timestamp_ms, open, high, low, close, volume]
        params : float64[:, 5]
            [threshold, multiplier, initial_pct, profit_threshold, sizing_leverage]
            for each combination
        results : float64[:, 8]
            Output metrics [net_return, annual_return, max_drawdown, sharpe,
            sortino, win_rate, profit_factor, total_trades]
        n_bars : int
            Number of bars
        n_combos : int
            Number of parameter combinations
        exchange_leverage : int
            Exchange leverage (for margin calculation)
        commission_rate : float
            Commission rate (e.g., 0.0004)
        funding_rate : float
            Funding rate (e.g., 0.0001)
        maintenance_margin : float
            Maintenance margin ratio (e.g., 0.5)
        initial_balance : float
            Starting balance in USDT
        """
        # One thread per combination
        idx = cuda.grid(1)
        if idx >= n_combos:
            return

        if n_bars == 0:
            results[idx, 0] = 0.0  # net_return
            results[idx, 1] = 0.0  # annual_return
            results[idx, 2] = 0.0  # max_drawdown
            results[idx, 3] = 0.0  # sharpe
            results[idx, 4] = 0.0  # sortino
            results[idx, 5] = 0.0  # win_rate
            results[idx, 6] = 0.0  # profit_factor
            results[idx, 7] = 0.0  # total_trades
            return

        # Extract parameters for this combination
        threshold = int(params[idx, 0])
        multiplier = params[idx, 1]
        initial_pct = params[idx, 2]
        profit_threshold = int(params[idx, 3])
        sizing_leverage = int(params[idx, 4])

        # Exchange state
        balance = initial_balance
        pos_side = NO_POS
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

        # === Main bar loop ===
        for i in range(n_bars):
            ts = bars[i, 0]
            open_price = bars[i, 1]
            close = bars[i, 4]

            # === 1. Settle funding ===
            ts_sec = int(ts) // 1000
            hour = (ts_sec // 3600) % 24
            minute = (ts_sec % 3600) // 60
            if minute == 0 and (hour == 0 or hour == 8 or hour == 16):
                if pos_side != NO_POS:
                    payment = pos_qty * funding_rate
                    if pos_side == LONG:
                        balance -= payment
                    else:
                        balance += payment

            # === 2. Match pending orders (all fill at this bar's open) ===
            if n_pending >= 1:
                (balance, pos_side, pos_qty, pos_entry, pos_margin,
                 total_trades, wins, losses, total_profit, total_loss) = device_fill_order(
                    pend_side_0, pend_qty_0, open_price,
                    balance, pos_side, pos_qty, pos_entry, pos_margin,
                    exchange_leverage, commission_rate,
                    total_trades, wins, losses, total_profit, total_loss)

            if n_pending >= 2:
                (balance, pos_side, pos_qty, pos_entry, pos_margin,
                 total_trades, wins, losses, total_profit, total_loss) = device_fill_order(
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
            if pos_side == LONG:
                pos_unrealized_pnl = pos_qty * (close - pos_entry) / pos_entry
            elif pos_side == SHORT:
                pos_unrealized_pnl = pos_qty * (pos_entry - close) / pos_entry
            else:
                pos_unrealized_pnl = 0.0

            # === 4. Check liquidation ===
            if pos_side != NO_POS:
                equity_in_pos = pos_margin + pos_unrealized_pnl
                if equity_in_pos <= 0 or (
                    pos_margin / equity_in_pos >= 1.0 / maintenance_margin
                ):
                    total_trades += 1
                    losses += 1
                    total_loss += pos_margin
                    # Use if-else instead of min() for CUDA compatibility
                    if pos_margin < balance:
                        balance -= pos_margin
                    else:
                        balance = 0.0
                    pos_side = NO_POS
                    pos_qty = 0.0
                    pos_entry = 0.0
                    pos_margin = 0.0
                    pos_unrealized_pnl = 0.0

            # === 5. Record equity & compute metrics ===
            equity = balance
            if pos_side != NO_POS:
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
            if pos_side == NO_POS:
                profit_candle_count = 0  # reset stale count
                # Try open
                qty = device_calc_quantity(
                    consecutive_count, threshold, balance, initial_pct, multiplier, sizing_leverage
                )
                if qty > 0:
                    if direction == 1:
                        pend_side_0 = SELL
                    else:
                        pend_side_0 = BUY
                    pend_qty_0 = qty
                    n_pending = 1
            else:
                # Check if profit candle
                is_profit = (pos_side == LONG and direction == 1) or (
                    pos_side == SHORT and direction == -1
                )
                if is_profit:
                    profit_candle_count += 1
                    if profit_candle_count >= profit_threshold:
                        # Close order
                        if pos_side == LONG:
                            pend_side_0 = SELL
                        else:
                            pend_side_0 = BUY
                        pend_qty_0 = pos_qty
                        n_pending = 1
                        profit_candle_count = 0

                        # Try reopen
                        reopen_qty = device_calc_quantity(
                            consecutive_count, threshold, balance, initial_pct, multiplier, sizing_leverage
                        )
                        if reopen_qty > 0:
                            if direction == 1:
                                pend_side_1 = SELL
                            else:
                                pend_side_1 = BUY
                            pend_qty_1 = reopen_qty
                            n_pending = 2
                else:
                    # Loss candle - close immediately
                    if pos_side == LONG:
                        pend_side_0 = SELL
                    else:
                        pend_side_0 = BUY
                    pend_qty_0 = pos_qty
                    n_pending = 1
                    profit_candle_count = 0

                    # Try reopen after close
                    reopen_qty = device_calc_quantity(
                        consecutive_count, threshold, balance, initial_pct, multiplier, sizing_leverage
                    )
                    if reopen_qty > 0:
                        if direction == 1:
                            pend_side_1 = SELL
                        else:
                            pend_side_1 = BUY
                        pend_qty_1 = reopen_qty
                        n_pending = 2

        # === Compute final metrics ===
        final_equity = balance
        if pos_side != NO_POS:
            final_equity += pos_margin + pos_unrealized_pnl

        net_return = (final_equity - initial_balance) / initial_balance if initial_balance > 0 else 0.0

        # Annual return
        annual_return = 0.0
        if n_bars >= 2:
            days = (bars[n_bars - 1, 0] - bars[0, 0]) / (1000.0 * 86400.0)
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

        # Write results
        results[idx, 0] = net_return
        results[idx, 1] = annual_return
        results[idx, 2] = max_dd
        results[idx, 3] = sharpe
        results[idx, 4] = sortino
        results[idx, 5] = win_rate
        results[idx, 6] = profit_factor
        results[idx, 7] = float(total_trades)

else:
    # Stub for when CUDA is not available
    def consecutive_reverse_kernel(*args, **kwargs):
        raise RuntimeError("CUDA not available")
