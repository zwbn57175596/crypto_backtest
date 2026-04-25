"""
CUDA device functions for exchange logic in backtesting simulation.

This module provides @cuda.jit(device=True) functions that implement
order filling and quantity calculation logic. These functions are designed
to be called from CUDA strategy kernels for GPU-accelerated backtesting.

The logic is identical to the CPU numba version in numba_simulate.py.
"""

try:
    from numba import cuda
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False

# Position side constants
NO_POS = 0
LONG = 1
SHORT = 2

# Order side constants
BUY = 1
SELL = 2


if HAS_CUDA:
    @cuda.jit(device=True)
    def device_fill_order(
        order_side, order_qty, fill_price,
        balance, pos_side, pos_qty, pos_entry, pos_margin,
        leverage, commission_rate,
        total_trades, wins, losses, total_profit, total_loss,
    ):
        """Process a single order fill. Returns updated state.

        This is a CUDA device function (runs on GPU). Logic is identical to
        numba_simulate._fill_order.

        Parameters
        ----------
        order_side : int
            BUY (1) or SELL (2)
        order_qty : float
            Order quantity in USDT notional
        fill_price : float
            Fill price in USDT
        balance : float
            Current balance in USDT
        pos_side : int
            Position side: NO_POS (0), LONG (1), or SHORT (2)
        pos_qty : float
            Current position quantity in USDT notional
        pos_entry : float
            Entry price of current position
        pos_margin : float
            Margin allocated to current position
        leverage : int
            Leverage used for margin calculation
        commission_rate : float
            Commission rate (e.g., 0.0004 = 0.04%)
        total_trades : int
            Count of total trades executed
        wins : int
            Count of profitable closes
        losses : int
            Count of loss closes
        total_profit : float
            Sum of all profits
        total_loss : float
            Sum of all losses (absolute values)

        Returns
        -------
        tuple
            (balance, pos_side, pos_qty, pos_entry, pos_margin,
             total_trades, wins, losses, total_profit, total_loss)
        """
        commission = order_qty * commission_rate
        balance -= commission

        pnl = 0.0

        if pos_side == NO_POS:
            # Open new position
            margin = order_qty / leverage
            balance -= margin
            pos_side = LONG if order_side == BUY else SHORT
            pos_qty = order_qty
            pos_entry = fill_price
            pos_margin = margin
            total_trades += 1
        elif (pos_side == LONG and order_side == SELL) or (
            pos_side == SHORT and order_side == BUY
        ):
            # Close position
            close_qty = min(order_qty, pos_qty)
            if pos_side == LONG:
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
                pos_side = NO_POS
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
                pos_side = LONG if order_side == BUY else SHORT
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


    @cuda.jit(device=True)
    def device_calc_quantity(
        consecutive_count,
        threshold,
        balance,
        initial_pct,
        multiplier,
        leverage,
    ):
        """Calculate position quantity (USDT notional).

        This is a CUDA device function (runs on GPU). Logic is identical to
        numba_simulate._calc_quantity.

        Parameters
        ----------
        consecutive_count : int
            Number of consecutive bars in the same direction
        threshold : int
            Minimum consecutive bars to trigger position opening
        balance : float
            Current account balance in USDT
        initial_pct : float
            Initial position size as a percentage of balance
        multiplier : float
            Multiplier for exponential position sizing
        leverage : int
            Leverage for position sizing

        Returns
        -------
        float
            Position quantity in USDT notional (0 if threshold not met)
        """
        if consecutive_count < threshold:
            return 0.0
        base = balance * initial_pct
        n = consecutive_count - threshold + 1
        # Use a simple loop to compute multiplier ** (n - 1)
        # since Numba/CUDA may not support ** for device functions
        mult = 1.0
        for _ in range(n - 1):
            mult *= multiplier
        return base * mult * leverage

else:
    # Stubs for when CUDA is not available
    def device_fill_order(*args, **kwargs):
        raise RuntimeError("CUDA not available")

    def device_calc_quantity(*args, **kwargs):
        raise RuntimeError("CUDA not available")
