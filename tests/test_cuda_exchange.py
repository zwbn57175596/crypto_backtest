"""
Tests for CUDA exchange device functions.

These tests verify that the CUDA device functions produce identical results
to their CPU numba counterparts. All tests are skipped if CUDA is not available.
"""

import pytest

# Check CUDA availability
try:
    from numba import cuda
    HAS_CUDA = cuda.is_available()
except Exception:
    HAS_CUDA = False

# Skip all tests in this module if CUDA is not available
pytestmark = pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")


if HAS_CUDA:
    import numpy as np
    from numba import cuda
    from backtest.cuda_exchange import (
        device_fill_order,
        device_calc_quantity,
        NO_POS,
        LONG,
        SHORT,
        BUY,
        SELL,
    )

    @cuda.jit
    def _kernel_fill_order(
        order_side, order_qty, fill_price,
        balance, pos_side, pos_qty, pos_entry, pos_margin,
        leverage, commission_rate,
        total_trades, wins, losses, total_profit, total_loss,
        result,
    ):
        """Kernel wrapper to call device_fill_order and store results."""
        if cuda.grid(1) == 0:
            (balance, pos_side, pos_qty, pos_entry, pos_margin,
             total_trades, wins, losses, total_profit, total_loss) = device_fill_order(
                order_side, order_qty, fill_price,
                balance, pos_side, pos_qty, pos_entry, pos_margin,
                leverage, commission_rate,
                total_trades, wins, losses, total_profit, total_loss,
            )
            result[0] = balance
            result[1] = pos_side
            result[2] = pos_qty
            result[3] = pos_entry
            result[4] = pos_margin
            result[5] = total_trades
            result[6] = wins
            result[7] = losses
            result[8] = total_profit
            result[9] = total_loss

    @cuda.jit
    def _kernel_calc_quantity(
        consecutive_count,
        threshold,
        balance,
        initial_pct,
        multiplier,
        leverage,
        result,
    ):
        """Kernel wrapper to call device_calc_quantity and store result."""
        if cuda.grid(1) == 0:
            qty = device_calc_quantity(
                consecutive_count,
                threshold,
                balance,
                initial_pct,
                multiplier,
                leverage,
            )
            result[0] = qty

    def call_device_fill_order(
        order_side, order_qty, fill_price,
        balance, pos_side, pos_qty, pos_entry, pos_margin,
        leverage, commission_rate,
        total_trades, wins, losses, total_profit, total_loss,
    ):
        """Call device_fill_order via kernel and return results."""
        result = np.zeros(10, dtype=np.float64)
        _kernel_fill_order[1, 32](
            order_side, order_qty, fill_price,
            balance, pos_side, pos_qty, pos_entry, pos_margin,
            leverage, commission_rate,
            total_trades, wins, losses, total_profit, total_loss,
            result,
        )
        return (
            result[0],  # balance
            int(result[1]),  # pos_side
            result[2],  # pos_qty
            result[3],  # pos_entry
            result[4],  # pos_margin
            int(result[5]),  # total_trades
            int(result[6]),  # wins
            int(result[7]),  # losses
            result[8],  # total_profit
            result[9],  # total_loss
        )

    def call_device_calc_quantity(
        consecutive_count,
        threshold,
        balance,
        initial_pct,
        multiplier,
        leverage,
    ):
        """Call device_calc_quantity via kernel and return result."""
        result = np.zeros(1, dtype=np.float64)
        _kernel_calc_quantity[1, 32](
            consecutive_count,
            threshold,
            balance,
            initial_pct,
            multiplier,
            leverage,
            result,
        )
        return result[0]

    class TestDeviceFillOrder:
        """Tests for device_fill_order function."""

        def test_open_long(self):
            """Test opening a long position."""
            order_side = BUY
            order_qty = 1000.0
            fill_price = 40000.0
            balance = 10000.0
            pos_side = NO_POS
            pos_qty = 0.0
            pos_entry = 0.0
            pos_margin = 0.0
            leverage = 10
            commission_rate = 0.0004
            total_trades = 0
            wins = 0
            losses = 0
            total_profit = 0.0
            total_loss = 0.0

            (new_balance, new_pos_side, new_pos_qty, new_pos_entry, new_pos_margin,
             new_total_trades, new_wins, new_losses, new_total_profit, new_total_loss) = (
                call_device_fill_order(
                    order_side, order_qty, fill_price,
                    balance, pos_side, pos_qty, pos_entry, pos_margin,
                    leverage, commission_rate,
                    total_trades, wins, losses, total_profit, total_loss,
                )
            )

            # Commission = 1000 * 0.0004 = 0.4
            # Margin = 1000 / 10 = 100
            # New balance = 10000 - 0.4 - 100 = 9899.6
            assert abs(new_balance - 9899.6) < 1e-6
            assert new_pos_side == LONG
            assert abs(new_pos_qty - 1000.0) < 1e-6
            assert abs(new_pos_entry - 40000.0) < 1e-6
            assert abs(new_pos_margin - 100.0) < 1e-6
            assert new_total_trades == 1
            assert new_wins == 0
            assert new_losses == 0

        def test_open_short(self):
            """Test opening a short position."""
            order_side = SELL
            order_qty = 500.0
            fill_price = 42000.0
            balance = 5000.0
            pos_side = NO_POS
            pos_qty = 0.0
            pos_entry = 0.0
            pos_margin = 0.0
            leverage = 50
            commission_rate = 0.0004
            total_trades = 0
            wins = 0
            losses = 0
            total_profit = 0.0
            total_loss = 0.0

            (new_balance, new_pos_side, new_pos_qty, new_pos_entry, new_pos_margin,
             new_total_trades, new_wins, new_losses, new_total_profit, new_total_loss) = (
                call_device_fill_order(
                    order_side, order_qty, fill_price,
                    balance, pos_side, pos_qty, pos_entry, pos_margin,
                    leverage, commission_rate,
                    total_trades, wins, losses, total_profit, total_loss,
                )
            )

            # Commission = 500 * 0.0004 = 0.2
            # Margin = 500 / 50 = 10
            # New balance = 5000 - 0.2 - 10 = 4989.8
            assert abs(new_balance - 4989.8) < 1e-6
            assert new_pos_side == SHORT
            assert abs(new_pos_qty - 500.0) < 1e-6
            assert abs(new_pos_entry - 42000.0) < 1e-6
            assert abs(new_pos_margin - 10.0) < 1e-6
            assert new_total_trades == 1

        def test_close_long_profit(self):
            """Test closing a long position with profit."""
            # Setup: existing long position
            order_side = SELL
            order_qty = 1000.0
            fill_price = 41000.0
            balance = 10024.6  # Arbitrary balance
            pos_side = LONG
            pos_qty = 1000.0
            pos_entry = 40000.0
            pos_margin = 100.0
            leverage = 10
            commission_rate = 0.0004
            total_trades = 1
            wins = 0
            losses = 0
            total_profit = 0.0
            total_loss = 0.0

            (new_balance, new_pos_side, new_pos_qty, new_pos_entry, new_pos_margin,
             new_total_trades, new_wins, new_losses, new_total_profit, new_total_loss) = (
                call_device_fill_order(
                    order_side, order_qty, fill_price,
                    balance, pos_side, pos_qty, pos_entry, pos_margin,
                    leverage, commission_rate,
                    total_trades, wins, losses, total_profit, total_loss,
                )
            )

            # Commission = 1000 * 0.0004 = 0.4
            # PnL = 1000 * (41000 - 40000) / 40000 = 25.0
            # Margin returned = 100 * (1000 / 1000) = 100
            # New balance = 10024.6 - 0.4 + 25.0 + 100 = 10149.2
            assert abs(new_balance - 10149.2) < 1e-6
            assert new_pos_side == NO_POS
            assert abs(new_pos_qty - 0.0) < 1e-6
            assert new_total_trades == 2
            assert new_wins == 1
            assert new_losses == 0
            assert abs(new_total_profit - 25.0) < 1e-6

        def test_close_short_loss(self):
            """Test closing a short position with loss."""
            # Setup: existing short position
            order_side = BUY
            order_qty = 1000.0
            fill_price = 41000.0
            balance = 5000.0
            pos_side = SHORT
            pos_qty = 1000.0
            pos_entry = 40000.0
            pos_margin = 20.0
            leverage = 50
            commission_rate = 0.0004
            total_trades = 1
            wins = 0
            losses = 0
            total_profit = 0.0
            total_loss = 0.0

            (new_balance, new_pos_side, new_pos_qty, new_pos_entry, new_pos_margin,
             new_total_trades, new_wins, new_losses, new_total_profit, new_total_loss) = (
                call_device_fill_order(
                    order_side, order_qty, fill_price,
                    balance, pos_side, pos_qty, pos_entry, pos_margin,
                    leverage, commission_rate,
                    total_trades, wins, losses, total_profit, total_loss,
                )
            )

            # Commission = 1000 * 0.0004 = 0.4
            # PnL = 1000 * (40000 - 41000) / 40000 = -25.0
            # Margin returned = 20 * (1000 / 1000) = 20
            # New balance = 5000 - 0.4 - 25.0 + 20 = 4994.6
            assert abs(new_balance - 4994.6) < 1e-6
            assert new_pos_side == NO_POS
            assert abs(new_pos_qty - 0.0) < 1e-6
            assert new_total_trades == 2
            assert new_wins == 0
            assert new_losses == 1
            assert abs(new_total_loss - 25.0) < 1e-6

    class TestDeviceCalcQuantity:
        """Tests for device_calc_quantity function."""

        def test_below_threshold(self):
            """Test that quantity is 0 below threshold."""
            qty = call_device_calc_quantity(
                consecutive_count=2,
                threshold=5,
                balance=1000.0,
                initial_pct=0.01,
                multiplier=1.1,
                leverage=10,
            )
            assert abs(qty - 0.0) < 1e-6

        def test_at_threshold(self):
            """Test quantity calculation at threshold."""
            # consecutive_count = 5, threshold = 5
            # n = 5 - 5 + 1 = 1
            # mult = 1.1 ** (1 - 1) = 1.0
            # qty = 1000 * 0.01 * 1.0 * 10 = 100.0
            qty = call_device_calc_quantity(
                consecutive_count=5,
                threshold=5,
                balance=1000.0,
                initial_pct=0.01,
                multiplier=1.1,
                leverage=10,
            )
            assert abs(qty - 100.0) < 1e-6

        def test_above_threshold(self):
            """Test quantity calculation above threshold with multiplier."""
            # consecutive_count = 7, threshold = 5
            # n = 7 - 5 + 1 = 3
            # mult = 1.1 ** (3 - 1) = 1.1 ** 2 = 1.21
            # qty = 1000 * 0.01 * 1.21 * 10 = 121.0
            qty = call_device_calc_quantity(
                consecutive_count=7,
                threshold=5,
                balance=1000.0,
                initial_pct=0.01,
                multiplier=1.1,
                leverage=10,
            )
            assert abs(qty - 121.0) < 1e-6

else:
    # Placeholder test class when CUDA is not available
    class TestCudaNotAvailable:
        def test_cuda_not_available(self):
            pytest.skip("CUDA not available")
