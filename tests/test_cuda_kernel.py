"""Tests for CUDA kernel implementation of ConsecutiveReverse strategy.

Tests compare CUDA kernel output to CPU numba simulate implementation to ensure
correctness and numerical accuracy.
"""

import pytest
import numpy as np

try:
    from numba import cuda
    HAS_CUDA = cuda.is_available()
except ImportError:
    HAS_CUDA = False

from backtest.numba_simulate import simulate
from backtest.cuda_strategies.consecutive_reverse import consecutive_reverse_kernel


# Skip all tests if CUDA is not available
pytestmark = pytest.mark.skipif(
    not HAS_CUDA,
    reason="CUDA not available (expected on macOS, CI without GPU)"
)


def _make_bars(n: int, start_ts: int = 0) -> np.ndarray:
    """Generate synthetic OHLCV bars for testing.

    Parameters
    ----------
    n : int
        Number of bars
    start_ts : int
        Starting timestamp in milliseconds

    Returns
    -------
    ndarray shape (n, 6)
        [timestamp_ms, open, high, low, close, volume]
    """
    bars = np.zeros((n, 6), dtype=np.float64)

    # Generate bars with some realistic price movement
    price = 100.0
    for i in range(n):
        bars[i, 0] = start_ts + i * 3600000  # 1-hour bars (3600000 ms)
        bars[i, 1] = price  # open

        # Add realistic movement: +/- 1-3%
        move = (i % 3 - 1) * 0.02 * price  # -2%, 0%, or +2%
        price_close = price + move
        bars[i, 4] = price_close  # close

        # High/low around the close
        bars[i, 2] = max(price, price_close) + 0.01 * price  # high
        bars[i, 3] = min(price, price_close) - 0.01 * price  # low
        bars[i, 5] = 1000.0  # volume

        price = price_close

    return bars


@pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
def test_single_combo_matches_cpu():
    """Test that a single parameter combo on CUDA matches CPU numba simulate."""
    bars = _make_bars(100)

    # CPU baseline
    cpu_result = simulate(
        bars=bars,
        threshold=3,
        multiplier=1.1,
        initial_pct=0.01,
        profit_threshold=1,
        sizing_leverage=10,
        exchange_leverage=10,
        commission_rate=0.0004,
        funding_rate=0.0001,
        maintenance_margin=0.5,
        initial_balance=10000.0,
    )

    # CUDA: prepare arrays
    params = np.array([[3, 1.1, 0.01, 1, 10]], dtype=np.float64)
    results = np.zeros((1, 8), dtype=np.float64)

    # Run kernel
    consecutive_reverse_kernel[1, 32](
        bars, params, results, len(bars), 1,
        exchange_leverage=10,
        commission_rate=0.0004,
        funding_rate=0.0001,
        maintenance_margin=0.5,
        initial_balance=10000.0
    )

    # Compare results (relative error < 1e-6 for each metric)
    for i in range(8):
        cpu_val = cpu_result[i]
        cuda_val = results[0, i]

        if cpu_val != 0:
            rel_err = abs(cuda_val - cpu_val) / abs(cpu_val)
            assert rel_err < 1e-6, (
                f"Metric {i}: CUDA={cuda_val}, CPU={cpu_val}, "
                f"rel_err={rel_err}"
            )
        else:
            # For zero values, allow small absolute error
            assert abs(cuda_val - cpu_val) < 1e-10, (
                f"Metric {i}: CUDA={cuda_val}, CPU={cpu_val}"
            )


@pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
def test_multiple_combos():
    """Test that multiple parameter combos on CUDA all match CPU."""
    bars = _make_bars(50)

    # Test 3 different parameter combinations
    combos = [
        (2, 1.05, 0.005, 2, 5),
        (3, 1.1, 0.01, 1, 10),
        (5, 1.2, 0.02, 3, 20),
    ]

    for threshold, multiplier, initial_pct, profit_threshold, sizing_leverage in combos:
        # CPU baseline
        cpu_result = simulate(
            bars=bars,
            threshold=threshold,
            multiplier=multiplier,
            initial_pct=initial_pct,
            profit_threshold=profit_threshold,
            sizing_leverage=sizing_leverage,
            exchange_leverage=10,
            commission_rate=0.0004,
            funding_rate=0.0001,
            maintenance_margin=0.5,
            initial_balance=10000.0,
        )

        # CUDA: prepare arrays for this combo
        params = np.array(
            [[threshold, multiplier, initial_pct, profit_threshold, sizing_leverage]],
            dtype=np.float64
        )
        results = np.zeros((1, 8), dtype=np.float64)

        # Run kernel
        consecutive_reverse_kernel[1, 32](
            bars, params, results, len(bars), 1,
            exchange_leverage=10,
            commission_rate=0.0004,
            funding_rate=0.0001,
            maintenance_margin=0.5,
            initial_balance=10000.0
        )

        # Compare all 8 metrics
        for i in range(8):
            cpu_val = cpu_result[i]
            cuda_val = results[0, i]

            if cpu_val != 0:
                rel_err = abs(cuda_val - cpu_val) / abs(cpu_val)
                assert rel_err < 1e-6, (
                    f"Combo {threshold},{multiplier},{initial_pct},"
                    f"{profit_threshold},{sizing_leverage}: "
                    f"Metric {i} mismatch: CUDA={cuda_val}, CPU={cpu_val}"
                )
            else:
                assert abs(cuda_val - cpu_val) < 1e-10


@pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
def test_zero_bars():
    """Test that zero bars produces all-zero metrics."""
    bars = np.zeros((0, 6), dtype=np.float64)

    params = np.array([[3, 1.1, 0.01, 1, 10]], dtype=np.float64)
    results = np.zeros((1, 8), dtype=np.float64)

    consecutive_reverse_kernel[1, 32](
        bars, params, results, 0, 1,
        exchange_leverage=10,
        commission_rate=0.0004,
        funding_rate=0.0001,
        maintenance_margin=0.5,
        initial_balance=10000.0
    )

    # All metrics should be 0
    for i in range(8):
        assert results[0, i] == 0.0, f"Metric {i} should be 0.0, got {results[0, i]}"
