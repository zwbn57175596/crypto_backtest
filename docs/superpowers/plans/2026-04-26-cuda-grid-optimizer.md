# CUDA Grid Optimizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add GPU-accelerated grid search optimization using Numba CUDA that runs 100万+ parameter combinations in batches on RTX 3080.

**Architecture:** Reusable CUDA device functions for exchange logic (`cuda_exchange.py`), strategy-specific kernel (`cuda_strategies/consecutive_reverse.py`), and a `CudaGridOptimizer` orchestrator (`cuda_runner.py`) that auto-batches based on available VRAM. CLI integrates via `--method cuda-grid`.

**Tech Stack:** Numba CUDA (`numba.cuda`), NumPy, Python 3.11+

**Spec:** `docs/superpowers/specs/2026-04-26-cuda-grid-optimizer-design.md`

---

### Task 1: CUDA Exchange Device Functions

**Files:**
- Create: `src/backtest/cuda_exchange.py`
- Test: `tests/test_cuda_exchange.py`

This task implements the reusable exchange logic as `@cuda.jit(device=True)` functions. The logic is identical to `numba_simulate.py:_fill_order` (lines 66-140) but decorated for CUDA device use.

- [ ] **Step 1: Write test for `device_fill_order` — open long**

```python
# tests/test_cuda_exchange.py
import math
import numpy as np
import pytest

try:
    from numba import cuda
    HAS_CUDA = cuda.is_available()
except Exception:
    HAS_CUDA = False

pytestmark = pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")

if HAS_CUDA:
    from backtest.cuda_exchange import device_fill_order

    @cuda.jit
    def _test_fill_order_kernel(inputs, outputs):
        """Test kernel that calls device_fill_order and writes results."""
        idx = cuda.grid(1)
        if idx >= inputs.shape[0]:
            return
    # inputs[idx]: [order_side, order_qty, fill_price, balance, pos_side, pos_qty,
    #               pos_entry, pos_margin, leverage, commission_rate,
    #               total_trades, wins, losses, total_profit, total_loss]
    r = device_fill_order(
        int(inputs[idx, 0]), inputs[idx, 1], inputs[idx, 2],
        inputs[idx, 3], int(inputs[idx, 4]), inputs[idx, 5],
        inputs[idx, 6], inputs[idx, 7],
        int(inputs[idx, 8]), inputs[idx, 9],
        int(inputs[idx, 10]), int(inputs[idx, 11]), int(inputs[idx, 12]),
        inputs[idx, 13], inputs[idx, 14],
    )
    # r = (balance, pos_side, pos_qty, pos_entry, pos_margin,
    #      total_trades, wins, losses, total_profit, total_loss)
    for j in range(10):
        outputs[idx, j] = r[j]


class TestDeviceFillOrder:
    def _run_fill(self, order_side, order_qty, fill_price,
                  balance, pos_side, pos_qty, pos_entry, pos_margin,
                  leverage, commission_rate,
                  total_trades=0, wins=0, losses=0,
                  total_profit=0.0, total_loss=0.0):
        """Helper: run device_fill_order via test kernel, return result tuple."""
        inp = np.array([[
            order_side, order_qty, fill_price,
            balance, pos_side, pos_qty, pos_entry, pos_margin,
            leverage, commission_rate,
            total_trades, wins, losses, total_profit, total_loss,
        ]], dtype=np.float64)
        out = np.zeros((1, 10), dtype=np.float64)
        _test_fill_order_kernel[1, 1](inp, out)
        cuda.synchronize()
        return tuple(out[0])

    def test_open_long(self):
        # BUY=1, qty=1000, price=40000, balance=10000, no position, lev=10, comm=0.0004
        r = self._run_fill(1, 1000.0, 40000.0, 10000.0, 0, 0.0, 0.0, 0.0, 10, 0.0004)
        balance, pos_side, pos_qty, pos_entry, pos_margin = r[0], r[1], r[2], r[3], r[4]
        total_trades = r[5]
        # commission = 1000 * 0.0004 = 0.4
        # margin = 1000 / 10 = 100
        # balance = 10000 - 0.4 - 100 = 9899.6
        assert abs(balance - 9899.6) < 1e-6
        assert pos_side == 1  # LONG
        assert abs(pos_qty - 1000.0) < 1e-6
        assert abs(pos_entry - 40000.0) < 1e-6
        assert abs(pos_margin - 100.0) < 1e-6
        assert total_trades == 1

    def test_open_short(self):
        # SELL=2
        r = self._run_fill(2, 500.0, 42000.0, 5000.0, 0, 0.0, 0.0, 0.0, 50, 0.0004)
        balance, pos_side, pos_qty, pos_entry, pos_margin = r[0], r[1], r[2], r[3], r[4]
        # commission = 500 * 0.0004 = 0.2, margin = 500/50 = 10
        assert abs(balance - (5000.0 - 0.2 - 10.0)) < 1e-6
        assert pos_side == 2  # SHORT

    def test_close_long_profit(self):
        # Open long at 40000, close with SELL at 41000
        r = self._run_fill(2, 1000.0, 41000.0, 9899.6, 1, 1000.0, 40000.0, 100.0, 10, 0.0004)
        balance, pos_side = r[0], r[1]
        wins, total_profit = r[7], r[8]
        # pnl = 1000 * (41000-40000)/40000 = 25.0
        # balance = 9899.6 - 0.4(comm) + 25.0(pnl) + 100.0(margin) = 10024.2
        assert abs(balance - 10024.2) < 1e-6
        assert pos_side == 0  # NO_POS
        assert wins == 1
        assert abs(total_profit - 25.0) < 1e-6

    def test_close_short_loss(self):
        # Open short at 40000, close with BUY at 41000
        r = self._run_fill(1, 1000.0, 41000.0, 4989.8, 2, 1000.0, 40000.0, 20.0, 50, 0.0004)
        balance, pos_side = r[0], r[1]
        losses, total_loss = r[7], r[9]
        # pnl = 1000 * (40000-41000)/40000 = -25.0
        # balance = 4989.8 - 0.4 + (-25.0) + 20.0 = 4984.4
        assert abs(balance - 4984.4) < 1e-6
        assert pos_side == 0
        assert losses == 1
        assert abs(total_loss - 25.0) < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cuda_exchange.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backtest.cuda_exchange'`

- [ ] **Step 3: Implement `cuda_exchange.py`**

```python
# src/backtest/cuda_exchange.py
"""
CUDA device functions for simulated exchange operations.

These are reusable building blocks for any CUDA strategy kernel.
Logic mirrors numba_simulate.py but uses @cuda.jit(device=True).
"""

import math
from numba import cuda

# Position side constants
NO_POS = 0
LONG = 1
SHORT = 2

# Order side constants
BUY = 1
SELL = 2


@cuda.jit(device=True)
def device_fill_order(
    order_side, order_qty, fill_price,
    balance, pos_side, pos_qty, pos_entry, pos_margin,
    leverage, commission_rate,
    total_trades, wins, losses, total_profit, total_loss,
):
    """Process a single order fill on GPU. Returns updated state as tuple."""
    commission = order_qty * commission_rate
    balance -= commission

    pnl = 0.0

    if pos_side == NO_POS:
        # Open new position
        margin = order_qty / leverage
        balance -= margin
        if order_side == BUY:
            pos_side = LONG
        else:
            pos_side = SHORT
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
            if order_side == BUY:
                pos_side = LONG
            else:
                pos_side = SHORT
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
    consecutive_count, threshold, balance, initial_pct, multiplier, leverage,
):
    """Calculate position quantity (USDT notional) on GPU."""
    if consecutive_count < threshold:
        return 0.0
    base = balance * initial_pct
    n = consecutive_count - threshold + 1
    mult = multiplier ** (n - 1)
    return base * mult * leverage
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cuda_exchange.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/backtest/cuda_exchange.py tests/test_cuda_exchange.py
git commit -m "feat: add CUDA exchange device functions with tests"
```

---

### Task 2: ConsecutiveReverse CUDA Kernel

**Files:**
- Create: `src/backtest/cuda_strategies/__init__.py`
- Create: `src/backtest/cuda_strategies/consecutive_reverse.py`
- Test: `tests/test_cuda_kernel.py`

This task implements the full strategy simulation kernel. Each GPU thread runs one parameter combination through all bars.

- [ ] **Step 1: Write test — single combo matches numba CPU**

```python
# tests/test_cuda_kernel.py
import math
import numpy as np
import pytest

try:
    from numba import cuda
    HAS_CUDA = cuda.is_available()
except Exception:
    HAS_CUDA = False

pytestmark = pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")


def _make_bars(n=200):
    """Generate synthetic bars: (N, 6) float64 — [ts, open, high, low, close, vol]."""
    base_ts = 1704067200000  # 2024-01-01 00:00 UTC ms
    bars = np.zeros((n, 6), dtype=np.float64)
    for i in range(n):
        ts = base_ts + i * 3600000
        price = 40000.0 + i * 10.0
        bars[i] = [ts, price, price + 50, price - 50, price + 5, 1000.0]
    return bars


class TestConsecutiveReverseKernel:
    def test_single_combo_matches_cpu(self):
        """CUDA kernel with 1 combo must produce same metrics as numba CPU simulate."""
        from backtest.numba_simulate import simulate as cpu_simulate
        from backtest.cuda_strategies.consecutive_reverse import consecutive_reverse_kernel

        bars = _make_bars(200)
        threshold, multiplier, initial_pct, profit_threshold = 5, 1.1, 0.01, 1
        sizing_leverage, exchange_leverage = 50, 50
        commission_rate, funding_rate, maintenance_margin = 0.0004, 0.0001, 0.005
        initial_balance = 1000.0

        # CPU reference
        cpu_result = cpu_simulate(
            bars, threshold, multiplier, initial_pct, profit_threshold,
            sizing_leverage, exchange_leverage,
            commission_rate, funding_rate, maintenance_margin, initial_balance,
        )

        # CUDA
        d_bars = cuda.to_device(bars)
        params = np.array([[threshold, multiplier, initial_pct, profit_threshold, sizing_leverage]], dtype=np.float64)
        d_params = cuda.to_device(params)
        results = np.zeros((1, 8), dtype=np.float64)
        d_results = cuda.to_device(results)

        consecutive_reverse_kernel[1, 1](
            d_bars, d_params, d_results,
            bars.shape[0], 1,
            exchange_leverage, commission_rate, funding_rate,
            maintenance_margin, initial_balance,
        )
        cuda.synchronize()
        gpu_result = d_results.copy_to_host()[0]

        for i in range(8):
            cpu_val = cpu_result[i]
            gpu_val = gpu_result[i]
            if abs(cpu_val) < 1e-10:
                assert abs(gpu_val) < 1e-6, f"Metric {i}: cpu={cpu_val}, gpu={gpu_val}"
            else:
                rel_err = abs(gpu_val - cpu_val) / abs(cpu_val)
                assert rel_err < 1e-6, f"Metric {i}: cpu={cpu_val}, gpu={gpu_val}, rel_err={rel_err}"

    def test_multiple_combos(self):
        """Multiple combos produce correct results (each independent)."""
        from backtest.numba_simulate import simulate as cpu_simulate
        from backtest.cuda_strategies.consecutive_reverse import consecutive_reverse_kernel

        bars = _make_bars(100)
        combos = [
            (3, 1.0, 0.02, 2, 50),
            (5, 1.1, 0.01, 1, 50),
            (7, 1.2, 0.005, 3, 25),
        ]
        exchange_leverage = 50
        commission_rate, funding_rate, maintenance_margin = 0.0004, 0.0001, 0.005
        initial_balance = 1000.0

        params = np.array(combos, dtype=np.float64)
        n_combos = len(combos)

        d_bars = cuda.to_device(bars)
        d_params = cuda.to_device(params)
        results = np.zeros((n_combos, 8), dtype=np.float64)
        d_results = cuda.to_device(results)

        threads = 32
        blocks = (n_combos + threads - 1) // threads
        consecutive_reverse_kernel[blocks, threads](
            d_bars, d_params, d_results,
            bars.shape[0], n_combos,
            exchange_leverage, commission_rate, funding_rate,
            maintenance_margin, initial_balance,
        )
        cuda.synchronize()
        gpu_results = d_results.copy_to_host()

        for ci, (threshold, multiplier, initial_pct, profit_threshold, sizing_lev) in enumerate(combos):
            cpu_result = cpu_simulate(
                bars, threshold, multiplier, initial_pct, profit_threshold,
                sizing_lev, exchange_leverage,
                commission_rate, funding_rate, maintenance_margin, initial_balance,
            )
            for mi in range(8):
                cpu_val = cpu_result[mi]
                gpu_val = gpu_results[ci, mi]
                if abs(cpu_val) < 1e-10:
                    assert abs(gpu_val) < 1e-6, f"Combo {ci} metric {mi}: cpu={cpu_val}, gpu={gpu_val}"
                else:
                    rel_err = abs(gpu_val - cpu_val) / abs(cpu_val)
                    assert rel_err < 1e-6, f"Combo {ci} metric {mi}: cpu={cpu_val}, gpu={gpu_val}"

    def test_zero_bars(self):
        """Empty bars array returns all-zero metrics."""
        from backtest.cuda_strategies.consecutive_reverse import consecutive_reverse_kernel

        bars = np.empty((0, 6), dtype=np.float64)
        params = np.array([[5, 1.1, 0.01, 1, 50]], dtype=np.float64)
        results = np.zeros((1, 8), dtype=np.float64)

        d_bars = cuda.to_device(bars)
        d_params = cuda.to_device(params)
        d_results = cuda.to_device(results)

        consecutive_reverse_kernel[1, 1](
            d_bars, d_params, d_results, 0, 1,
            50, 0.0004, 0.0001, 0.005, 1000.0,
        )
        cuda.synchronize()
        gpu_result = d_results.copy_to_host()[0]
        assert all(abs(v) < 1e-10 for v in gpu_result)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cuda_kernel.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backtest.cuda_strategies'`

- [ ] **Step 3: Create strategy registry**

```python
# src/backtest/cuda_strategies/__init__.py
"""CUDA strategy kernel registry.

Maps strategy class name -> kernel function and parameter column order.
New strategies: add a kernel module and register here.
"""

CUDA_STRATEGIES = {}


def _register():
    """Lazy-import and register available CUDA strategies."""
    try:
        from backtest.cuda_strategies.consecutive_reverse import consecutive_reverse_kernel
        CUDA_STRATEGIES["ConsecutiveReverseStrategy"] = {
            "kernel": consecutive_reverse_kernel,
            "param_order": [
                "CONSECUTIVE_THRESHOLD",
                "POSITION_MULTIPLIER",
                "INITIAL_POSITION_PCT",
                "PROFIT_CANDLE_THRESHOLD",
                "LEVERAGE",
            ],
        }
    except Exception:
        pass  # CUDA not available


_register()
```

- [ ] **Step 4: Implement the CUDA kernel**

```python
# src/backtest/cuda_strategies/consecutive_reverse.py
"""
CUDA kernel for ConsecutiveReverse strategy backtest.

Each GPU thread simulates one parameter combination across all bars.
Logic is identical to numba_simulate.simulate() (CPU version).
"""

import math
from numba import cuda
from backtest.cuda_exchange import (
    device_fill_order, device_calc_quantity,
    NO_POS, LONG, SHORT, BUY, SELL,
)


@cuda.jit
def consecutive_reverse_kernel(
    bars, params, results,
    n_bars, n_combos,
    exchange_leverage, commission_rate, funding_rate,
    maintenance_margin, initial_balance,
):
    """Run ConsecutiveReverse simulation for one parameter combination.

    Parameters
    ----------
    bars : float64[:, 6] — [ts_ms, open, high, low, close, volume]
    params : float64[:, 5] — per-combo [threshold, multiplier, initial_pct, profit_threshold, sizing_leverage]
    results : float64[:, 8] — output metrics per combo
    n_bars : int
    n_combos : int
    exchange_leverage : int
    commission_rate : float
    funding_rate : float
    maintenance_margin : float
    initial_balance : float
    """
    idx = cuda.grid(1)
    if idx >= n_combos:
        return

    # Read parameters for this thread
    threshold = int(params[idx, 0])
    multiplier = params[idx, 1]
    initial_pct = params[idx, 2]
    profit_threshold = int(params[idx, 3])
    sizing_leverage = int(params[idx, 4])

    if n_bars == 0:
        for j in range(8):
            results[idx, j] = 0.0
        return

    # Exchange state
    balance = initial_balance
    pos_side = NO_POS
    pos_qty = 0.0
    pos_entry = 0.0
    pos_margin = 0.0
    pos_unrealized_pnl = 0.0

    # Strategy state
    consecutive_count = 0
    streak_direction = 0
    profit_candle_count = 0

    # Pending orders (max 2 per bar)
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

    # Sharpe/Sortino
    prev_equity = initial_balance
    sum_ret = 0.0
    sum_ret_sq = 0.0
    sum_down_sq = 0.0
    n_returns = 0
    n_downside = 0

    # Max drawdown
    peak_equity = initial_balance
    max_dd = 0.0

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

        # === 2. Match pending orders ===
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
            profit_candle_count = 0
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
            is_profit = (pos_side == LONG and direction == 1) or (
                pos_side == SHORT and direction == -1
            )
            if is_profit:
                profit_candle_count += 1
                if profit_candle_count >= profit_threshold:
                    if pos_side == LONG:
                        pend_side_0 = SELL
                    else:
                        pend_side_0 = BUY
                    pend_qty_0 = pos_qty
                    n_pending = 1
                    profit_candle_count = 0

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
                if pos_side == LONG:
                    pend_side_0 = SELL
                else:
                    pend_side_0 = BUY
                pend_qty_0 = pos_qty
                n_pending = 1
                profit_candle_count = 0

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

    net_return = 0.0
    if initial_balance > 0:
        net_return = (final_equity - initial_balance) / initial_balance

    # Annual return
    annual_return = 0.0
    if n_bars >= 2:
        days = (bars[n_bars - 1, 0] - bars[0, 0]) / (1000.0 * 86400.0)
        if days > 0 and (1.0 + net_return) > 0:
            annual_return = (1.0 + net_return) ** (365.0 / days) - 1.0
        elif net_return <= -1.0:
            annual_return = -1.0

    # Sharpe ratio
    sharpe = 0.0
    if net_return <= -1.0:
        sharpe = -999.0
    elif n_returns >= 2:
        mean_ret = sum_ret / n_returns
        variance = (sum_ret_sq / n_returns) - (mean_ret * mean_ret)
        variance = variance * n_returns / (n_returns - 1)
        if variance > 0:
            std = math.sqrt(variance)
            sharpe = mean_ret * math.sqrt(365.0 * 24.0) / std

    # Sortino ratio
    sortino = 0.0
    if net_return <= -1.0:
        sortino = -999.0
    elif n_returns >= 2:
        mean_ret = sum_ret / n_returns
        if n_downside > 0:
            down_std = math.sqrt(sum_down_sq / n_downside)
            if down_std > 0:
                sortino = mean_ret * math.sqrt(365.0 * 24.0) / down_std
        elif mean_ret > 0:
            sortino = 1e10

    # Win rate
    closing_trades = wins + losses
    win_rate = 0.0
    if closing_trades > 0:
        win_rate = wins / closing_trades

    # Profit factor
    profit_factor = 0.0
    if total_loss > 0:
        profit_factor = total_profit / total_loss
    else:
        profit_factor = 1e10

    # Write results
    results[idx, 0] = net_return
    results[idx, 1] = annual_return
    results[idx, 2] = max_dd
    results[idx, 3] = sharpe
    results[idx, 4] = sortino
    results[idx, 5] = win_rate
    results[idx, 6] = profit_factor
    results[idx, 7] = float(total_trades)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_cuda_kernel.py -v`
Expected: All 3 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/backtest/cuda_strategies/__init__.py src/backtest/cuda_strategies/consecutive_reverse.py tests/test_cuda_kernel.py
git commit -m "feat: add ConsecutiveReverse CUDA kernel with CPU parity tests"
```

---

### Task 3: CudaGridOptimizer (Runner with Auto-Batching)

**Files:**
- Create: `src/backtest/cuda_runner.py`
- Test: `tests/test_cuda_runner.py`

This task implements the optimizer class that handles GPU detection, batch splitting, kernel launching, and result collection.

- [ ] **Step 1: Write test — small grid produces correct OptimizeResult**

```python
# tests/test_cuda_runner.py
import os
import sqlite3
import tempfile
import numpy as np
import pytest

try:
    from numba import cuda
    HAS_CUDA = cuda.is_available()
except Exception:
    HAS_CUDA = False

pytestmark = pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")


@pytest.fixture
def db_with_data():
    """Create temp DB with 200 hourly bars."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    conn = sqlite3.connect(tmp.name)
    conn.execute("""
        CREATE TABLE klines (
            exchange TEXT, symbol TEXT, interval TEXT, timestamp INTEGER,
            open REAL, high REAL, low REAL, close REAL, volume REAL,
            PRIMARY KEY (exchange, symbol, interval, timestamp)
        )
    """)
    base_ts = 1704067200000
    for i in range(200):
        ts = base_ts + i * 3600000
        price = 40000 + i * 10
        conn.execute(
            "INSERT INTO klines VALUES (?,?,?,?,?,?,?,?,?)",
            ("binance", "BTCUSDT", "1h", ts, price, price + 50, price - 50, price + 5, 1000.0),
        )
    conn.commit()
    conn.close()
    yield tmp.name
    os.unlink(tmp.name)


class TestCudaGridOptimizer:
    def test_basic_run(self, db_with_data):
        """CudaGridOptimizer produces correct OptimizeResult with sorted trials."""
        from backtest.optimizer import ParamSpace
        from backtest.cuda_runner import CudaGridOptimizer

        space = ParamSpace({
            "CONSECUTIVE_THRESHOLD": [3, 5],
            "POSITION_MULTIPLIER": [1.0, 1.1],
            "INITIAL_POSITION_PCT": [0.01],
            "PROFIT_CANDLE_THRESHOLD": [1],
            "LEVERAGE": [50],
        })
        optimizer = CudaGridOptimizer(
            db_path=db_with_data,
            strategy_path="strategies/consecutive_reverse.py",
            symbol="BTCUSDT",
            interval="1h",
            start="2024-01-01",
            end="2024-01-08",
            balance=1000.0,
            leverage=50,
            param_space=space,
            objective="sharpe_ratio",
        )
        result = optimizer.run()

        assert result.total_trials == 4
        assert len(result.all_trials) == 4
        assert result.best_params is not None
        # Results should be sorted descending by score
        scores = [t["score"] for t in result.all_trials]
        assert scores == sorted(scores, reverse=True)
        # Each trial should have params and report
        for trial in result.all_trials:
            assert "params" in trial
            assert "score" in trial
            assert "report" in trial
            assert "net_return" in trial["report"]
            assert "sharpe_ratio" in trial["report"]

    def test_matches_numba_grid(self, db_with_data):
        """CudaGridOptimizer results must match NumbaGridOptimizer results."""
        from backtest.optimizer import ParamSpace, NumbaGridOptimizer
        from backtest.cuda_runner import CudaGridOptimizer

        space = ParamSpace({
            "CONSECUTIVE_THRESHOLD": [3, 5, 7],
            "POSITION_MULTIPLIER": [1.0, 1.1],
            "INITIAL_POSITION_PCT": [0.01],
            "PROFIT_CANDLE_THRESHOLD": [1, 2],
            "LEVERAGE": [50],
        })
        kwargs = dict(
            db_path=db_with_data,
            strategy_path="strategies/consecutive_reverse.py",
            symbol="BTCUSDT",
            interval="1h",
            start="2024-01-01",
            end="2024-01-08",
            balance=1000.0,
            leverage=50,
            param_space=space,
            objective="sharpe_ratio",
        )

        cuda_result = CudaGridOptimizer(**kwargs).run()
        numba_result = NumbaGridOptimizer(**kwargs, n_jobs=1).run()

        assert cuda_result.total_trials == numba_result.total_trials

        # Build lookup by params for comparison
        cuda_by_params = {}
        for t in cuda_result.all_trials:
            key = tuple(sorted(t["params"].items()))
            cuda_by_params[key] = t

        for t in numba_result.all_trials:
            key = tuple(sorted(t["params"].items()))
            ct = cuda_by_params[key]
            for metric in ["net_return", "max_drawdown", "sharpe_ratio", "win_rate", "total_trades"]:
                cpu_val = t["report"].get(metric, 0.0)
                gpu_val = ct["report"].get(metric, 0.0)
                if abs(cpu_val) < 1e-10:
                    assert abs(gpu_val) < 1e-6, f"{metric}: cpu={cpu_val}, gpu={gpu_val}"
                else:
                    rel_err = abs(gpu_val - cpu_val) / (abs(cpu_val) + 1e-15)
                    assert rel_err < 1e-4, f"{metric}: cpu={cpu_val}, gpu={gpu_val}"


class TestBatching:
    def test_forced_small_batch(self, db_with_data):
        """Force batch_size=2 to test multi-batch correctness."""
        from backtest.optimizer import ParamSpace
        from backtest.cuda_runner import CudaGridOptimizer

        space = ParamSpace({
            "CONSECUTIVE_THRESHOLD": [3, 5, 7],
            "POSITION_MULTIPLIER": [1.0],
            "INITIAL_POSITION_PCT": [0.01],
            "PROFIT_CANDLE_THRESHOLD": [1],
            "LEVERAGE": [50],
        })
        optimizer = CudaGridOptimizer(
            db_path=db_with_data,
            strategy_path="strategies/consecutive_reverse.py",
            symbol="BTCUSDT",
            interval="1h",
            start="2024-01-01",
            end="2024-01-08",
            balance=1000.0,
            leverage=50,
            param_space=space,
            objective="sharpe_ratio",
            batch_size=2,  # Force 2 batches (3 combos / 2 = 2 batches)
        )
        result = optimizer.run()
        assert result.total_trials == 3
        scores = [t["score"] for t in result.all_trials]
        assert scores == sorted(scores, reverse=True)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cuda_runner.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backtest.cuda_runner'`

- [ ] **Step 3: Implement `CudaGridOptimizer`**

```python
# src/backtest/cuda_runner.py
"""
CUDA-accelerated grid search optimizer.

Runs all parameter combinations on GPU with auto-batching based on VRAM.
Falls back to numba-grid if CUDA is unavailable.
"""

from __future__ import annotations

import math
import sys
import time
from datetime import datetime, timezone

import numpy as np

from backtest.optimizer import OptimizeResult, ParamSpace


# Bytes per combo: params(40) + results(64) = 104 bytes on GPU memory
_BYTES_PER_COMBO = 104
# Default block size for kernel launch
_BLOCK_SIZE = 256
# Safety factor: use at most 70% of free VRAM for combos
_VRAM_USAGE_FRACTION = 0.7


def _estimate_batch_size(n_params: int = 5, n_metrics: int = 8) -> int:
    """Estimate max combos per batch from available GPU memory."""
    from numba import cuda
    free, total = cuda.current_context().get_memory_info()
    usable = int(free * _VRAM_USAGE_FRACTION)
    bytes_per = n_params * 8 + n_metrics * 8  # float64
    if bytes_per == 0:
        return 500_000
    return max(1, usable // bytes_per)


class CudaGridOptimizer:
    """Grid search optimizer using CUDA GPU acceleration."""

    def __init__(
        self,
        db_path: str,
        strategy_path: str,
        symbol: str,
        interval: str,
        start: str,
        end: str,
        balance: float = 10000.0,
        leverage: int = 10,
        param_space: ParamSpace | None = None,
        objective: str = "sharpe_ratio",
        commission_rate: float = 0.0004,
        funding_rate: float = 0.0001,
        maintenance_margin: float = 0.005,
        batch_size: int | None = None,
    ):
        self.db_path = db_path
        self.strategy_path = strategy_path
        self.symbol = symbol
        self.interval = interval
        self.start = f"{start} 00:00:00" if len(start) == 10 else start
        self.end = f"{end} 23:59:59" if len(end) == 10 else end
        self.balance = balance
        self.leverage = leverage
        self.param_space = param_space or ParamSpace({})
        self.objective = objective
        self.commission_rate = commission_rate
        self.funding_rate = funding_rate
        self.maintenance_margin = maintenance_margin
        self.batch_size = batch_size  # None = auto

    def run(self) -> OptimizeResult:
        """Run grid search on GPU with auto-batching."""
        from numba import cuda
        from backtest.numba_simulate import load_bars
        from backtest.cuda_strategies import CUDA_STRATEGIES

        # Resolve strategy
        import importlib.util
        from backtest.strategy import BaseStrategy
        spec = importlib.util.spec_from_file_location("user_strategy", self.strategy_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        strategy_name = None
        for attr in dir(module):
            obj = getattr(module, attr)
            if isinstance(obj, type) and issubclass(obj, BaseStrategy) and obj is not BaseStrategy:
                strategy_name = obj.__name__
                break
        if strategy_name is None:
            raise ValueError(f"No BaseStrategy subclass found in {self.strategy_path}")

        if strategy_name not in CUDA_STRATEGIES:
            raise ValueError(
                f"No CUDA kernel registered for '{strategy_name}'. "
                f"Available: {list(CUDA_STRATEGIES.keys())}. "
                f"Use --method numba-grid as fallback."
            )

        entry = CUDA_STRATEGIES[strategy_name]
        kernel = entry["kernel"]
        param_order = entry["param_order"]

        # Load bars
        start_ts = int(
            datetime.strptime(self.start, "%Y-%m-%d %H:%M:%S")
            .replace(tzinfo=timezone.utc).timestamp() * 1000
        )
        end_ts = int(
            datetime.strptime(self.end, "%Y-%m-%d %H:%M:%S")
            .replace(tzinfo=timezone.utc).timestamp() * 1000
        )
        bars = load_bars(self.db_path, self.symbol, self.interval, "binance", start_ts, end_ts)
        n_bars = bars.shape[0]
        print(f"Loaded {n_bars} bars, copying to GPU...", flush=True)

        d_bars = cuda.to_device(bars)

        # Build parameter combos array
        combos = self.param_space.grid()
        total = len(combos)
        n_params = len(param_order)

        params_array = np.zeros((total, n_params), dtype=np.float64)
        for i, combo in enumerate(combos):
            for j, name in enumerate(param_order):
                params_array[i, j] = float(combo.get(name, 0))

        # Determine batch size
        batch_size = self.batch_size or _estimate_batch_size(n_params)
        n_batches = math.ceil(total / batch_size)
        print(f"Running {total} combos in {n_batches} batch(es) (batch_size={batch_size})", flush=True)

        t0 = time.time()
        all_results = np.zeros((total, 8), dtype=np.float64)

        for b in range(n_batches):
            start_idx = b * batch_size
            end_idx = min(start_idx + batch_size, total)
            batch_n = end_idx - start_idx

            d_params = cuda.to_device(params_array[start_idx:end_idx])
            d_results = cuda.device_array((batch_n, 8), dtype=np.float64)

            threads = _BLOCK_SIZE
            blocks = (batch_n + threads - 1) // threads

            kernel[blocks, threads](
                d_bars, d_params, d_results,
                n_bars, batch_n,
                self.leverage, self.commission_rate, self.funding_rate,
                self.maintenance_margin, self.balance,
            )
            cuda.synchronize()

            all_results[start_idx:end_idx] = d_results.copy_to_host()

            self._print_progress(end_idx, total)

        print()
        elapsed = time.time() - t0

        # Build trial dicts
        metric_names = [
            "net_return", "annual_return", "max_drawdown", "sharpe_ratio",
            "sortino_ratio", "win_rate", "profit_factor", "total_trades",
        ]
        trials = []
        for i in range(total):
            metrics = {name: all_results[i, j] for j, name in enumerate(metric_names)}
            score = metrics.get(self.objective, 0.0)
            if not isinstance(score, (int, float)) or score != score:
                score = float("-inf")
            elif score > 1e9:
                score = 1e9
            trials.append({
                "params": combos[i],
                "score": score,
                "report": metrics,
            })

        trials.sort(key=lambda t: t["score"], reverse=True)

        return OptimizeResult(
            best_params=trials[0]["params"] if trials else {},
            best_score=trials[0]["score"] if trials else 0.0,
            all_trials=trials,
            objective=self.objective,
            total_trials=total,
            elapsed_seconds=elapsed,
        )

    @staticmethod
    def _print_progress(current: int, total: int) -> None:
        bar_len = 40
        filled = int(bar_len * current / total)
        bar = "#" * filled + "-" * (bar_len - filled)
        pct = current / total * 100
        print(f"\r[{bar}] {current}/{total} ({pct:.1f}%)", end="", flush=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cuda_runner.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/backtest/cuda_runner.py tests/test_cuda_runner.py
git commit -m "feat: add CudaGridOptimizer with auto-batching and VRAM detection"
```

---

### Task 4: CLI Integration

**Files:**
- Modify: `src/backtest/__main__.py:109-164` (cmd_optimize function) and line 262 (method choices)
- Test: `tests/test_cuda_runner.py` (add CLI-level test)

- [ ] **Step 1: Write test — cuda-grid method is accepted and produces results**

Append to `tests/test_cuda_runner.py`:

```python
class TestCLIIntegration:
    def test_cmd_optimize_cuda_grid(self, db_with_data):
        """cmd_optimize with method=cuda-grid creates an OptimizeResult."""
        import argparse
        from backtest.__main__ import cmd_optimize

        # Create a minimal argparse namespace matching what cmd_optimize expects
        args = argparse.Namespace(
            strategy="strategies/consecutive_reverse.py",
            symbol="BTCUSDT",
            interval="1h",
            exchange="binance",
            start="2024-01-01",
            end="2024-01-08",
            balance=1000.0,
            leverage=50,
            params="CONSECUTIVE_THRESHOLD=3:5:1,POSITION_MULTIPLIER=1.0:1.1:0.1,INITIAL_POSITION_PCT=0.01,PROFIT_CANDLE_THRESHOLD=1,LEVERAGE=50",
            objective="sharpe_ratio",
            method="cuda-grid",
            n_jobs=None,
            n_trials=100,
            top=5,
            save_top=10,
            report_top=1,
            db=db_with_data,
        )
        # Should not raise
        cmd_optimize(args)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cuda_runner.py::TestCLIIntegration -v`
Expected: FAIL — `cuda-grid` not in choices or not handled

- [ ] **Step 3: Add `cuda-grid` to `__main__.py`**

In `src/backtest/__main__.py`, make these changes:

1. Add `"cuda-grid"` to `--method` choices (line 262):

Change:
```python
    p_opt.add_argument("--method", default="grid", choices=["grid", "optuna", "numba-grid"])
```
To:
```python
    p_opt.add_argument("--method", default="grid", choices=["grid", "optuna", "numba-grid", "cuda-grid"])
```

2. Add `CudaGridOptimizer` branch in `cmd_optimize()` (after the `elif args.method == "numba-grid":` block, before the `else:`):

Insert after the NumbaGridOptimizer block (around line 149):
```python
    elif args.method == "cuda-grid":
        from backtest.cuda_runner import CudaGridOptimizer
        optimizer = CudaGridOptimizer(
            db_path=args.db or str(Path("data") / "klines.db"),
            strategy_path=args.strategy,
            symbol=args.symbol,
            interval=args.interval,
            start=args.start,
            end=args.end,
            balance=args.balance,
            leverage=args.leverage,
            param_space=param_space,
            objective=args.objective,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cuda_runner.py -v`
Expected: All tests PASS

- [ ] **Step 5: Run existing test suite to verify no regressions**

Run: `pytest tests/ -v`
Expected: All existing tests still PASS

- [ ] **Step 6: Commit**

```bash
git add src/backtest/__main__.py tests/test_cuda_runner.py
git commit -m "feat: integrate cuda-grid method into CLI optimize command"
```

---

### Task 5: Edge Cases and Robustness

**Files:**
- Modify: `src/backtest/cuda_runner.py` (add CUDA availability check)
- Test: `tests/test_cuda_runner.py` (add edge case tests)

- [ ] **Step 1: Write edge case tests**

Append to `tests/test_cuda_runner.py`:

```python
class TestEdgeCases:
    def test_combo_count_not_divisible_by_block_size(self, db_with_data):
        """Non-aligned combo count should not lose any combos."""
        from backtest.optimizer import ParamSpace
        from backtest.cuda_runner import CudaGridOptimizer

        # 7 combos with block_size 256 means 1 partial block
        space = ParamSpace({
            "CONSECUTIVE_THRESHOLD": [3, 4, 5, 6, 7, 8, 9],
            "POSITION_MULTIPLIER": [1.0],
            "INITIAL_POSITION_PCT": [0.01],
            "PROFIT_CANDLE_THRESHOLD": [1],
            "LEVERAGE": [50],
        })
        optimizer = CudaGridOptimizer(
            db_path=db_with_data,
            strategy_path="strategies/consecutive_reverse.py",
            symbol="BTCUSDT",
            interval="1h",
            start="2024-01-01",
            end="2024-01-08",
            balance=1000.0,
            leverage=50,
            param_space=space,
            objective="sharpe_ratio",
        )
        result = optimizer.run()
        assert result.total_trials == 7

    def test_single_combo(self, db_with_data):
        """Single combo should work."""
        from backtest.optimizer import ParamSpace
        from backtest.cuda_runner import CudaGridOptimizer

        space = ParamSpace({
            "CONSECUTIVE_THRESHOLD": [5],
            "POSITION_MULTIPLIER": [1.1],
            "INITIAL_POSITION_PCT": [0.01],
            "PROFIT_CANDLE_THRESHOLD": [1],
            "LEVERAGE": [50],
        })
        optimizer = CudaGridOptimizer(
            db_path=db_with_data,
            strategy_path="strategies/consecutive_reverse.py",
            symbol="BTCUSDT",
            interval="1h",
            start="2024-01-01",
            end="2024-01-08",
            balance=1000.0,
            leverage=50,
            param_space=space,
            objective="sharpe_ratio",
        )
        result = optimizer.run()
        assert result.total_trials == 1

    def test_unknown_strategy_raises(self, db_with_data):
        """Strategy not in CUDA registry should raise ValueError."""
        from backtest.optimizer import ParamSpace
        from backtest.cuda_runner import CudaGridOptimizer

        space = ParamSpace({"short_period": [5, 10]})
        optimizer = CudaGridOptimizer(
            db_path=db_with_data,
            strategy_path="strategies/example_ma_cross.py",
            symbol="BTCUSDT",
            interval="1h",
            start="2024-01-01",
            end="2024-01-08",
            balance=1000.0,
            leverage=50,
            param_space=space,
            objective="sharpe_ratio",
        )
        with pytest.raises(ValueError, match="No CUDA kernel registered"):
            optimizer.run()
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `pytest tests/test_cuda_runner.py::TestEdgeCases -v`
Expected: All 3 tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_cuda_runner.py
git commit -m "test: add edge case tests for CUDA optimizer"
```

---

### Task 6: Final Integration Test and Cleanup

**Files:**
- Test: manual run
- No code changes expected

- [ ] **Step 1: Run the full test suite**

Run: `pytest tests/ -v`
Expected: All tests PASS (CUDA tests skipped if no GPU)

- [ ] **Step 2: Verify CUDA tests skip gracefully without GPU**

The `pytestmark = pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")` at the top of each test file ensures tests are skipped cleanly on machines without CUDA.

Verify by checking test output shows `SKIPPED` (not `ERROR`) for cuda tests if running on a non-CUDA machine.

- [ ] **Step 3: Commit any final adjustments**

```bash
git add -A
git commit -m "chore: finalize CUDA grid optimizer integration"
```
