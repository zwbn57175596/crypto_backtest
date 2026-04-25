"""Tests for CUDA grid search optimizer (CudaGridOptimizer).

Tests verify that GPU-accelerated grid search produces correct results and
handles batching properly. Tests skip on non-CUDA platforms (e.g., macOS).
"""

import os
import sqlite3
import sys
import tempfile

import pytest

# Skip all tests on macOS (no CUDA support)
pytestmark = pytest.mark.skipif(
    sys.platform == "darwin", reason="CUDA not available on macOS"
)

try:
    from backtest.cuda_runner import CudaGridOptimizer
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False


@pytest.fixture
def db_with_data():
    """Create a temp DB with 200 hourly bars for testing."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    conn = sqlite3.connect(tmp.name)
    conn.execute(
        """
        CREATE TABLE klines (
            exchange TEXT, symbol TEXT, interval TEXT, timestamp INTEGER,
            open REAL, high REAL, low REAL, close REAL, volume REAL,
            PRIMARY KEY (exchange, symbol, interval, timestamp)
        )
    """
    )
    # Insert 200 bars of 1h data starting 2024-01-01 00:00 UTC
    base_ts = 1704067200000  # 2024-01-01 00:00:00 UTC in ms
    for i in range(200):
        ts = base_ts + i * 3600000
        price = 40000 + i * 10
        conn.execute(
            "INSERT INTO klines VALUES (?,?,?,?,?,?,?,?,?)",
            (
                "binance",
                "BTCUSDT",
                "1h",
                ts,
                price,
                price + 50,
                price - 50,
                price + 5,
                1000.0,
            ),
        )
    conn.commit()
    conn.close()
    yield tmp.name
    os.unlink(tmp.name)


@pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
class TestCudaGridOptimizer:
    def test_basic_run(self, db_with_data):
        """Test basic GPU grid search with 4 parameter combinations."""
        from backtest.optimizer import ParamSpace

        param_space = ParamSpace(
            {
                "CONSECUTIVE_THRESHOLD": [3, 5],
                "POSITION_MULTIPLIER": [1.1, 1.2],
                "INITIAL_POSITION_PCT": 0.01,
                "PROFIT_CANDLE_THRESHOLD": 1,
                "LEVERAGE": 10,
            }
        )

        optimizer = CudaGridOptimizer(
            db_path=db_with_data,
            strategy_path="strategies/consecutive_reverse.py",
            symbol="BTCUSDT",
            interval="1h",
            start="2024-01-01",
            end="2024-01-09",
            balance=10000.0,
            leverage=10,
            param_space=param_space,
            objective="sharpe_ratio",
        )

        result = optimizer.run()

        # Verify OptimizeResult structure
        assert result.total_trials == 4
        assert result.best_params is not None
        assert result.best_score >= float("-inf")
        assert len(result.all_trials) == 4

        # Verify all trials have correct structure
        for trial in result.all_trials:
            assert "params" in trial
            assert "score" in trial
            assert "report" in trial
            assert "CONSECUTIVE_THRESHOLD" in trial["params"]
            assert "net_return" in trial["report"]
            assert "annual_return" in trial["report"]
            assert "max_drawdown" in trial["report"]
            assert "sharpe_ratio" in trial["report"]
            assert "sortino_ratio" in trial["report"]
            assert "win_rate" in trial["report"]
            assert "profit_factor" in trial["report"]
            assert "total_trades" in trial["report"]

        # Verify results are sorted by score (descending)
        scores = [t["score"] for t in result.all_trials]
        assert scores == sorted(scores, reverse=True)

    def test_forced_small_batch(self, db_with_data):
        """Test multi-batch processing with batch_size=2 and 3 combos."""
        from backtest.optimizer import ParamSpace

        param_space = ParamSpace(
            {
                "CONSECUTIVE_THRESHOLD": [3, 5, 7],
                "POSITION_MULTIPLIER": 1.1,
                "INITIAL_POSITION_PCT": 0.01,
                "PROFIT_CANDLE_THRESHOLD": 1,
                "LEVERAGE": 10,
            }
        )

        optimizer = CudaGridOptimizer(
            db_path=db_with_data,
            strategy_path="strategies/consecutive_reverse.py",
            symbol="BTCUSDT",
            interval="1h",
            start="2024-01-01",
            end="2024-01-09",
            balance=10000.0,
            leverage=10,
            param_space=param_space,
            objective="sharpe_ratio",
            batch_size=2,  # Force small batches: 3 combos = 2 batches
        )

        result = optimizer.run()

        # Verify correct number of trials
        assert result.total_trials == 3
        assert len(result.all_trials) == 3

        # Verify results are valid
        for trial in result.all_trials:
            assert isinstance(trial["score"], float)
            assert "CONSECUTIVE_THRESHOLD" in trial["params"]

    def test_matches_numba_grid(self, db_with_data):
        """Verify CUDA results match NumbaGridOptimizer (n_jobs=1) within tolerance.

        This test compares GPU results against CPU results to ensure correctness.
        """
        from backtest.optimizer import NumbaGridOptimizer, ParamSpace

        param_space = ParamSpace(
            {
                "CONSECUTIVE_THRESHOLD": [3, 5],
                "POSITION_MULTIPLIER": 1.1,
                "INITIAL_POSITION_PCT": 0.01,
                "PROFIT_CANDLE_THRESHOLD": 1,
                "LEVERAGE": 10,
            }
        )

        # Run CUDA optimizer
        cuda_opt = CudaGridOptimizer(
            db_path=db_with_data,
            strategy_path="strategies/consecutive_reverse.py",
            symbol="BTCUSDT",
            interval="1h",
            start="2024-01-01",
            end="2024-01-09",
            balance=10000.0,
            leverage=10,
            param_space=param_space,
            objective="sharpe_ratio",
        )
        cuda_result = cuda_opt.run()

        # Run Numba optimizer (CPU)
        numba_opt = NumbaGridOptimizer(
            db_path=db_with_data,
            strategy_path="strategies/consecutive_reverse.py",
            symbol="BTCUSDT",
            interval="1h",
            start="2024-01-01",
            end="2024-01-09",
            balance=10000.0,
            leverage=10,
            param_space=param_space,
            objective="sharpe_ratio",
            n_jobs=1,
        )
        numba_result = numba_opt.run()

        # Compare results
        assert len(cuda_result.all_trials) == len(numba_result.all_trials)

        # Sort both by params for comparison
        cuda_sorted = sorted(cuda_result.all_trials, key=lambda t: str(t["params"]))
        numba_sorted = sorted(numba_result.all_trials, key=lambda t: str(t["params"]))

        # Compare metrics within 1e-4 relative error
        tol = 1e-4
        for cuda_trial, numba_trial in zip(cuda_sorted, numba_sorted):
            assert cuda_trial["params"] == numba_trial["params"]

            cuda_score = cuda_trial["score"]
            numba_score = numba_trial["score"]

            # Handle special values
            if cuda_score == float("-inf") and numba_score == float("-inf"):
                continue
            if cuda_score != cuda_score or numba_score != numba_score:  # NaN
                continue

            # Relative error check
            if abs(numba_score) > 1e-6:
                rel_error = abs(cuda_score - numba_score) / abs(numba_score)
                assert rel_error < tol, (
                    f"Score mismatch for {cuda_trial['params']}: "
                    f"CUDA={cuda_score}, Numba={numba_score}, rel_error={rel_error}"
                )

            # Also check metrics
            for metric_key in [
                "net_return",
                "annual_return",
                "max_drawdown",
                "sharpe_ratio",
                "sortino_ratio",
                "win_rate",
                "profit_factor",
                "total_trades",
            ]:
                cuda_val = cuda_trial["report"][metric_key]
                numba_val = numba_trial["report"][metric_key]

                if cuda_val == float("-inf") and numba_val == float("-inf"):
                    continue
                if cuda_val != cuda_val or numba_val != numba_val:  # NaN
                    continue

                if abs(numba_val) > 1e-6:
                    rel_error = abs(cuda_val - numba_val) / abs(numba_val)
                    assert rel_error < tol, (
                        f"Metric {metric_key} mismatch for {cuda_trial['params']}: "
                        f"CUDA={cuda_val}, Numba={numba_val}, rel_error={rel_error}"
                    )
