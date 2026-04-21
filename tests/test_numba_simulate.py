"""Validation tests: Numba simulate vs original BacktestEngine."""

import importlib.util
import math

import numpy as np
import pytest
from datetime import datetime, timezone

from backtest.numba_simulate import simulate, load_bars


DB_PATH = "data/klines.db"
STRATEGY_PATH = "strategies/consecutive_reverse.py"


def _load_strategy_class():
    spec = importlib.util.spec_from_file_location("strat", STRATEGY_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ConsecutiveReverseStrategy


def _run_original(start: str, end: str, **params):
    from backtest.engine import BacktestEngine
    from backtest.reporter import Reporter

    StratClass = _load_strategy_class()
    # Override class attributes with params
    attrs = {
        "CONSECUTIVE_THRESHOLD": params.get("threshold", 5),
        "POSITION_MULTIPLIER": params.get("multiplier", 1.1),
        "INITIAL_POSITION_PCT": params.get("initial_pct", 0.01),
        "PROFIT_CANDLE_THRESHOLD": params.get("profit_threshold", 1),
        "LEVERAGE": params.get("leverage", 50),
    }
    TrialClass = type("Trial", (StratClass,), attrs)

    engine = BacktestEngine(
        db_path=DB_PATH, symbol="BTCUSDT", interval="1h", exchange="binance",
        strategy_class=TrialClass, balance=1000.0, leverage=50,
        commission_rate=0.0004, funding_rate=0.0001, maintenance_margin=0.005,
        start=start, end=end,
    )
    result = engine.run()
    return Reporter.generate(result)


def _run_numba(start: str, end: str, **params):
    start_ts = int(datetime.strptime(start, "%Y-%m-%d %H:%M:%S").replace(
        tzinfo=timezone.utc).timestamp() * 1000)
    end_ts = int(datetime.strptime(end, "%Y-%m-%d %H:%M:%S").replace(
        tzinfo=timezone.utc).timestamp() * 1000)

    bars = load_bars(DB_PATH, "BTCUSDT", "1h", "binance", start_ts, end_ts)
    # Strategy.LEVERAGE is for sizing; engine leverage (50) is for margin
    sizing_lev = params.get("leverage", 50)
    result = simulate(
        bars,
        threshold=params.get("threshold", 5),
        multiplier=params.get("multiplier", 1.1),
        initial_pct=params.get("initial_pct", 0.01),
        profit_threshold=params.get("profit_threshold", 1),
        sizing_leverage=sizing_lev,
        exchange_leverage=50,  # engine always uses 50
        commission_rate=0.0004,
        funding_rate=0.0001,
        maintenance_margin=0.005,
        initial_balance=1000.0,
    )
    return {
        "net_return": result[0],
        "annual_return": result[1],
        "max_drawdown": result[2],
        "sharpe_ratio": result[3],
        "sortino_ratio": result[4],
        "win_rate": result[5],
        "profit_factor": result[6],
        "total_trades": result[7],
    }


class TestNumbaMatchesEngine:
    """Verify numba simulate produces identical results to the original engine."""

    @pytest.mark.parametrize("start,end", [
        ("2024-01-01 00:00:00", "2024-03-01 00:00:00"),
        ("2024-06-01 00:00:00", "2024-09-01 00:00:00"),
        ("2023-01-01 00:00:00", "2023-06-01 00:00:00"),
    ])
    def test_default_params_various_periods(self, start, end):
        orig = _run_original(start, end)
        numba = _run_numba(start, end)

        assert numba["total_trades"] == orig["total_trades"]
        assert abs(numba["net_return"] - orig["net_return"]) < 1e-8
        assert abs(numba["max_drawdown"] - orig["max_drawdown"]) < 1e-8
        assert abs(numba["win_rate"] - orig["win_rate"]) < 1e-8
        assert abs(numba["sharpe_ratio"] - orig["sharpe_ratio"]) < 1e-6
        assert abs(numba["profit_factor"] - orig["profit_factor"]) < 1e-6

    @pytest.mark.parametrize("params", [
        {"threshold": 3, "multiplier": 1.0, "initial_pct": 0.02, "profit_threshold": 2, "leverage": 20},
        {"threshold": 8, "multiplier": 1.5, "initial_pct": 0.005, "profit_threshold": 5, "leverage": 100},
        {"threshold": 4, "multiplier": 1.3, "initial_pct": 0.03, "profit_threshold": 1, "leverage": 150},
    ])
    def test_various_params(self, params):
        start, end = "2024-01-01 00:00:00", "2024-06-01 00:00:00"
        orig = _run_original(start, end, **params)
        numba = _run_numba(start, end, **params)

        assert numba["total_trades"] == orig["total_trades"]
        assert abs(numba["net_return"] - orig["net_return"]) < 1e-8
        assert abs(numba["max_drawdown"] - orig["max_drawdown"]) < 1e-8
        assert abs(numba["win_rate"] - orig["win_rate"]) < 1e-8

    def test_empty_bars(self):
        bars = np.empty((0, 6), dtype=np.float64)
        result = simulate(bars, 5, 1.1, 0.01, 1, 50, 50, 0.0004, 0.0001, 0.005, 1000.0)
        assert result[0] == 0.0  # net_return
        assert result[7] == 0    # total_trades

    def test_all_doji(self):
        """All bars have open == close (doji), no trades should occur."""
        bars = np.zeros((100, 6), dtype=np.float64)
        for i in range(100):
            bars[i, 0] = 1704067200000 + i * 3600000
            bars[i, 1] = 40000.0  # open
            bars[i, 4] = 40000.0  # close == open (doji)
            bars[i, 2] = 40100.0
            bars[i, 3] = 39900.0
            bars[i, 5] = 1000.0
        result = simulate(bars, 5, 1.1, 0.01, 1, 50, 50, 0.0004, 0.0001, 0.005, 1000.0)
        assert result[7] == 0  # no trades

    def test_high_leverage_liquidation(self):
        """High leverage with adverse move should trigger liquidation."""
        # 5 up candles then sharp reversal
        bars = np.zeros((20, 6), dtype=np.float64)
        for i in range(20):
            bars[i, 0] = 1704067200000 + i * 3600000
            bars[i, 1] = 40000.0 + i * 100
            if i < 7:
                bars[i, 4] = bars[i, 1] + 50  # up (triggers short at threshold)
            else:
                bars[i, 4] = bars[i, 1] + 2000  # massive up (hurts short position)
            bars[i, 2] = max(bars[i, 1], bars[i, 4]) + 50
            bars[i, 3] = min(bars[i, 1], bars[i, 4]) - 50
            bars[i, 5] = 1000.0

        result = simulate(bars, 5, 1.5, 0.05, 3, 200, 50, 0.0004, 0.0001, 0.005, 1000.0)
        # Should have some trades and potentially liquidation
        assert result[7] > 0
        # High leverage + adverse move = likely loss
        assert result[0] < 0  # net_return negative


class TestNumbaBenchmark:
    """Performance benchmarks."""

    def test_speedup_over_python(self):
        """Numba should be at least 20x faster than original engine."""
        import time

        start = "2024-01-01 00:00:00"
        end = "2024-07-01 00:00:00"

        # Original
        t0 = time.perf_counter()
        _run_original(start, end)
        t_orig = time.perf_counter() - t0

        # Numba (warm)
        start_ts = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        end_ts = int(datetime(2024, 7, 1, tzinfo=timezone.utc).timestamp() * 1000)
        bars = load_bars(DB_PATH, "BTCUSDT", "1h", "binance", start_ts, end_ts)
        # Warm up
        simulate(bars, 5, 1.1, 0.01, 1, 50, 50, 0.0004, 0.0001, 0.005, 1000.0)

        t0 = time.perf_counter()
        for _ in range(100):
            simulate(bars, 5, 1.1, 0.01, 1, 50, 50, 0.0004, 0.0001, 0.005, 1000.0)
        t_numba = (time.perf_counter() - t0) / 100

        speedup = t_orig / t_numba
        print(f"\nOriginal: {t_orig*1000:.1f}ms, Numba: {t_numba*1000:.3f}ms, Speedup: {speedup:.0f}x")
        assert speedup > 20
