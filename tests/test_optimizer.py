import importlib.util
import json
import os
import sqlite3
import tempfile

import pytest
from backtest.optimizer import OptimizeResult, ParamSpace, _run_single_trial


class TestParamSpace:
    def test_int_range(self):
        space = ParamSpace({"X": (10, 30, 10)})
        assert space.grid() == [{"X": 10}, {"X": 20}, {"X": 30}]

    def test_float_range(self):
        space = ParamSpace({"Y": (1.0, 2.0, 0.5)})
        results = space.grid()
        assert len(results) == 3
        assert results[0] == {"Y": 1.0}
        assert results[1] == {"Y": 1.5}
        assert results[2] == {"Y": 2.0}

    def test_choice_list(self):
        space = ParamSpace({"Z": [1, 2, 3]})
        assert space.grid() == [{"Z": 1}, {"Z": 2}, {"Z": 3}]

    def test_cartesian_product(self):
        space = ParamSpace({"A": (1, 2, 1), "B": [10, 20]})
        combos = space.grid()
        assert len(combos) == 4
        assert {"A": 1, "B": 10} in combos
        assert {"A": 2, "B": 20} in combos

    def test_total_combinations(self):
        space = ParamSpace({"A": (1, 3, 1), "B": [10, 20]})
        assert space.total_combinations == 6

    def test_empty_space(self):
        space = ParamSpace({})
        assert space.grid() == [{}]
        assert space.total_combinations == 1


class TestRunSingleTrial:
    @pytest.fixture
    def db_with_data(self):
        """Create a temp DB with enough 1h bars for a simple strategy."""
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        conn = sqlite3.connect(tmp.name)
        conn.execute("""
            CREATE TABLE klines (
                exchange TEXT, symbol TEXT, interval TEXT, timestamp INTEGER,
                open REAL, high REAL, low REAL, close REAL, volume REAL,
                PRIMARY KEY (exchange, symbol, interval, timestamp)
            )
        """)
        # Insert 100 bars of 1h data starting 2024-01-01 00:00 UTC
        base_ts = 1704067200000  # 2024-01-01 00:00:00 UTC in ms
        for i in range(100):
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

    def test_run_single_trial_returns_dict(self, db_with_data):
        trial_args = {
            "db_path": db_with_data,
            "strategy_path": "strategies/example_ma_cross.py",
            "symbol": "BTCUSDT",
            "interval": "1h",
            "exchange": "binance",
            "start": "2024-01-01 00:00:00",
            "end": "2024-01-05 00:00:00",
            "balance": 10000.0,
            "leverage": 10,
            "params": {"short_period": 5, "long_period": 20},
        }
        result = _run_single_trial(trial_args)
        assert "params" in result
        assert "score" in result
        assert "report" in result
        assert result["params"] == {"short_period": 5, "long_period": 20}
        assert isinstance(result["score"], float)


class TestOptimizeResult:
    def test_fields(self):
        r = OptimizeResult(
            best_params={"X": 1},
            best_score=2.5,
            all_trials=[{"params": {"X": 1}, "score": 2.5, "report": {}}],
            objective="sharpe_ratio",
            total_trials=1,
            elapsed_seconds=1.0,
        )
        assert r.best_score == 2.5
        assert r.total_trials == 1


class TestGridSearchOptimizer:
    @pytest.fixture
    def db_with_data(self):
        """Create a temp DB with enough 1h bars."""
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

    def test_grid_search_runs_all_combinations(self, db_with_data):
        from backtest.optimizer import GridSearchOptimizer

        space = ParamSpace({"short_period": [5, 7], "long_period": [20, 25]})
        optimizer = GridSearchOptimizer(
            db_path=db_with_data,
            strategy_path="strategies/example_ma_cross.py",
            symbol="BTCUSDT",
            interval="1h",
            start="2024-01-01",
            end="2024-01-08",
            balance=10000,
            leverage=10,
            param_space=space,
            objective="sharpe_ratio",
            n_jobs=1,
        )
        result = optimizer.run()
        assert result.total_trials == 4
        assert len(result.all_trials) == 4
        assert result.best_params in [t["params"] for t in result.all_trials]
        assert result.all_trials[0]["score"] >= result.all_trials[-1]["score"]

    def test_grid_search_parallel(self, db_with_data):
        from backtest.optimizer import GridSearchOptimizer

        space = ParamSpace({"short_period": [5, 7], "long_period": [20, 25]})
        optimizer = GridSearchOptimizer(
            db_path=db_with_data,
            strategy_path="strategies/example_ma_cross.py",
            symbol="BTCUSDT",
            interval="1h",
            start="2024-01-01",
            end="2024-01-08",
            balance=10000,
            leverage=10,
            param_space=space,
            objective="sharpe_ratio",
            n_jobs=2,
        )
        result = optimizer.run()
        assert result.total_trials == 4


class TestOptunaOptimizer:
    @pytest.fixture
    def db_with_data(self):
        """Create a temp DB with enough 1h bars."""
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

    @pytest.mark.skipif(
        not importlib.util.find_spec("optuna"),
        reason="optuna not installed",
    )
    def test_optuna_optimizer_runs(self, db_with_data):
        from backtest.optimizer import OptunaOptimizer

        space = ParamSpace({"short_period": (5, 10, 1), "long_period": [20, 25, 30]})
        optimizer = OptunaOptimizer(
            db_path=db_with_data,
            strategy_path="strategies/example_ma_cross.py",
            symbol="BTCUSDT",
            interval="1h",
            start="2024-01-01",
            end="2024-01-08",
            balance=10000,
            leverage=10,
            param_space=space,
            objective="sharpe_ratio",
            n_trials=6,
            n_jobs=1,
        )
        result = optimizer.run()
        assert result.total_trials == 6
        assert result.best_params is not None
        assert "short_period" in result.best_params


class TestSaveResults:
    def test_save_and_query(self):
        from backtest.optimizer import save_results

        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        db_path = tmp.name

        result = OptimizeResult(
            best_params={"X": 10, "Y": 2.5},
            best_score=1.85,
            all_trials=[
                {"params": {"X": 10, "Y": 2.5}, "score": 1.85, "report": {"net_return": 0.5}},
                {"params": {"X": 20, "Y": 3.0}, "score": 1.20, "report": {"net_return": 0.3}},
            ],
            objective="sharpe_ratio",
            total_trials=2,
            elapsed_seconds=5.0,
        )

        save_results(
            db_path=db_path,
            strategy="TestStrategy",
            symbol="BTCUSDT",
            interval="1h",
            start_date="2024-01-01",
            end_date="2024-12-31",
            result=result,
        )

        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            "SELECT strategy, score, params_json FROM optimize_results ORDER BY score DESC"
        ).fetchall()
        conn.close()
        os.unlink(db_path)

        assert len(rows) == 2
        assert rows[0][0] == "TestStrategy"
        assert rows[0][1] == 1.85
        assert json.loads(rows[0][2]) == {"X": 10, "Y": 2.5}
