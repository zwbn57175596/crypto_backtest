import importlib.util
import json
import math
import os
import sqlite3
import tempfile
import time

import pytest
from backtest.optimizer import OptimizeResult, ParamSpace, _run_single_trial, parse_params_string, GridSearchOptimizer, save_results


class TestParseParamsString:
    def test_range_params(self):
        space = parse_params_string("DECISION_LEN=20:80:10")
        combos = space.grid()
        assert combos[0] == {"DECISION_LEN": 20}
        assert combos[-1] == {"DECISION_LEN": 80}

    def test_choice_params(self):
        space = parse_params_string("TOLERANCE_RATE=0.005|0.00618|0.008")
        combos = space.grid()
        assert len(combos) == 3
        assert combos[0] == {"TOLERANCE_RATE": 0.005}

    def test_multiple_params(self):
        space = parse_params_string("X=1:3:1,Y=10|20")
        assert space.total_combinations == 6

    def test_float_range(self):
        space = parse_params_string("SF=1.5:3.0:0.5")
        combos = space.grid()
        assert len(combos) == 4
        assert combos[0] == {"SF": 1.5}


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


class TestShadowPowerOptimize:
    """Integration test: optimize Shadow Power strategy params on synthetic 15m data."""

    @pytest.fixture
    def db_with_15m_data(self):
        """Create DB with 15m bars spanning multiple 4H periods."""
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        conn = sqlite3.connect(tmp.name)
        conn.execute("""
            CREATE TABLE klines (
                exchange TEXT, symbol TEXT, interval TEXT, timestamp INTEGER,
                open REAL, high REAL, low REAL, close REAL, volume REAL,
                PRIMARY KEY (exchange, symbol, interval, timestamp)
            )
        """)
        # 2000 bars of 15m = ~20 days, enough for DECISION_LEN=10 on 4H
        base_ts = 1704067200000  # 2024-01-01 00:00 UTC
        for i in range(2000):
            ts = base_ts + i * 900000  # 15min = 900000ms
            # Synthetic price with some wave pattern
            price = 40000 + 2000 * math.sin(i / 50.0) + i * 0.5
            high = price + 100 + 50 * abs(math.sin(i / 7.0))
            low = price - 100 - 50 * abs(math.sin(i / 11.0))
            vol = 1000 + 500 * abs(math.sin(i / 30.0))
            conn.execute(
                "INSERT INTO klines VALUES (?,?,?,?,?,?,?,?,?)",
                ("binance", "BTCUSDT", "15m", ts, price, high, low, price + 10, vol),
            )
        conn.commit()
        conn.close()
        yield tmp.name
        os.unlink(tmp.name)

    def test_optimize_shadow_power_params(self, db_with_15m_data):
        space = ParamSpace({
            "DECISION_LEN": [10, 20],
            "SHADOW_FACTOR": [2.0, 3.0],
        })
        optimizer = GridSearchOptimizer(
            db_path=db_with_15m_data,
            strategy_path="strategies/shadow_power_backtest.py",
            symbol="BTCUSDT",
            interval="15m",
            start="2024-01-01",
            end="2024-01-20",
            balance=1000,
            leverage=49,
            param_space=space,
            objective="sharpe_ratio",
            n_jobs=1,
        )
        result = optimizer.run()
        assert result.total_trials == 4
        assert result.best_params is not None
        assert "DECISION_LEN" in result.best_params
        assert "SHADOW_FACTOR" in result.best_params


class TestTopNAutoSave:
    @pytest.fixture
    def db_with_data(self):
        """Create a temp DB with 1h bars."""
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

    def test_save_top_reports(self, db_with_data):
        from backtest.optimizer import GridSearchOptimizer, ParamSpace, save_top_reports

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

        report_db = db_with_data.replace(".db", "_reports.db")
        save_top_reports(
            result=result,
            top_n=2,
            db_path=db_with_data,
            report_db_path=report_db,
            strategy_path="strategies/example_ma_cross.py",
            symbol="BTCUSDT",
            interval="1h",
            start="2024-01-01",
            end="2024-01-08",
            balance=10000,
            leverage=10,
        )

        conn = sqlite3.connect(report_db)
        rows = conn.execute("SELECT strategy, report_json FROM reports").fetchall()
        conn.close()
        os.unlink(report_db)

        assert len(rows) == 2
        assert "_opt1" in rows[0][0]
        report = json.loads(rows[0][1])
        assert "equity_curve" in report
        assert "trades" in report


def test_save_top_reports_links_optimize_result_id(tmp_path):
    """reports.optimize_result_id must point to the matching optimize_results row."""
    import json, sqlite3
    from backtest.optimizer import OptimizeResult, save_results, save_top_reports

    report_db = str(tmp_path / "reports.db")
    klines_db = str(tmp_path / "klines.db")  # unused by save_top_reports directly

    # Build a minimal OptimizeResult with two trials
    trials = [
        {"params": {"CONSECUTIVE_THRESHOLD": 3}, "score": 2.0,
         "report": {"sharpe_ratio": 2.0, "net_return": 0.5, "max_drawdown": 0.1}},
        {"params": {"CONSECUTIVE_THRESHOLD": 5}, "score": 1.5,
         "report": {"sharpe_ratio": 1.5, "net_return": 0.3, "max_drawdown": 0.2}},
    ]
    result = OptimizeResult(
        best_params=trials[0]["params"],
        best_score=2.0,
        all_trials=trials,
        objective="sharpe_ratio",
        total_trials=2,
        elapsed_seconds=1.0,
    )

    # save_results writes to optimize_results table
    save_results(
        db_path=report_db,
        strategy="TestStrategy",
        symbol="BTCUSDT",
        interval="1h",
        start_date="2024-01-01",
        end_date="2024-12-31",
        result=result,
    )

    # Verify the optimize_results rows exist with expected IDs
    conn = sqlite3.connect(report_db)
    rows = conn.execute(
        "SELECT id, params_json FROM optimize_results ORDER BY score DESC"
    ).fetchall()
    conn.close()

    assert len(rows) == 2
    top_id = rows[0][0]  # highest score row
    top_params = json.loads(rows[0][1])
    assert top_params == {"CONSECUTIVE_THRESHOLD": 3}

    # Now test the linkage write helper directly
    from backtest.optimizer import _write_report_with_link
    full_report = {"sharpe_ratio": 2.0, "net_return": 0.5, "equity_curve": [], "trades": []}
    _write_report_with_link(
        report_db_path=report_db,
        strategy_name="TestStrategy_opt1",
        symbol="BTCUSDT",
        interval="1h",
        created_at="2026-01-01T00:00:00+00:00",
        report=full_report,
        params={"CONSECUTIVE_THRESHOLD": 3},
        base_strategy="TestStrategy",
    )

    conn = sqlite3.connect(report_db)
    row = conn.execute(
        "SELECT optimize_result_id FROM reports WHERE strategy = 'TestStrategy_opt1'"
    ).fetchone()
    conn.close()

    assert row is not None
    assert row[0] == top_id


class TestBatchId:
    def test_save_results_generates_batch_id(self, tmp_path):
        """save_results should auto-generate batch_id for all trials in one call."""
        db = str(tmp_path / "test.db")
        result = OptimizeResult(
            best_params={"x": 1},
            best_score=1.5,
            all_trials=[
                {"params": {"x": 1}, "score": 1.5, "report": {"net_return": 0.1}},
                {"params": {"x": 2}, "score": 1.0, "report": {"net_return": 0.05}},
            ],
            objective="sharpe_ratio",
            total_trials=2,
            elapsed_seconds=1.0,
        )
        save_results(db, "MaCross", "BTCUSDT", "1h", "2026-01-01", "2026-06-30", result)

        conn = sqlite3.connect(db)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT batch_id FROM optimize_results").fetchall()
        conn.close()

        assert len(rows) == 2
        assert rows[0]["batch_id"] == rows[1]["batch_id"]
        batch_id = rows[0]["batch_id"]
        assert "MaCross" in batch_id
        assert "BTCUSDT" in batch_id

    def test_save_results_different_calls_different_batch_ids(self, tmp_path):
        """Two separate save_results calls should produce different batch_ids."""
        db = str(tmp_path / "test.db")
        result = OptimizeResult(
            best_params={"x": 1},
            best_score=1.5,
            all_trials=[{"params": {"x": 1}, "score": 1.5, "report": {"net_return": 0.1}}],
            objective="sharpe_ratio",
            total_trials=1,
            elapsed_seconds=1.0,
        )
        save_results(db, "MaCross", "BTCUSDT", "1h", "2026-01-01", "2026-06-30", result)
        time.sleep(1.1)  # ensure different second-level timestamp
        save_results(db, "MaCross", "BTCUSDT", "1h", "2026-01-01", "2026-06-30", result)

        conn = sqlite3.connect(db)
        rows = conn.execute("SELECT DISTINCT batch_id FROM optimize_results").fetchall()
        conn.close()

        assert len(rows) == 2

    def test_batch_id_migration_for_old_data(self, tmp_path):
        """Old rows without batch_id should get backfilled."""
        db = str(tmp_path / "test.db")
        conn = sqlite3.connect(db)
        conn.execute("""
            CREATE TABLE optimize_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy TEXT NOT NULL,
                symbol TEXT NOT NULL,
                interval TEXT NOT NULL,
                start_date TEXT NOT NULL,
                end_date TEXT NOT NULL,
                objective TEXT NOT NULL,
                score REAL NOT NULL,
                params_json TEXT NOT NULL,
                report_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute(
            "INSERT INTO optimize_results (strategy,symbol,interval,start_date,end_date,objective,score,params_json,report_json,created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            ("MaCross", "BTCUSDT", "1h", "2026-01-01", "2026-06-30", "sharpe_ratio", 1.5, '{"x":1}', '{"net_return":0.1}', "2026-04-20T09:30:00+00:00"),
        )
        conn.commit()
        conn.close()

        result = OptimizeResult(
            best_params={"x": 2},
            best_score=2.0,
            all_trials=[{"params": {"x": 2}, "score": 2.0, "report": {"net_return": 0.2}}],
            objective="sharpe_ratio",
            total_trials=1,
            elapsed_seconds=1.0,
        )
        save_results(db, "MaCross", "BTCUSDT", "1h", "2026-01-01", "2026-06-30", result)

        conn = sqlite3.connect(db)
        rows = conn.execute("SELECT batch_id FROM optimize_results WHERE id = 1").fetchall()
        conn.close()

        assert rows[0][0] is not None
        assert "MaCross" in rows[0][0]
