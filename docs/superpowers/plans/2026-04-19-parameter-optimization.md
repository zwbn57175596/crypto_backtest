# Parameter Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add parameter optimization (grid search + Optuna) to the backtesting framework so users can find optimal strategy parameters via CLI.

**Architecture:** Independent `optimizer.py` module with `ParamSpace` for defining search ranges, `GridSearchOptimizer` and `OptunaOptimizer` sharing a common interface, multiprocessing parallelism, and results persisted to SQLite. CLI gains an `optimize` subcommand.

**Tech Stack:** Python stdlib (multiprocessing, itertools, sqlite3, json), optuna (optional)

---

## File Structure

| File | Responsibility |
|------|---------------|
| `src/backtest/optimizer.py` | ParamSpace, OptimizeResult, GridSearchOptimizer, OptunaOptimizer, _run_single_trial worker |
| `src/backtest/__main__.py` | Add `cmd_optimize()` + `optimize` subparser |
| `tests/test_optimizer.py` | All optimizer tests |

---

### Task 1: ParamSpace

**Files:**
- Create: `src/backtest/optimizer.py`
- Create: `tests/test_optimizer.py`

- [ ] **Step 1: Write failing tests for ParamSpace**

```python
# tests/test_optimizer.py
import pytest
from backtest.optimizer import ParamSpace


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_optimizer.py -v`
Expected: FAIL with "ModuleNotFoundError" or "ImportError"

- [ ] **Step 3: Implement ParamSpace**

```python
# src/backtest/optimizer.py
from __future__ import annotations

import itertools
from dataclasses import dataclass, field


class ParamSpace:
    """Defines a parameter search space for strategy optimization."""

    def __init__(self, space: dict):
        self._space = space
        self._axes: dict[str, list] = {}
        for name, spec in space.items():
            if isinstance(spec, list):
                self._axes[name] = spec
            elif isinstance(spec, tuple) and len(spec) == 3:
                min_val, max_val, step = spec
                if isinstance(min_val, int) and isinstance(max_val, int) and isinstance(step, int):
                    self._axes[name] = list(range(min_val, max_val + 1, step))
                else:
                    values = []
                    v = float(min_val)
                    while v <= float(max_val) + float(step) * 0.01:
                        values.append(round(v, 10))
                        v += float(step)
                    self._axes[name] = values
            else:
                raise ValueError(f"Invalid param spec for '{name}': {spec}")

    @property
    def total_combinations(self) -> int:
        if not self._axes:
            return 1
        result = 1
        for values in self._axes.values():
            result *= len(values)
        return result

    def grid(self) -> list[dict]:
        if not self._axes:
            return [{}]
        names = list(self._axes.keys())
        value_lists = [self._axes[n] for n in names]
        return [dict(zip(names, combo)) for combo in itertools.product(*value_lists)]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_optimizer.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/backtest/optimizer.py tests/test_optimizer.py
git commit -m "feat(optimizer): add ParamSpace with grid generation"
```

---

### Task 2: OptimizeResult and _run_single_trial

**Files:**
- Modify: `src/backtest/optimizer.py`
- Modify: `tests/test_optimizer.py`

- [ ] **Step 1: Write failing tests for _run_single_trial**

```python
# tests/test_optimizer.py (append to file)
import os
import sqlite3
import tempfile
from backtest.optimizer import OptimizeResult, _run_single_trial


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_optimizer.py::TestRunSingleTrial -v`
Expected: FAIL with "ImportError" (OptimizeResult and _run_single_trial not defined)

- [ ] **Step 3: Implement OptimizeResult and _run_single_trial**

Add to `src/backtest/optimizer.py`:

```python
import importlib.util
import sys
from pathlib import Path


@dataclass
class OptimizeResult:
    best_params: dict
    best_score: float
    all_trials: list[dict]
    objective: str
    total_trials: int
    elapsed_seconds: float


def _load_strategy_class(path: str):
    """Load BaseStrategy subclass from file path."""
    from backtest.strategy import BaseStrategy

    spec = importlib.util.spec_from_file_location("user_strategy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for attr in dir(module):
        obj = getattr(module, attr)
        if isinstance(obj, type) and issubclass(obj, BaseStrategy) and obj is not BaseStrategy:
            return obj
    raise ValueError(f"No BaseStrategy subclass found in {path}")


def _make_strategy(base_class: type, params: dict) -> type:
    """Create strategy subclass with overridden class attributes."""
    return type(f"{base_class.__name__}_trial", (base_class,), params)


def _run_single_trial(args: dict) -> dict:
    """Worker function for multiprocessing. Must be top-level for pickling."""
    from backtest.engine import BacktestEngine
    from backtest.reporter import Reporter

    strategy_class = _load_strategy_class(args["strategy_path"])
    trial_class = _make_strategy(strategy_class, args["params"])

    engine = BacktestEngine(
        db_path=args["db_path"],
        symbol=args["symbol"],
        interval=args["interval"],
        exchange=args["exchange"],
        strategy_class=trial_class,
        balance=args["balance"],
        leverage=args["leverage"],
        start=args["start"],
        end=args["end"],
    )

    result = engine.run()
    report = Reporter.generate(result)

    objective = args.get("objective", "sharpe_ratio")
    score = report.get(objective, 0.0)

    return {
        "params": args["params"],
        "score": score,
        "report": {k: v for k, v in report.items() if k not in ("equity_curve", "trades")},
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_optimizer.py::TestRunSingleTrial tests/test_optimizer.py::TestOptimizeResult -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/backtest/optimizer.py tests/test_optimizer.py
git commit -m "feat(optimizer): add OptimizeResult and _run_single_trial worker"
```

---

### Task 3: GridSearchOptimizer

**Files:**
- Modify: `src/backtest/optimizer.py`
- Modify: `tests/test_optimizer.py`

- [ ] **Step 1: Write failing test for GridSearchOptimizer**

```python
# tests/test_optimizer.py (append)
from backtest.optimizer import GridSearchOptimizer, ParamSpace


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_optimizer.py::TestGridSearchOptimizer -v`
Expected: FAIL with "ImportError" (GridSearchOptimizer not defined)

- [ ] **Step 3: Implement GridSearchOptimizer**

Add to `src/backtest/optimizer.py`:

```python
import multiprocessing
import time
import os


class GridSearchOptimizer:
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
        n_jobs: int | None = None,
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
        self.n_jobs = n_jobs or os.cpu_count() or 1

    def run(self) -> OptimizeResult:
        combos = self.param_space.grid()
        total = len(combos)
        t0 = time.time()

        trial_args_list = [
            {
                "db_path": self.db_path,
                "strategy_path": self.strategy_path,
                "symbol": self.symbol,
                "interval": self.interval,
                "exchange": "binance",
                "start": self.start,
                "end": self.end,
                "balance": self.balance,
                "leverage": self.leverage,
                "params": params,
                "objective": self.objective,
            }
            for params in combos
        ]

        if self.n_jobs == 1:
            results = [_run_single_trial(a) for a in trial_args_list]
        else:
            with multiprocessing.Pool(self.n_jobs) as pool:
                results = pool.map(_run_single_trial, trial_args_list)

        results.sort(key=lambda r: r["score"], reverse=True)
        elapsed = time.time() - t0

        return OptimizeResult(
            best_params=results[0]["params"] if results else {},
            best_score=results[0]["score"] if results else 0.0,
            all_trials=results,
            objective=self.objective,
            total_trials=total,
            elapsed_seconds=elapsed,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_optimizer.py::TestGridSearchOptimizer -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/backtest/optimizer.py tests/test_optimizer.py
git commit -m "feat(optimizer): add GridSearchOptimizer with multiprocessing"
```

---

### Task 4: OptunaOptimizer

**Files:**
- Modify: `src/backtest/optimizer.py`
- Modify: `tests/test_optimizer.py`

- [ ] **Step 1: Write failing test for OptunaOptimizer**

```python
# tests/test_optimizer.py (append)
from backtest.optimizer import OptunaOptimizer


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
```

Add at top of test file:

```python
import importlib.util
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_optimizer.py::TestOptunaOptimizer -v`
Expected: FAIL with "ImportError" (OptunaOptimizer not defined)

- [ ] **Step 3: Implement OptunaOptimizer**

Add to `src/backtest/optimizer.py`:

```python
class OptunaOptimizer:
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
        n_trials: int = 100,
        n_jobs: int = 1,
    ):
        try:
            import optuna  # noqa: F401
        except ImportError:
            raise ImportError(
                "OptunaOptimizer requires the 'optuna' package. "
                "Install it with: pip install optuna"
            )

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
        self.n_trials = n_trials
        self.n_jobs = n_jobs

    def _suggest_params(self, trial) -> dict:
        """Map ParamSpace to Optuna trial suggestions."""
        params = {}
        for name, spec in self.param_space._space.items():
            if isinstance(spec, list):
                params[name] = trial.suggest_categorical(name, spec)
            elif isinstance(spec, tuple):
                min_val, max_val, step = spec
                if isinstance(min_val, int) and isinstance(max_val, int) and isinstance(step, int):
                    params[name] = trial.suggest_int(name, min_val, max_val, step=step)
                else:
                    params[name] = trial.suggest_float(name, float(min_val), float(max_val), step=float(step))
        return params

    def run(self) -> OptimizeResult:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        all_trials: list[dict] = []
        t0 = time.time()

        def objective_fn(trial):
            params = self._suggest_params(trial)
            trial_args = {
                "db_path": self.db_path,
                "strategy_path": self.strategy_path,
                "symbol": self.symbol,
                "interval": self.interval,
                "exchange": "binance",
                "start": self.start,
                "end": self.end,
                "balance": self.balance,
                "leverage": self.leverage,
                "params": params,
                "objective": self.objective,
            }
            result = _run_single_trial(trial_args)
            all_trials.append(result)
            return result["score"]

        study = optuna.create_study(direction="maximize")
        study.optimize(objective_fn, n_trials=self.n_trials, n_jobs=self.n_jobs)

        all_trials.sort(key=lambda r: r["score"], reverse=True)
        elapsed = time.time() - t0

        return OptimizeResult(
            best_params=all_trials[0]["params"] if all_trials else {},
            best_score=all_trials[0]["score"] if all_trials else 0.0,
            all_trials=all_trials,
            objective=self.objective,
            total_trials=len(all_trials),
            elapsed_seconds=elapsed,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_optimizer.py::TestOptunaOptimizer -v`
Expected: PASS (or skip if optuna not installed)

- [ ] **Step 5: Commit**

```bash
git add src/backtest/optimizer.py tests/test_optimizer.py
git commit -m "feat(optimizer): add OptunaOptimizer with optional dependency"
```

---

### Task 5: Database Persistence

**Files:**
- Modify: `src/backtest/optimizer.py`
- Modify: `tests/test_optimizer.py`

- [ ] **Step 1: Write failing test for save_results**

```python
# tests/test_optimizer.py (append)
from backtest.optimizer import save_results, OptimizeResult


class TestSaveResults:
    def test_save_and_query(self):
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
        import json
        assert json.loads(rows[0][2]) == {"X": 10, "Y": 2.5}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_optimizer.py::TestSaveResults -v`
Expected: FAIL with "ImportError" (save_results not defined)

- [ ] **Step 3: Implement save_results**

Add to `src/backtest/optimizer.py`:

```python
import json
import sqlite3


def save_results(
    db_path: str,
    strategy: str,
    symbol: str,
    interval: str,
    start_date: str,
    end_date: str,
    result: OptimizeResult,
) -> None:
    """Save all optimization trials to SQLite."""
    from datetime import datetime, timezone

    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS optimize_results (
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

    now = datetime.now(timezone.utc).isoformat()
    rows = [
        (
            strategy, symbol, interval, start_date, end_date,
            result.objective, trial["score"],
            json.dumps(trial["params"]),
            json.dumps(trial.get("report", {})),
            now,
        )
        for trial in result.all_trials
    ]

    conn.executemany(
        "INSERT INTO optimize_results "
        "(strategy, symbol, interval, start_date, end_date, objective, score, params_json, report_json, created_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()
    conn.close()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_optimizer.py::TestSaveResults -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/backtest/optimizer.py tests/test_optimizer.py
git commit -m "feat(optimizer): add save_results for SQLite persistence"
```

---

### Task 6: CLI `optimize` Subcommand

**Files:**
- Modify: `src/backtest/__main__.py`
- Modify: `tests/test_optimizer.py`

- [ ] **Step 1: Write failing test for CLI param parsing**

```python
# tests/test_optimizer.py (append)
from backtest.optimizer import parse_params_string


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_optimizer.py::TestParseParamsString -v`
Expected: FAIL with "ImportError"

- [ ] **Step 3: Implement parse_params_string**

Add to `src/backtest/optimizer.py`:

```python
def parse_params_string(params_str: str) -> ParamSpace:
    """Parse CLI params string like 'X=1:10:2,Y=a|b|c' into ParamSpace."""
    space = {}
    for part in params_str.split(","):
        part = part.strip()
        if not part:
            continue
        name, spec_str = part.split("=", 1)
        name = name.strip()

        if "|" in spec_str:
            # Choice list
            values = []
            for v in spec_str.split("|"):
                values.append(_parse_number(v.strip()))
            space[name] = values
        elif ":" in spec_str:
            # Range: min:max:step
            parts = spec_str.split(":")
            if len(parts) != 3:
                raise ValueError(f"Range must be min:max:step, got: {spec_str}")
            min_val = _parse_number(parts[0])
            max_val = _parse_number(parts[1])
            step = _parse_number(parts[2])
            space[name] = (min_val, max_val, step)
        else:
            # Single value
            space[name] = [_parse_number(spec_str)]

    return ParamSpace(space)


def _parse_number(s: str):
    """Parse string to int or float."""
    try:
        val = int(s)
        return val
    except ValueError:
        return float(s)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_optimizer.py::TestParseParamsString -v`
Expected: All PASS

- [ ] **Step 5: Implement cmd_optimize in __main__.py**

Add to `src/backtest/__main__.py`:

```python
def cmd_optimize(args: argparse.Namespace) -> None:
    from backtest.optimizer import (
        GridSearchOptimizer, OptunaOptimizer, parse_params_string, save_results,
    )

    param_space = parse_params_string(args.params)
    strategy_class = _load_strategy(args.strategy)

    print(f"Optimizing {strategy_class.__name__}: {param_space.total_combinations} combinations, "
          f"{args.n_jobs} workers, objective={args.objective}")

    if args.method == "optuna":
        optimizer = OptunaOptimizer(
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
            n_trials=args.n_trials,
            n_jobs=args.n_jobs,
        )
    else:
        optimizer = GridSearchOptimizer(
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
            n_jobs=args.n_jobs,
        )

    result = optimizer.run()

    # Print results table
    print(f"\nOptimization Complete: {result.total_trials} trials, "
          f"best {result.objective} = {result.best_score:.4f} "
          f"({result.elapsed_seconds:.1f}s)\n")

    # Table header
    param_names = list(result.best_params.keys()) if result.best_params else []
    header = " Rank | " + " | ".join(f"{p:>12}" for p in param_names)
    header += f" | {'Score':>8} | {'Return':>8} | {'MaxDD':>8}"
    print(header)
    print("-" * len(header))

    top_n = min(args.top, len(result.all_trials))
    for i, trial in enumerate(result.all_trials[:top_n]):
        row = f" {i+1:>4} | "
        row += " | ".join(f"{trial['params'].get(p, ''):>12}" for p in param_names)
        report = trial.get("report", {})
        row += f" | {trial['score']:>8.4f}"
        row += f" | {report.get('net_return', 0):>+7.1%}"
        row += f" | {report.get('max_drawdown', 0):>-7.1%}"
        print(row)

    # Save to database
    report_db = str(Path(args.db or str(Path("data") / "klines.db")).parent / "reports.db")
    save_results(
        db_path=report_db,
        strategy=strategy_class.__name__,
        symbol=args.symbol,
        interval=args.interval,
        start_date=args.start,
        end_date=args.end,
        result=result,
    )
    print(f"\nResults saved to database ({result.total_trials} rows).")
```

Add the subparser in `main()` function, after `p_web`:

```python
    p_opt = sub.add_parser("optimize", help="Optimize strategy parameters")
    p_opt.add_argument("--strategy", required=True)
    p_opt.add_argument("--symbol", required=True)
    p_opt.add_argument("--interval", required=True)
    p_opt.add_argument("--exchange", default="binance")
    p_opt.add_argument("--start", required=True, help="YYYY-MM-DD")
    p_opt.add_argument("--end", required=True, help="YYYY-MM-DD")
    p_opt.add_argument("--balance", type=float, default=10000.0)
    p_opt.add_argument("--leverage", type=int, default=10)
    p_opt.add_argument("--params", required=True, help="e.g. X=1:10:2,Y=a|b|c")
    p_opt.add_argument("--objective", default="sharpe_ratio",
                       choices=["sharpe_ratio", "net_return", "sortino_ratio", "profit_factor", "win_rate"])
    p_opt.add_argument("--method", default="grid", choices=["grid", "optuna"])
    p_opt.add_argument("--n-jobs", type=int, default=None)
    p_opt.add_argument("--n-trials", type=int, default=100, help="For optuna method")
    p_opt.add_argument("--top", type=int, default=10, help="Show top N results")
    p_opt.add_argument("--db", default=None)
```

Add the command dispatch in the `if/elif` chain:

```python
    elif args.command == "optimize":
        cmd_optimize(args)
```

- [ ] **Step 6: Run all tests**

Run: `pytest tests/ -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add src/backtest/__main__.py src/backtest/optimizer.py tests/test_optimizer.py
git commit -m "feat(optimizer): add CLI optimize subcommand"
```

---

### Task 7: Integration Test with Shadow Power Strategy

**Files:**
- Modify: `tests/test_optimizer.py`

- [ ] **Step 1: Write integration test**

```python
# tests/test_optimizer.py (append)
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
        import math
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
```

- [ ] **Step 2: Run integration test**

Run: `pytest tests/test_optimizer.py::TestShadowPowerOptimize -v`
Expected: PASS

- [ ] **Step 3: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests PASS (existing + new)

- [ ] **Step 4: Commit**

```bash
git add tests/test_optimizer.py
git commit -m "test(optimizer): add integration test with Shadow Power strategy"
```

---

### Task 8: Progress Output and Final Polish

**Files:**
- Modify: `src/backtest/optimizer.py`

- [ ] **Step 1: Add progress callback to GridSearchOptimizer**

Update `GridSearchOptimizer.run()` to print progress:

```python
    def run(self) -> OptimizeResult:
        combos = self.param_space.grid()
        total = len(combos)
        t0 = time.time()

        trial_args_list = [
            {
                "db_path": self.db_path,
                "strategy_path": self.strategy_path,
                "symbol": self.symbol,
                "interval": self.interval,
                "exchange": "binance",
                "start": self.start,
                "end": self.end,
                "balance": self.balance,
                "leverage": self.leverage,
                "params": params,
                "objective": self.objective,
            }
            for params in combos
        ]

        if self.n_jobs == 1:
            results = []
            for i, args in enumerate(trial_args_list, 1):
                results.append(_run_single_trial(args))
                self._print_progress(i, total)
        else:
            with multiprocessing.Pool(self.n_jobs) as pool:
                results = []
                for i, r in enumerate(pool.imap_unordered(_run_single_trial, trial_args_list), 1):
                    results.append(r)
                    self._print_progress(i, total)

        print()  # newline after progress bar
        results.sort(key=lambda r: r["score"], reverse=True)
        elapsed = time.time() - t0

        return OptimizeResult(
            best_params=results[0]["params"] if results else {},
            best_score=results[0]["score"] if results else 0.0,
            all_trials=results,
            objective=self.objective,
            total_trials=total,
            elapsed_seconds=elapsed,
        )

    @staticmethod
    def _print_progress(current: int, total: int) -> None:
        bar_len = 40
        filled = int(bar_len * current / total)
        bar = "#" * filled + "-" * (bar_len - filled)
        print(f"\r[{bar}] {current}/{total}", end="", flush=True)
```

- [ ] **Step 2: Run full test suite**

Run: `pytest tests/ -v`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
git add src/backtest/optimizer.py
git commit -m "feat(optimizer): add progress bar output"
```

---

### Task 9: Manual Smoke Test

- [ ] **Step 1: Verify CLI help**

Run: `python -m backtest optimize --help`
Expected: Shows all optimizer flags

- [ ] **Step 2: Run a real optimization (if data available)**

```bash
python -m backtest optimize \
    --strategy strategies/example_ma_cross.py \
    --symbol BTCUSDT --interval 1h \
    --start 2024-01-01 --end 2024-03-01 \
    --balance 10000 --leverage 10 \
    --params "short_period=5:10:1,long_period=20:30:5" \
    --objective sharpe_ratio \
    --method grid --n-jobs 4 --top 5
```

Expected: Table output with ranked results, saved to DB.

- [ ] **Step 3: Final commit if any fixes needed**
