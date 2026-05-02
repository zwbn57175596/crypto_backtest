from __future__ import annotations

import importlib.util
import itertools
import json
import multiprocessing
import os
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class OptimizeResult:
    best_params: dict
    best_score: float
    all_trials: list[dict]
    objective: str
    total_trials: int
    elapsed_seconds: float


def _is_int_value(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _parse_number(s: str):
    """Parse string to int or float."""
    try:
        val = int(s)
        return val
    except ValueError:
        return float(s)


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


def expand_param_values(spec) -> list:
    """Expand one parameter spec into a concrete value list."""
    if isinstance(spec, list):
        return list(spec)
    if isinstance(spec, tuple) and len(spec) == 3:
        min_val, max_val, step = spec
        if _is_int_value(min_val) and _is_int_value(max_val) and _is_int_value(step):
            return list(range(min_val, max_val + 1, step))

        values = []
        v = float(min_val)
        while v <= float(max_val) + float(step) * 0.01:
            values.append(round(v, 10))
            v += float(step)
        return values
    return [spec]


def load_strategy_optimize_space(strategy_path: str) -> ParamSpace:
    """Load strategy-defined OPTIMIZE_SPACE into a ParamSpace."""
    strategy_class = _load_strategy_class(strategy_path)
    raw_space = getattr(strategy_class, "OPTIMIZE_SPACE", None)
    if not raw_space:
        raise ValueError(
            f"Strategy '{strategy_class.__name__}' does not define OPTIMIZE_SPACE"
        )
    if not isinstance(raw_space, dict):
        raise ValueError("OPTIMIZE_SPACE must be a dict of param specs")
    return ParamSpace(raw_space)


def load_strategy_auto_optimize_config(strategy_path: str) -> dict:
    """Load strategy-defined AUTO_OPTIMIZE_CONFIG."""
    strategy_class = _load_strategy_class(strategy_path)
    config = getattr(strategy_class, "AUTO_OPTIMIZE_CONFIG", {})
    if config is None:
        return {}
    if not isinstance(config, dict):
        raise ValueError("AUTO_OPTIMIZE_CONFIG must be a dict")
    return config


def load_strategy_param_defaults(strategy_path: str, param_names: list[str] | None = None) -> dict:
    """Load default class attribute values for strategy params."""
    strategy_class = _load_strategy_class(strategy_path)
    if param_names is None:
        names = [
            name for name in dir(strategy_class)
            if name.isupper() and not name.startswith("_")
        ]
    else:
        names = param_names
    defaults = {}
    for name in names:
        if hasattr(strategy_class, name):
            defaults[name] = getattr(strategy_class, name)
    return defaults


class ExplicitParamSpace:
    """Parameter space backed by an explicit list of parameter combinations."""

    def __init__(self, combos: list[dict]):
        deduped: list[dict] = []
        seen: set[str] = set()
        for combo in combos:
            key = json.dumps(combo, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(dict(combo))
        self._combos = deduped

    @property
    def total_combinations(self) -> int:
        return len(self._combos)

    def grid(self) -> list[dict]:
        return [dict(combo) for combo in self._combos]


def merge_trials(*trial_groups: list[dict]) -> list[dict]:
    """Merge trial lists by params, keeping the highest score per combination."""
    merged: dict[str, dict] = {}
    for trials in trial_groups:
        for trial in trials:
            key = json.dumps(trial["params"], sort_keys=True)
            existing = merged.get(key)
            if existing is None or trial["score"] > existing["score"]:
                merged[key] = trial
    return sorted(merged.values(), key=lambda item: item["score"], reverse=True)


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


class ParamSpace:
    """Defines a parameter search space for strategy optimization."""

    def __init__(self, space: dict):
        self._space = space
        self._axes: dict[str, list] = {}
        for name, spec in space.items():
            self._axes[name] = expand_param_values(spec)

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


class GridSearchOptimizer:
    """Grid search optimizer that runs all parameter combinations."""

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
        exchange: str = "binance",
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
        self.exchange = exchange
        self.param_space = param_space or ParamSpace({})
        self.objective = objective
        self.n_jobs = n_jobs or os.cpu_count() or 1

    def run(self) -> OptimizeResult:
        """Run grid search over all parameter combinations."""
        combos = self.param_space.grid()
        total = len(combos)
        t0 = time.time()

        trial_args_list = [
            {
                "db_path": self.db_path,
                "strategy_path": self.strategy_path,
                "symbol": self.symbol,
                "interval": self.interval,
                "exchange": self.exchange,
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


class OptunaOptimizer:
    """Bayesian optimization using Optuna."""

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
        exchange: str = "binance",
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
        self.exchange = exchange
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
                    params[name] = trial.suggest_float(
                        name, float(min_val), float(max_val), step=float(step)
                    )
            else:
                params[name] = trial.suggest_categorical(name, [spec])
        return params

    def run(self) -> OptimizeResult:
        """Run Bayesian optimization with Optuna."""
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
                "exchange": self.exchange,
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


def _numba_worker(args: tuple) -> dict:
    """Worker function for NumbaGridOptimizer multiprocessing."""
    from backtest.numba_simulate import simulate

    bars, params, sizing_leverage, exchange_leverage, commission_rate, \
        funding_rate, maintenance_margin, initial_balance, objective = args

    result = simulate(
        bars,
        threshold=int(params.get("CONSECUTIVE_THRESHOLD", 5)),
        multiplier=float(params.get("POSITION_MULTIPLIER", 1.1)),
        initial_pct=float(params.get("INITIAL_POSITION_PCT", 0.01)),
        profit_threshold=int(params.get("PROFIT_CANDLE_THRESHOLD", 1)),
        sizing_leverage=sizing_leverage,
        exchange_leverage=exchange_leverage,
        commission_rate=commission_rate,
        funding_rate=funding_rate,
        maintenance_margin=maintenance_margin,
        initial_balance=initial_balance,
    )

    # Map result tuple to metric names
    metrics = {
        "net_return": result[0],
        "annual_return": result[1],
        "max_drawdown": result[2],
        "sharpe_ratio": result[3],
        "sortino_ratio": result[4],
        "win_rate": result[5],
        "profit_factor": result[6],
        "total_trades": result[7],
    }

    score = metrics.get(objective, 0.0)
    if not isinstance(score, (int, float)) or score != score:  # nan check
        score = float("-inf")
    elif score > 1e9:
        score = 1e9

    return {
        "params": params,
        "score": score,
        "report": metrics,
    }


# Shared bars array for multiprocessing (avoids pickling per worker)
_shared_bars = None


def _numba_worker_shared(args: tuple) -> dict:
    """Worker using module-level shared bars to avoid pickling large arrays."""
    global _shared_bars
    from backtest.numba_simulate import simulate

    params, sizing_leverage, exchange_leverage, commission_rate, \
        funding_rate, maintenance_margin, initial_balance, objective = args

    result = simulate(
        _shared_bars,
        threshold=int(params.get("CONSECUTIVE_THRESHOLD", 5)),
        multiplier=float(params.get("POSITION_MULTIPLIER", 1.1)),
        initial_pct=float(params.get("INITIAL_POSITION_PCT", 0.01)),
        profit_threshold=int(params.get("PROFIT_CANDLE_THRESHOLD", 1)),
        sizing_leverage=sizing_leverage,
        exchange_leverage=exchange_leverage,
        commission_rate=commission_rate,
        funding_rate=funding_rate,
        maintenance_margin=maintenance_margin,
        initial_balance=initial_balance,
    )

    metrics = {
        "net_return": result[0],
        "annual_return": result[1],
        "max_drawdown": result[2],
        "sharpe_ratio": result[3],
        "sortino_ratio": result[4],
        "win_rate": result[5],
        "profit_factor": result[6],
        "total_trades": result[7],
    }

    score = metrics.get(objective, 0.0)
    if not isinstance(score, (int, float)) or score != score:
        score = float("-inf")
    elif score > 1e9:
        score = 1e9

    return {"params": params, "score": score, "report": metrics}


def _init_shared_bars(bars):
    """Initializer for multiprocessing.Pool to set shared bars."""
    global _shared_bars
    _shared_bars = bars


class NumbaGridOptimizer:
    """Grid search optimizer using Numba JIT-compiled simulation.

    ~50-200x faster than GridSearchOptimizer for ConsecutiveReverse strategy.
    """

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
        exchange: str = "binance",
        param_space: ParamSpace | None = None,
        objective: str = "sharpe_ratio",
        n_jobs: int | None = None,
        commission_rate: float = 0.0004,
        funding_rate: float = 0.0001,
        maintenance_margin: float = 0.005,
    ):
        self.db_path = db_path
        self.strategy_path = strategy_path
        self.symbol = symbol
        self.interval = interval
        self.start = f"{start} 00:00:00" if len(start) == 10 else start
        self.end = f"{end} 23:59:59" if len(end) == 10 else end
        self.balance = balance
        self.leverage = leverage
        self.exchange = exchange
        self.param_space = param_space or ParamSpace({})
        self.objective = objective
        self.n_jobs = n_jobs or os.cpu_count() or 1
        self.commission_rate = commission_rate
        self.funding_rate = funding_rate
        self.maintenance_margin = maintenance_margin

    def run(self) -> OptimizeResult:
        """Run grid search with Numba-accelerated simulation."""
        from datetime import datetime, timezone
        from backtest.numba_simulate import load_bars, simulate

        # Pre-load bars once
        start_ts = int(
            datetime.strptime(self.start, "%Y-%m-%d %H:%M:%S")
            .replace(tzinfo=timezone.utc).timestamp() * 1000
        )
        end_ts = int(
            datetime.strptime(self.end, "%Y-%m-%d %H:%M:%S")
            .replace(tzinfo=timezone.utc).timestamp() * 1000
        )
        bars = load_bars(self.db_path, self.symbol, self.interval, self.exchange, start_ts, end_ts)
        print(f"Loaded {bars.shape[0]} bars, warming up JIT...", flush=True)

        # Warm up JIT compilation
        simulate(bars[:10], 5, 1.1, 0.01, 1, 50, 50, 0.0004, 0.0001, 0.005, 1000.0)

        combos = self.param_space.grid()
        total = len(combos)
        t0 = time.time()
        default_params = load_strategy_param_defaults(
            self.strategy_path,
            [
                "CONSECUTIVE_THRESHOLD",
                "POSITION_MULTIPLIER",
                "INITIAL_POSITION_PCT",
                "PROFIT_CANDLE_THRESHOLD",
                "LEVERAGE",
            ],
        )

        if self.n_jobs == 1:
            # Single process - fastest for small grids
            results = []
            for i, params in enumerate(combos, 1):
                sizing_lev = int(params.get("LEVERAGE", default_params.get("LEVERAGE", self.leverage)))
                result = simulate(
                    bars,
                    threshold=int(params.get("CONSECUTIVE_THRESHOLD", default_params.get("CONSECUTIVE_THRESHOLD", 5))),
                    multiplier=float(params.get("POSITION_MULTIPLIER", default_params.get("POSITION_MULTIPLIER", 1.1))),
                    initial_pct=float(params.get("INITIAL_POSITION_PCT", default_params.get("INITIAL_POSITION_PCT", 0.01))),
                    profit_threshold=int(params.get("PROFIT_CANDLE_THRESHOLD", default_params.get("PROFIT_CANDLE_THRESHOLD", 1))),
                    sizing_leverage=sizing_lev,
                    exchange_leverage=self.leverage,
                    commission_rate=self.commission_rate,
                    funding_rate=self.funding_rate,
                    maintenance_margin=self.maintenance_margin,
                    initial_balance=self.balance,
                )
                metrics = {
                    "net_return": result[0], "annual_return": result[1],
                    "max_drawdown": result[2], "sharpe_ratio": result[3],
                    "sortino_ratio": result[4], "win_rate": result[5],
                    "profit_factor": result[6], "total_trades": result[7],
                }
                score = metrics.get(self.objective, 0.0)
                if score != score or abs(score) > 1e9:
                    score = float("-inf")
                results.append({"params": params, "score": score, "report": metrics})
                if i % 1000 == 0 or i == total:
                    self._print_progress(i, total)
        else:
            # Multiprocessing with shared bars
            trial_args = [
                (
                    params,
                    int(params.get("LEVERAGE", default_params.get("LEVERAGE", self.leverage))),
                    self.leverage,
                    self.commission_rate,
                    self.funding_rate,
                    self.maintenance_margin,
                    self.balance,
                    self.objective,
                )
                for params in combos
            ]

            with multiprocessing.Pool(
                self.n_jobs, initializer=_init_shared_bars, initargs=(bars,)
            ) as pool:
                results = []
                for i, r in enumerate(
                    pool.imap_unordered(_numba_worker_shared, trial_args, chunksize=256), 1
                ):
                    results.append(r)
                    if i % 1000 == 0 or i == total:
                        self._print_progress(i, total)

        print()
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
        pct = current / total * 100
        print(f"\r[{bar}] {current}/{total} ({pct:.1f}%)", end="", flush=True)


def save_results(
    db_path: str,
    strategy: str,
    symbol: str,
    interval: str,
    start_date: str,
    end_date: str,
    result: OptimizeResult,
    top_n: int | None = 1000,
) -> None:
    """Save top N optimization trials to SQLite (default: top 1000)."""
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
            created_at TEXT NOT NULL,
            batch_id TEXT
        )
    """)

    # Idempotent migration — add batch_id column if not present
    try:
        conn.execute("ALTER TABLE optimize_results ADD COLUMN batch_id TEXT")
        conn.commit()
    except sqlite3.OperationalError:
        pass  # column already exists

    # Backfill old rows without batch_id
    conn.execute("""
        UPDATE optimize_results
        SET batch_id = strftime('%Y%m%dT%H%M%S', created_at) || '_' || strategy || '_' || symbol
        WHERE batch_id IS NULL
    """)
    conn.commit()

    trials = result.all_trials[:top_n] if top_n else result.all_trials
    now = datetime.now(timezone.utc).isoformat()
    # Generate batch_id from the timestamp: convert "2026-04-23T12:00:00+00:00" to "20260423T120000"
    batch_id = now[:19].replace("-", "").replace(":", "") + "_" + strategy + "_" + symbol

    rows = [
        (
            strategy, symbol, interval, start_date, end_date,
            result.objective, trial["score"],
            json.dumps(trial["params"], sort_keys=True),
            json.dumps(trial.get("report", {}), sort_keys=True),
            now,
            batch_id,
        )
        for trial in trials
    ]

    conn.executemany(
        "INSERT INTO optimize_results "
        "(strategy, symbol, interval, start_date, end_date, objective, score, params_json, report_json, created_at, batch_id) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()
    conn.close()


def _write_report_with_link(
    report_db_path: str,
    strategy_name: str,
    symbol: str,
    interval: str,
    created_at: str,
    report: dict,
    params: dict,
    base_strategy: str,
) -> None:
    """Insert one report row, linking to the matching optimize_results row."""
    conn = sqlite3.connect(report_db_path)
    try:
        # Ensure reports table exists
        conn.execute("""
            CREATE TABLE IF NOT EXISTS reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy TEXT, symbol TEXT, interval TEXT,
                created_at TEXT, report_json TEXT
            )
        """)

        # Idempotent migration — add column if not present
        try:
            conn.execute("ALTER TABLE reports ADD COLUMN optimize_result_id INTEGER")
            conn.commit()
        except sqlite3.OperationalError:
            pass  # column already exists

        # Look up matching optimize_results row
        params_json = json.dumps(params, sort_keys=True)
        optimize_result_id = None
        try:
            row = conn.execute(
                """SELECT id FROM optimize_results
                   WHERE params_json = ? AND strategy = ? AND symbol = ? AND interval = ?
                   ORDER BY created_at DESC LIMIT 1""",
                (params_json, base_strategy, symbol, interval),
            ).fetchone()
            optimize_result_id = row[0] if row else None
        except sqlite3.OperationalError:
            pass  # optimize_results table doesn't exist yet

        conn.execute(
            """INSERT INTO reports
               (strategy, symbol, interval, created_at, report_json, optimize_result_id)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (strategy_name, symbol, interval, created_at,
             json.dumps(report, sort_keys=True), optimize_result_id),
        )
        conn.commit()
    finally:
        conn.close()


def save_top_reports(
    result: OptimizeResult,
    top_n: int,
    db_path: str,
    report_db_path: str,
    strategy_path: str,
    symbol: str,
    interval: str,
    start: str,
    end: str,
    balance: float,
    leverage: int,
    exchange: str = "binance",
) -> None:
    """Re-run top N trials and save full reports (with equity_curve + trades) to reports table."""
    from backtest.engine import BacktestEngine
    from backtest.reporter import Reporter
    from datetime import datetime, timezone

    strategy_class = _load_strategy_class(strategy_path)
    start_fmt = f"{start} 00:00:00" if len(start) == 10 else start
    end_fmt = f"{end} 23:59:59" if len(end) == 10 else end

    # Ensure reports table exists
    conn = sqlite3.connect(report_db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy TEXT, symbol TEXT, interval TEXT,
            created_at TEXT, report_json TEXT
        )
    """)
    conn.commit()
    conn.close()

    trials = result.all_trials[:top_n]
    now = datetime.now(timezone.utc).isoformat()

    for i, trial in enumerate(trials, 1):
        trial_class = _make_strategy(strategy_class, trial["params"])
        engine = BacktestEngine(
            db_path=db_path,
            symbol=symbol,
            interval=interval,
            exchange=exchange,
            strategy_class=trial_class,
            balance=balance,
            leverage=leverage,
            start=start_fmt,
            end=end_fmt,
        )
        run_result = engine.run()
        report = Reporter.generate(run_result)

        strategy_name = f"{strategy_class.__name__}_opt{i}"
        _write_report_with_link(
            report_db_path=report_db_path,
            strategy_name=strategy_name,
            symbol=symbol,
            interval=interval,
            created_at=now,
            report=report,
            params=trial["params"],
            base_strategy=strategy_class.__name__,
        )
