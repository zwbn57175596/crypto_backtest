from __future__ import annotations

import importlib.util
import itertools
import multiprocessing
import os
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
