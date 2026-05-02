"""CUDA grid search optimizer for GPU-accelerated backtesting.

This module provides the CudaGridOptimizer class that orchestrates GPU-accelerated
grid search for strategy parameter optimization using CUDA.
"""

import itertools
import math
import time
from datetime import datetime, timezone

from backtest.optimizer import (
    ExplicitParamSpace,
    OptimizeResult,
    ParamSpace,
    expand_param_values,
    load_strategy_auto_optimize_config,
    load_strategy_optimize_space,
    load_strategy_param_defaults,
    merge_trials,
)


class CudaGridOptimizer:
    """Grid search optimizer using CUDA GPU for parallel simulation.

    Runs thousands of parameter combinations on GPU in batches, with automatic
    VRAM-aware batching to maximize throughput.
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
        commission_rate: float = 0.0004,
        funding_rate: float = 0.0001,
        maintenance_margin: float = 0.005,
        batch_size: int | None = None,
        default_params: dict | None = None,
    ):
        """Initialize CudaGridOptimizer.

        Parameters
        ----------
        db_path : str
            Path to SQLite database with klines data
        strategy_path : str
            Path to strategy file (must contain ConsecutiveReverseStrategy)
        symbol : str
            Symbol to backtest (e.g., 'BTCUSDT')
        interval : str
            Kline interval (e.g., '1h', '1d')
        start : str
            Start date (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)
        end : str
            End date (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)
        balance : float
            Initial balance in USDT (default 10000.0)
        leverage : int
            Exchange leverage (default 10)
        param_space : ParamSpace | None
            Parameter search space (default empty)
        objective : str
            Objective metric to maximize (default 'sharpe_ratio')
        commission_rate : float
            Commission rate (default 0.0004)
        funding_rate : float
            Funding rate (default 0.0001)
        maintenance_margin : float
            Maintenance margin ratio (default 0.005)
        batch_size : int | None
            Number of combos per batch (None = auto-detect from VRAM)
        """
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
        self.commission_rate = commission_rate
        self.funding_rate = funding_rate
        self.maintenance_margin = maintenance_margin
        self.batch_size = batch_size
        self.default_params = default_params

    def run(self) -> OptimizeResult:
        """Run GPU-accelerated grid search.

        Returns
        -------
        OptimizeResult
            Optimization results with sorted trials by score
        """
        try:
            from numba import cuda
            import numpy as np
        except ImportError:
            raise ImportError("CUDA optimization requires 'numba' and 'numpy'")
        try:
            if not cuda.is_available():
                raise RuntimeError("CUDA runtime is not available on this machine")
        except Exception as exc:
            raise RuntimeError("CUDA runtime is not available on this machine") from exc

        from backtest.cuda_strategies import CUDA_STRATEGIES
        from backtest.numba_simulate import load_bars

        # 1. Load strategy class and verify it's registered
        strategy_name = self._get_strategy_name(self.strategy_path)
        if strategy_name not in CUDA_STRATEGIES:
            available = ", ".join(CUDA_STRATEGIES.keys())
            raise ValueError(
                f"Strategy '{strategy_name}' not found in CUDA_STRATEGIES registry. "
                f"Available: {available}"
            )

        registry_entry = CUDA_STRATEGIES[strategy_name]
        kernel = registry_entry["kernel"]
        param_order = registry_entry["param_order"]
        default_params = self.default_params or load_strategy_param_defaults(
            self.strategy_path, param_order
        )

        # 2. Load bars from database
        start_ts = int(
            datetime.strptime(self.start, "%Y-%m-%d %H:%M:%S")
            .replace(tzinfo=timezone.utc)
            .timestamp()
            * 1000
        )
        end_ts = int(
            datetime.strptime(self.end, "%Y-%m-%d %H:%M:%S")
            .replace(tzinfo=timezone.utc)
            .timestamp()
            * 1000
        )

        bars = load_bars(self.db_path, self.symbol, self.interval, self.exchange, start_ts, end_ts)
        print(f"Loaded {bars.shape[0]} bars, preparing GPU...", flush=True)

        # 3. Copy bars to GPU once
        bars_gpu = cuda.to_device(bars.astype(np.float64))

        # 4. Build params array from param_space
        combos = self.param_space.grid()
        total = len(combos)

        # Convert combos to params array matching param_order
        params_list = []
        for combo in combos:
            param_row = []
            for param_name in param_order:
                value = combo.get(param_name, default_params.get(param_name, 1))
                param_row.append(float(value))
            params_list.append(param_row)

        params_array = np.array(params_list, dtype=np.float64)

        # 5. Auto-detect batch_size if not set
        batch_size = self.batch_size
        if batch_size is None:
            batch_size = self._auto_detect_batch_size(len(param_order))

        # 6. Process batches
        results = []
        t0 = time.time()
        n_batches = (total + batch_size - 1) // batch_size

        for batch_idx in range(n_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, total)
            batch_n = batch_end - batch_start

            # Copy batch params to GPU
            batch_params = params_array[batch_start:batch_end].copy()
            params_gpu = cuda.to_device(batch_params)

            # Allocate results array on GPU
            results_gpu = cuda.device_array((batch_n, 8), dtype=np.float64)

            # Launch kernel with 256 threads per block
            threads = 256
            blocks = (batch_n + threads - 1) // threads

            kernel[blocks, threads](
                bars_gpu,
                params_gpu,
                results_gpu,
                bars.shape[0],
                batch_n,
                self.leverage,
                self.commission_rate,
                self.funding_rate,
                self.maintenance_margin,
                self.balance,
            )

            cuda.synchronize()

            # Copy results back
            batch_results = results_gpu.copy_to_host()

            # Build trial dicts
            for i, param_row in enumerate(batch_params):
                combo = combos[batch_start + i]
                metrics_array = batch_results[i]

                # Extract metrics in order
                metrics = {
                    "net_return": float(metrics_array[0]),
                    "annual_return": float(metrics_array[1]),
                    "max_drawdown": float(metrics_array[2]),
                    "sharpe_ratio": float(metrics_array[3]),
                    "sortino_ratio": float(metrics_array[4]),
                    "win_rate": float(metrics_array[5]),
                    "profit_factor": float(metrics_array[6]),
                    "total_trades": float(metrics_array[7]),
                }

                # Sanitize score
                score = metrics.get(self.objective, 0.0)
                if score != score:  # NaN check
                    score = float("-inf")
                elif abs(score) > 1e9:
                    score = 1e9 if score > 0 else -1e9

                results.append({"params": combo, "score": score, "report": metrics})

            # Progress bar
            current = batch_end
            self._print_progress(current, total)

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
    def _get_strategy_name(strategy_path: str) -> str:
        """Load strategy file and return the BaseStrategy subclass name."""
        import importlib.util
        from backtest.strategy import BaseStrategy

        spec = importlib.util.spec_from_file_location("user_strategy", strategy_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for attr in dir(module):
            obj = getattr(module, attr)
            if isinstance(obj, type) and issubclass(obj, BaseStrategy) and obj is not BaseStrategy:
                return obj.__name__
        raise ValueError(f"No BaseStrategy subclass found in {strategy_path}")

    @staticmethod
    def _auto_detect_batch_size(n_params: int) -> int:
        """Auto-detect batch size from available VRAM.

        Allocates 70% of free VRAM, with each combo needing:
        n_params * 8 + 8 * 8 bytes (params + metrics)
        """
        try:
            from numba import cuda

            free_bytes, _ = cuda.current_context().get_memory_info()
        except Exception:
            # Fallback if VRAM detection fails
            return 128

        # Use 70% of free memory
        usable_bytes = int(free_bytes * 0.7)

        # Each combo: n_params * 8 + results 8 * 8
        bytes_per_combo = n_params * 8 + 64

        batch_size = max(1, usable_bytes // bytes_per_combo)
        return batch_size

    @staticmethod
    def _print_progress(current: int, total: int) -> None:
        """Print progress bar."""
        bar_len = 40
        filled = int(bar_len * current / total)
        bar = "#" * filled + "-" * (bar_len - filled)
        pct = current / total * 100
        print(f"\r[{bar}] {current}/{total} ({pct:.1f}%)", end="", flush=True)


def _is_int_step(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _default_refine_step(spec):
    if isinstance(spec, tuple) and len(spec) == 3:
        step = spec[2]
        if _is_int_step(step):
            return max(1, step // 2) if step > 1 else 1
        return round(float(step) / 2, 10)
    return None


def _default_refine_radius(spec):
    if isinstance(spec, tuple) and len(spec) == 3:
        return spec[2]
    return 0


def _build_local_values(center, coarse_spec, refine_spec: dict) -> list:
    if isinstance(coarse_spec, list):
        values = expand_param_values(coarse_spec)
        radius = refine_spec.get("radius")
        if radius is None:
            return [center] if center in values else values
        lower = center - radius
        upper = center + radius
        filtered = [v for v in values if lower <= v <= upper]
        return filtered or [center]

    if isinstance(coarse_spec, tuple) and len(coarse_spec) == 3:
        coarse_min, coarse_max, _ = coarse_spec
        radius = refine_spec.get("radius", _default_refine_radius(coarse_spec))
        step = refine_spec.get("step", _default_refine_step(coarse_spec))
        if step in (None, 0):
            return [center]

        min_v = refine_spec.get("min", coarse_min)
        max_v = refine_spec.get("max", coarse_max)
        lower = max(float(min_v), float(center) - float(radius))
        upper = min(float(max_v), float(center) + float(radius))
        local_spec = (lower, upper, step)
        values = expand_param_values(local_spec)
        if _is_int_step(step):
            center = int(center)
            values = [int(v) for v in values]
        else:
            center = round(float(center), 10)
            values = [round(float(v), 10) for v in values]
        if center not in values:
            values.append(center)
        return sorted(set(values))

    return [center]


class CudaAutoOptimizer:
    """Two-stage CUDA optimizer using strategy-defined search space."""

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
        commission_rate: float = 0.0004,
        funding_rate: float = 0.0001,
        maintenance_margin: float = 0.005,
        batch_size: int | None = None,
    ):
        self.db_path = db_path
        self.strategy_path = strategy_path
        self.symbol = symbol
        self.interval = interval
        self.start = start
        self.end = end
        self.balance = balance
        self.leverage = leverage
        self.exchange = exchange
        self.param_space = param_space
        self.objective = objective
        self.commission_rate = commission_rate
        self.funding_rate = funding_rate
        self.maintenance_margin = maintenance_margin
        self.batch_size = batch_size

    def run(self) -> OptimizeResult:
        coarse_space = self.param_space or load_strategy_optimize_space(self.strategy_path)
        auto_config = load_strategy_auto_optimize_config(self.strategy_path)
        refine_top_k = int(auto_config.get("refine_top_k", 8))
        refine_space_config = auto_config.get("refine_space", {})
        default_params = load_strategy_param_defaults(self.strategy_path)

        print(
            f"Stage 1/2: coarse CUDA grid over {coarse_space.total_combinations} combinations...",
            flush=True,
        )
        stage1 = self._run_cuda_stage(coarse_space, default_params)

        refine_space = self._build_refine_param_space(
            coarse_space=coarse_space,
            top_trials=stage1.all_trials[:refine_top_k],
            refine_space_config=refine_space_config,
        )
        if refine_space.total_combinations == 0:
            return stage1

        print(
            f"Stage 2/2: refined CUDA grid over {refine_space.total_combinations} combinations...",
            flush=True,
        )
        stage2 = self._run_cuda_stage(refine_space, default_params)

        merged_trials = merge_trials(stage1.all_trials, stage2.all_trials)
        elapsed = stage1.elapsed_seconds + stage2.elapsed_seconds
        return OptimizeResult(
            best_params=merged_trials[0]["params"] if merged_trials else {},
            best_score=merged_trials[0]["score"] if merged_trials else 0.0,
            all_trials=merged_trials,
            objective=self.objective,
            total_trials=len(merged_trials),
            elapsed_seconds=elapsed,
        )

    def _run_cuda_stage(self, param_space, default_params: dict) -> OptimizeResult:
        optimizer = CudaGridOptimizer(
            db_path=self.db_path,
            strategy_path=self.strategy_path,
            symbol=self.symbol,
            interval=self.interval,
            start=self.start,
            end=self.end,
            balance=self.balance,
            leverage=self.leverage,
            exchange=self.exchange,
            param_space=param_space,
            objective=self.objective,
            commission_rate=self.commission_rate,
            funding_rate=self.funding_rate,
            maintenance_margin=self.maintenance_margin,
            batch_size=self.batch_size,
            default_params=default_params,
        )
        return optimizer.run()

    def _build_refine_param_space(
        self,
        coarse_space: ParamSpace,
        top_trials: list[dict],
        refine_space_config: dict,
    ) -> ExplicitParamSpace:
        if not top_trials:
            return ExplicitParamSpace([])

        combos: list[dict] = []
        for trial in top_trials:
            local_axes: dict[str, list] = {}
            for param_name, coarse_spec in coarse_space._space.items():
                center = trial["params"].get(param_name)
                if center is None:
                    continue
                local_axes[param_name] = _build_local_values(
                    center=center,
                    coarse_spec=coarse_spec,
                    refine_spec=refine_space_config.get(param_name, {}),
                )

            if not local_axes:
                continue

            names = list(local_axes.keys())
            values = [local_axes[name] for name in names]
            for combo in itertools.product(*values):
                combos.append(dict(zip(names, combo)))

        return ExplicitParamSpace(combos)
