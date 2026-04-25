"""CUDA grid search optimizer for GPU-accelerated backtesting.

This module provides the CudaGridOptimizer class that orchestrates GPU-accelerated
grid search for strategy parameter optimization using CUDA.
"""

import math
import time
from datetime import datetime, timezone

from backtest.optimizer import OptimizeResult, ParamSpace


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
        param_space: ParamSpace | None = None,
        objective: str = "sharpe_ratio",
        commission_rate: float = 0.0004,
        funding_rate: float = 0.0001,
        maintenance_margin: float = 0.005,
        batch_size: int | None = None,
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
        self.param_space = param_space or ParamSpace({})
        self.objective = objective
        self.commission_rate = commission_rate
        self.funding_rate = funding_rate
        self.maintenance_margin = maintenance_margin
        self.batch_size = batch_size

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

        bars = load_bars(self.db_path, self.symbol, self.interval, "binance", start_ts, end_ts)
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
                value = combo.get(param_name, 1)  # Default fallback
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
        """Extract strategy class name from file path.

        Expects filename like consecutive_reverse.py → ConsecutiveReverseStrategy
        """
        import os

        basename = os.path.basename(strategy_path)
        # Convert snake_case to CamelCase and append "Strategy"
        parts = basename.replace(".py", "").split("_")
        class_name = "".join(p.capitalize() for p in parts) + "Strategy"
        return class_name

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
