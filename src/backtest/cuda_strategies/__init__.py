"""CUDA strategy kernels for GPU-accelerated backtesting.

This module provides a registry of CUDA-compiled strategy kernels that can be
used for parallel backtesting with GPU acceleration.

Note: The current consecutive_reverse_kernel implements the martingale (add-to-
position on loss candle) logic, NOT the original close+reopen logic. It is only
registered for ConsecutiveReverseMartingaleStrategy. Optimizing the close+reopen
version (ConsecutiveReverseStrategy) requires the slower grid method.
"""

CUDA_STRATEGIES = {}


def _register():
    """Register available CUDA strategy kernels."""
    try:
        from backtest.cuda_strategies.consecutive_reverse import consecutive_reverse_kernel
        CUDA_STRATEGIES["ConsecutiveReverseMartingaleStrategy"] = {
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
        pass


_register()
