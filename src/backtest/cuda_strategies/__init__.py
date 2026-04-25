"""CUDA strategy kernels for GPU-accelerated backtesting.

This module provides a registry of CUDA-compiled strategy kernels that can be
used for parallel backtesting with GPU acceleration.
"""

CUDA_STRATEGIES = {}


def _register():
    """Register available CUDA strategy kernels."""
    try:
        from backtest.cuda_strategies.consecutive_reverse import consecutive_reverse_kernel
        CUDA_STRATEGIES["ConsecutiveReverseStrategy"] = {
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
