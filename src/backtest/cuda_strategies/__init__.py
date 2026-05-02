"""CUDA strategy kernels for GPU-accelerated backtesting.

This module provides a registry of CUDA-compiled strategy kernels that can be
used for parallel backtesting with GPU acceleration.

Two ConsecutiveReverse variants are registered:
  - ConsecutiveReverseMartingaleStrategy: on loss candle, add to existing
    contrarian position up to current target size (martingale).
  - ConsecutiveReverseStrategy: on loss candle, close immediately and reopen
    a fresh contrarian position (matches strategies/consecutive_reverse.py).
"""

CUDA_STRATEGIES = {}


def _register():
    """Register available CUDA strategy kernels."""
    try:
        from backtest.cuda_strategies.consecutive_reverse import (
            consecutive_reverse_kernel,
            consecutive_reverse_close_reopen_kernel,
        )
        param_order = [
            "CONSECUTIVE_THRESHOLD",
            "POSITION_MULTIPLIER",
            "INITIAL_POSITION_PCT",
            "PROFIT_CANDLE_THRESHOLD",
            "LEVERAGE",
        ]
        CUDA_STRATEGIES["ConsecutiveReverseMartingaleStrategy"] = {
            "kernel": consecutive_reverse_kernel,
            "param_order": param_order,
        }
        CUDA_STRATEGIES["ConsecutiveReverseStrategy"] = {
            "kernel": consecutive_reverse_close_reopen_kernel,
            "param_order": param_order,
        }
    except Exception:
        pass


_register()
