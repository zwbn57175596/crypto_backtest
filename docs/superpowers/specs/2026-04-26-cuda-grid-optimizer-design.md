# CUDA Grid Optimizer Design

## Overview

Add GPU-accelerated grid search optimization using Numba CUDA, enabling 100万+ parameter combinations to run on RTX 3080 (10GB VRAM). The design provides a reusable exchange device-function layer so new strategies only need to implement a CUDA kernel.

## Target Environment

- **GPU**: NVIDIA RTX 3080 (10GB VRAM, 8704 CUDA cores, Compute Capability 8.6)
- **OS**: Windows 11
- **Tech**: Numba CUDA (`numba.cuda`)
- **Python**: 3.11+

## Architecture

### File Structure

```
src/backtest/
├── cuda_exchange.py              # Device functions: fill_order, settle_funding, check_liquidation
├── cuda_strategies/
│   ├── __init__.py               # Strategy registry (name -> kernel mapping)
│   └── consecutive_reverse.py    # CUDA kernel for ConsecutiveReverse
├── cuda_runner.py                # CudaGridOptimizer class + batch scheduler
```

### Component Responsibilities

**`cuda_exchange.py`** — Reusable exchange logic as `@cuda.jit(device=True)` functions:
- `device_fill_order(...)` — Process order fill: open/close/add position, compute PnL, update balance
- `device_settle_funding(...)` — Apply funding payment at 0/8/16 UTC hours
- `device_check_liquidation(...)` — Check margin ratio and force-close if needed
- All functions operate on scalar values (no arrays), same logic as `numba_simulate.py`

**`cuda_strategies/consecutive_reverse.py`** — The CUDA kernel:
- One `@cuda.jit` global kernel: `consecutive_reverse_kernel(bars, params, results, n_bars, n_combos)`
- Each thread picks its combo index via `cuda.grid(1)`, runs the full bar loop calling device functions
- Writes 8 metric floats to `results[thread_idx]`

**`cuda_runner.py`** — Orchestration:
- `CudaGridOptimizer` class with same interface as `NumbaGridOptimizer`
- Handles: GPU detection, memory estimation, batch splitting, kernel launch, result collection
- Integrates with existing `OptimizeResult`, `save_results`, `save_top_reports`

### Strategy Registry

```python
# cuda_strategies/__init__.py
CUDA_STRATEGIES = {
    "ConsecutiveReverseStrategy": {
        "kernel": consecutive_reverse_kernel,
        "param_order": [
            "CONSECUTIVE_THRESHOLD",
            "POSITION_MULTIPLIER",
            "INITIAL_POSITION_PCT",
            "PROFIT_CANDLE_THRESHOLD",
            "LEVERAGE",
        ],
    },
}
```

The optimizer looks up the strategy class name in this registry. `param_order` defines the column mapping for the params array passed to the kernel. If not found, falls back to `numba-grid` or `grid` method with a warning.

## Data Layout (GPU Memory)

### Input Arrays

| Array | Shape | Dtype | Size (8760 bars, 500K combos) |
|-------|-------|-------|-------------------------------|
| `bars` | `(N_bars, 6)` | float64 | ~420 KB |
| `params` | `(N_combos, N_params)` | float64 | ~20 MB (5 params) |

### Output Array

| Array | Shape | Dtype | Size |
|-------|-------|-------|------|
| `results` | `(N_combos, 8)` | float64 | ~32 MB |

### Per-Thread State

All strategy/exchange state lives in registers/local memory — no shared memory needed:
- Exchange: balance, pos_side, pos_qty, pos_entry, pos_margin, pos_unrealized_pnl (6 floats)
- Strategy: consecutive_count, streak_direction, profit_candle_count (3 ints)
- Pending orders: n_pending, 2x (side, qty) (5 values)
- Metrics: total_trades, wins, losses, total_profit, total_loss, peak_equity, max_dd, sum_ret, sum_ret_sq, sum_down_sq, n_returns, n_downside, prev_equity (13 values)
- Total: ~27 registers per thread (~216 bytes)

### Memory Budget (RTX 3080, 10GB)

- Reserved for OS/driver: ~1.5 GB
- Available: ~8.5 GB
- Per combo overhead: bars(shared) + params(40B) + results(64B) + registers(216B) = ~320 bytes
- **Safe batch size**: 500,000 combos per batch (~160 MB data + register pressure)
- Auto-detection: query `cuda.current_context().get_memory_info()` and compute dynamically

## Batch Processing

```
Total combos: 1,000,000
Batch size: 500,000 (auto-calculated from free VRAM)

Batch 1: combos[0:500000]
  1. Copy params[0:500K] to GPU
  2. Allocate results[500K, 8] on GPU
  3. Launch kernel: blocks=ceil(500K/256), threads=256
  4. Copy results back to CPU
  5. Free GPU params & results arrays

Batch 2: combos[500000:1000000]
  ... same ...

Merge & sort all results on CPU
```

The `bars` array is copied to GPU once and reused across all batches.

## Kernel Design

### Thread Mapping

```
grid_size = ceil(n_combos / block_size)
block_size = 256  (tunable, 128-512 range)
thread_idx = cuda.grid(1)
if thread_idx >= n_combos: return
```

### Kernel Pseudocode

```python
@cuda.jit
def consecutive_reverse_kernel(bars, params, results, n_bars, n_combos, exchange_leverage, commission_rate, funding_rate, maintenance_margin, initial_balance):
    idx = cuda.grid(1)
    if idx >= n_combos:
        return

    # Read this thread's parameters
    threshold = int(params[idx, 0])
    multiplier = params[idx, 1]
    initial_pct = params[idx, 2]
    profit_threshold = int(params[idx, 3])
    sizing_leverage = int(params[idx, 4])

    # Initialize state (all in registers)
    balance = initial_balance
    pos_side = 0  # 0=none, 1=long, 2=short
    # ... (same as numba_simulate.simulate)

    # Main bar loop
    for i in range(n_bars):
        # 1. Settle funding — call device_settle_funding()
        # 2. Match pending orders — call device_fill_order()
        # 3. Update unrealized PnL
        # 4. Check liquidation — call device_check_liquidation()
        # 5. Record equity & metrics
        # 6. Strategy logic (consecutive reverse)

    # Write 8 metrics to results
    results[idx, 0] = net_return
    results[idx, 1] = annual_return
    # ... etc
```

### Parameter Mapping

The `params` array columns are ordered by the strategy registry:

| Column | Parameter | Type |
|--------|-----------|------|
| 0 | CONSECUTIVE_THRESHOLD | int (stored as float64) |
| 1 | POSITION_MULTIPLIER | float |
| 2 | INITIAL_POSITION_PCT | float |
| 3 | PROFIT_CANDLE_THRESHOLD | int (stored as float64) |
| 4 | LEVERAGE | int (stored as float64) |

## CLI Integration

Add `cuda-grid` to `--method` choices:

```bash
python -m backtest optimize \
    --strategy strategies/consecutive_reverse.py \
    --symbol BTCUSDT --interval 1h \
    --start 2024-01-01 --end 2024-12-31 \
    --balance 1000 --leverage 50 \
    --params "CONSECUTIVE_THRESHOLD=3:8:1,POSITION_MULTIPLIER=1.0:1.5:0.1,INITIAL_POSITION_PCT=0.005:0.03:0.005,PROFIT_CANDLE_THRESHOLD=1:5:1" \
    --method cuda-grid --objective sharpe_ratio
```

### __main__.py Changes

- Add `"cuda-grid"` to `--method` choices
- Add `CudaGridOptimizer` import and instantiation branch in `cmd_optimize()`
- Same constructor interface as `NumbaGridOptimizer` (minus `n_jobs`)

## Error Handling

- **No CUDA device**: Print warning, suggest `numba-grid` fallback, exit with error
- **Insufficient VRAM**: Auto-reduce batch size; if even 1 combo doesn't fit, error out
- **Numba CUDA not available**: Catch `ImportError` / `CudaSupportError`, suggest installing `cudatoolkit`
- **Kernel errors**: Catch `numba.cuda.cudadrv.driver.CudaAPIError`, report and abort

## Testing Strategy

- **Unit test device functions**: Use `@cuda.jit` wrapper to call device functions from a test kernel, compare output with `numba_simulate._fill_order`
- **Single-combo correctness**: Run CUDA kernel with 1 combo, compare all 8 metrics against `numba_simulate.simulate()` with same params — must match within float64 tolerance (1e-6 relative)
- **Multi-combo correctness**: Run N combos via CUDA and Numba CPU, compare all results
- **Batch boundary**: Test that batch splitting produces identical results to single-batch
- **Edge cases**: 0 bars, 1 bar, 0 combos, combo count not divisible by block_size

## Performance Expectations

Based on RTX 3080 specs and the workload characteristics:
- Each thread: ~8760 iterations × ~50 FLOPs/iter ≈ 438K FLOPs
- RTX 3080 FP64: ~238 GFLOPS (1/32 of FP32)
- Theoretical: 238G / 438K ≈ 543K combos/sec
- Realistic (memory bound, register pressure): ~100K-300K combos/sec
- **100万 combos: ~3-10 seconds** (vs numba-grid CPU: minutes)

Note: FP64 is critical for financial calculations. RTX 3080 FP64 rate is 1/32 of FP32. If precision allows, we could explore FP32 for another 32x boost, but default to FP64 for correctness.
