# Parameter Optimization Design

## Overview

Add parameter optimization capability to the backtesting framework, enabling automated exploration of strategy parameter combinations to find optimal configurations.

## Scope

- New module: `src/backtest/optimizer.py`
- New CLI subcommand: `python -m backtest optimize`
- New DB table: `optimize_results` in `data/reports.db`
- No changes to existing Engine, Strategy, Reporter, or DataFeed

## Parameter Space Definition

`ParamSpace` accepts a dict mapping parameter names to ranges:

```python
from backtest.optimizer import ParamSpace

space = ParamSpace({
    "DECISION_LEN": (20, 80, 10),                   # tuple(min, max, step) -> range
    "SHADOW_FACTOR": (1.5, 4.0, 0.5),               # float range
    "TOLERANCE_RATE": [0.005, 0.00618, 0.008],       # list -> explicit choices
})
```

Rules:
- `tuple (min, max, step)` — generates evenly-spaced values; int vs float inferred from inputs
- `list` — explicit set of allowed values
- Parameters not in the space retain the strategy class default values

`ParamSpace` exposes:
- `grid() -> list[dict]` — all combinations (cartesian product)
- `total_combinations -> int` — count of grid points
- `to_optuna_suggest(trial) -> dict` — maps to Optuna trial suggestions

## Optimizer Interface

Both optimizers share a common pattern:

```python
class BaseOptimizer:
    def __init__(self, db_path, strategy_path, symbol, interval, start, end,
                 balance, leverage, param_space, objective, n_jobs):
        ...

    def run(self) -> OptimizeResult:
        ...
```

### GridSearchOptimizer

- Generates all parameter combinations from `ParamSpace.grid()`
- Dispatches to `multiprocessing.Pool(n_jobs)` workers
- Each worker: creates strategy subclass with injected params, runs BacktestEngine, returns report
- Collects all results, sorts by objective

### OptunaOptimizer

- Requires `optuna` as optional dependency (import guarded with helpful error message)
- Additional param: `n_trials=100`
- Creates Optuna study, objective function calls BacktestEngine per trial
- Uses `ParamSpace` to map ranges to `trial.suggest_int/suggest_float/suggest_categorical`
- Supports parallel via Optuna's built-in `n_jobs` on study.optimize()

### Parameter Injection

Each trial dynamically creates a strategy subclass with overridden class attributes:

```python
def _make_strategy(base_class: type, params: dict) -> type:
    return type(f"{base_class.__name__}_trial", (base_class,), params)
```

This preserves the original strategy class and avoids mutation.

## OptimizeResult

```python
@dataclass
class OptimizeResult:
    best_params: dict          # parameter combination with highest score
    best_score: float          # objective value of best trial
    all_trials: list[dict]     # [{params, score, report}, ...] sorted by score desc
    objective: str             # metric name used
    total_trials: int          # number of trials executed
    elapsed_seconds: float     # wall-clock time
```

## Objective Function

Default: `sharpe_ratio`. User selects via `--objective` flag.

Supported built-in objectives (all from Reporter output):
- `sharpe_ratio`
- `net_return`
- `sortino_ratio`
- `profit_factor`
- `win_rate`

The optimizer extracts `report[objective]` as the score for each trial.

## CLI Subcommand

```bash
python -m backtest optimize \
    --strategy strategies/shadow_power_backtest.py \
    --symbol BTCUSDT --interval 15m \
    --start 2024-01-01 --end 2024-12-31 \
    --balance 1000 --leverage 49 \
    --params "DECISION_LEN=20:80:10,SHADOW_FACTOR=1.5:4.0:0.5,TOLERANCE_RATE=0.005|0.00618|0.008" \
    --objective sharpe_ratio \
    --method grid \
    --n-jobs 4 \
    --top 10
```

Parameter syntax:
- `KEY=min:max:step` — range (colon-separated)
- `KEY=v1|v2|v3` — explicit choices (pipe-separated)
- Multiple params comma-separated

Additional flags for Optuna:
- `--method optuna --n-trials 200`

## Terminal Output

```
Optimizing ShadowPowerStrategy: 120 combinations, 4 workers ...
[########################################] 120/120 (3m 42s)

Best Sharpe Ratio = 2.31

 Rank | DECISION_LEN | SHADOW_FACTOR | TOLERANCE_RATE | Sharpe | Return  | MaxDD
------+--------------+---------------+----------------+--------+---------+-------
    1 |           40 |           2.5 |        0.00618 |   2.31 |  +185%  | -12.3%
    2 |           50 |           3.0 |        0.00618 |   2.15 |  +162%  | -14.1%
    3 |           30 |           2.0 |        0.00500 |   1.98 |  +143%  | -11.8%

Results saved to database (3 rows). View with:
    SELECT * FROM optimize_results WHERE strategy='ShadowPowerStrategy' ORDER BY score DESC;
```

Progress bar via simple print (no external dependency like tqdm).

## Database Schema

New table in `data/reports.db`:

```sql
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
);
```

- `params_json`: JSON object of the parameter combination for this trial
- `report_json`: Full Reporter output (metrics + equity curve + trades) as JSON
- `score`: Extracted objective value as a real column for efficient sorting/filtering
- Each trial is one row; a single optimization run inserts `top N` rows (configurable, default all)

## File Structure

```
src/backtest/
├── optimizer.py          # ParamSpace, BaseOptimizer, GridSearchOptimizer,
│                         # OptunaOptimizer, OptimizeResult, _run_single_trial()
└── __main__.py           # + cmd_optimize() + optimize subparser
```

## Dependencies

- **No new required dependencies** — Grid search uses stdlib only (multiprocessing, itertools)
- **Optional**: `optuna` — only imported when `--method optuna` is used; clear error message if missing

## Design Constraints

- Does NOT modify BacktestEngine, BaseStrategy, SimExchange, Reporter, or DataFeed
- Strategy class is never mutated; each trial uses a dynamically-created subclass
- Worker function is a module-level function (pickle-compatible for multiprocessing)
- Database writes happen after all trials complete (single batch insert)
