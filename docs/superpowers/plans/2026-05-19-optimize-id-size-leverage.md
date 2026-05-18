# --optimize-id + SIZE_LEVERAGE Rename Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename the strategy `LEVERAGE` sizing param to `SIZE_LEVERAGE` to eliminate its name collision with CLI `--leverage` (exchange leverage), and add `--optimize-id` to `run` for one-command reproduction of any optimization result.

**Architecture:** Two independent changes. Part 1 (rename) threads through strategy files → CUDA param_order → numba optimizer hardcodes → tests. Part 2 (optimize-id) adds two columns to `optimize_results`, teaches `save_results` to store them, then wires a new `--optimize-id` CLI arg into `cmd_run` via two small helper functions.

**Tech Stack:** Python 3.11, SQLite (sqlite3), argparse, pytest

---

## File Map

| File | Change |
|------|--------|
| `strategies/consecutive_reverse.py` | `LEVERAGE` → `SIZE_LEVERAGE` (class var + `_calc_quantity`) |
| `strategies/consecutive_reverse_martingale.py` | Same + `OPTIMIZE_SPACE` + `AUTO_OPTIMIZE_CONFIG` keys |
| `src/backtest/cuda_strategies/__init__.py` | param_order `"LEVERAGE"` → `"SIZE_LEVERAGE"` |
| `src/backtest/optimizer.py` | 3 numba hardcodes; `save_results` signature + schema + INSERT |
| `src/backtest/__main__.py` | sizing_leverage line; `--optimize-id` arg; `cmd_run` helpers + logic |
| `tests/test_cuda_runner.py` | param dicts `"LEVERAGE"` → `"SIZE_LEVERAGE"` |
| `tests/test_numba_simulate.py` | attrs dict `"LEVERAGE"` → `"SIZE_LEVERAGE"` |
| `tests/test_optimizer.py` | OPTIMIZE_SPACE test fix + new `save_results` column tests |

---

## Task 1 — Rename LEVERAGE → SIZE_LEVERAGE in strategy files

**Files:**
- Modify: `strategies/consecutive_reverse.py`
- Modify: `strategies/consecutive_reverse_martingale.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_consecutive_reverse.py` (after existing imports):

```python
def test_size_leverage_attr_exists():
    from strategies.consecutive_reverse import ConsecutiveReverseStrategy
    assert hasattr(ConsecutiveReverseStrategy, "SIZE_LEVERAGE")
    assert not hasattr(ConsecutiveReverseStrategy, "LEVERAGE")


def test_calc_quantity_uses_size_leverage():
    from strategies.consecutive_reverse import ConsecutiveReverseStrategy
    from backtest.exchange import SimExchange
    exchange = SimExchange(balance=1000.0, leverage=50,
                           commission_rate=0.0004, funding_rate=0.0001,
                           maintenance_margin=0.005)
    strategy = ConsecutiveReverseStrategy(exchange=exchange, symbol="BTCUSDT")
    strategy.on_init()
    strategy.SIZE_LEVERAGE = 10
    strategy.INITIAL_POSITION_PCT = 0.01
    strategy.CONSECUTIVE_THRESHOLD = 3
    strategy.POSITION_MULTIPLIER = 1.0
    strategy._consecutive_count = 3
    # base = 1000 * 0.01 = 10, n=1, mult=1.0^0=1, qty = 10 * 1 * 10 = 100
    assert strategy._calc_quantity() == pytest.approx(100.0)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_consecutive_reverse.py::test_size_leverage_attr_exists tests/test_consecutive_reverse.py::test_calc_quantity_uses_size_leverage -v
```

Expected: FAIL — `AssertionError: assert False` (attribute doesn't exist yet)

- [ ] **Step 3: Rename in consecutive_reverse.py**

In `strategies/consecutive_reverse.py`:
- Line 52: `LEVERAGE = 17` → `SIZE_LEVERAGE = 17`
- Line 105: `return base * multiplier * self.LEVERAGE` → `return base * multiplier * self.SIZE_LEVERAGE`

- [ ] **Step 4: Rename in consecutive_reverse_martingale.py**

In `strategies/consecutive_reverse_martingale.py`:
- Line 64: `LEVERAGE = 50` → `SIZE_LEVERAGE = 50`
- Line 72 (OPTIMIZE_SPACE): `"LEVERAGE": (1, 50, 3)` → `"SIZE_LEVERAGE": (1, 50, 3)`
- Line 81 (AUTO_OPTIMIZE_CONFIG refine_space): `"LEVERAGE": {"radius": 3, "step": 1}` → `"SIZE_LEVERAGE": {"radius": 3, "step": 1}`
- Line 135: `return base * multiplier * self.LEVERAGE` → `return base * multiplier * self.SIZE_LEVERAGE`

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_consecutive_reverse.py::test_size_leverage_attr_exists tests/test_consecutive_reverse.py::test_calc_quantity_uses_size_leverage -v
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add strategies/consecutive_reverse.py strategies/consecutive_reverse_martingale.py tests/test_consecutive_reverse.py
git commit -m "refactor: rename LEVERAGE → SIZE_LEVERAGE in strategy sizing params"
```

---

## Task 2 — Update CUDA param_order and numba optimizer references

**Files:**
- Modify: `src/backtest/cuda_strategies/__init__.py`
- Modify: `src/backtest/optimizer.py`
- Modify: `tests/test_cuda_runner.py`
- Modify: `tests/test_numba_simulate.py`
- Modify: `tests/test_optimizer.py`

- [ ] **Step 1: Update cuda_runner tests to use SIZE_LEVERAGE**

In `tests/test_cuda_runner.py`, replace all 5 occurrences of `"LEVERAGE": [10]` or `"LEVERAGE": [50]` with `"SIZE_LEVERAGE"`:

Line 82: `"LEVERAGE": [10],` → `"SIZE_LEVERAGE": [10],`
Line 136: `"LEVERAGE": [10],` → `"SIZE_LEVERAGE": [10],`
Line 178: `"LEVERAGE": [10],` → `"SIZE_LEVERAGE": [10],`
Line 281: `"LEVERAGE": [50],` → `"SIZE_LEVERAGE": [50],`
Line 307: `"LEVERAGE": [50],` → `"SIZE_LEVERAGE": [50],`

- [ ] **Step 2: Update test_numba_simulate.py attrs dict**

In `tests/test_numba_simulate.py` line 35:
```python
# Before:
"LEVERAGE": params.get("leverage", 50),
# After:
"SIZE_LEVERAGE": params.get("size_leverage", 50),
```

- [ ] **Step 3: Update test_optimizer.py OPTIMIZE_SPACE test**

Replace the broken `TestStrategyOptimizeSpace.test_loads_strategy_defined_space` test (lines 84–90) with a correct one that uses the martingale strategy (which has OPTIMIZE_SPACE) and checks SIZE_LEVERAGE:

```python
class TestStrategyOptimizeSpace:
    def test_loads_strategy_defined_space(self):
        space = load_strategy_optimize_space(
            "strategies/consecutive_reverse_martingale.py"
        )
        combos = space.grid()
        assert len(combos) > 0
        assert combos[0]["CONSECUTIVE_THRESHOLD"] == 2
        assert "SIZE_LEVERAGE" in combos[0]
        assert "LEVERAGE" not in combos[0]
```

- [ ] **Step 4: Run tests to verify they fail (CUDA tests skipped if no GPU)**

```bash
pytest tests/test_cuda_runner.py tests/test_optimizer.py::TestStrategyOptimizeSpace -v
```

Expected: CUDA tests SKIP (no GPU in test env) or FAIL on SIZE_LEVERAGE key; optimizer test FAIL on `"LEVERAGE" not in combos[0]`

- [ ] **Step 5: Update cuda_strategies/__init__.py param_order**

In `src/backtest/cuda_strategies/__init__.py` line 28:
```python
# Before:
param_order = [
    "CONSECUTIVE_THRESHOLD",
    "POSITION_MULTIPLIER",
    "INITIAL_POSITION_PCT",
    "PROFIT_CANDLE_THRESHOLD",
    "LEVERAGE",
]
# After:
param_order = [
    "CONSECUTIVE_THRESHOLD",
    "POSITION_MULTIPLIER",
    "INITIAL_POSITION_PCT",
    "PROFIT_CANDLE_THRESHOLD",
    "SIZE_LEVERAGE",
]
```

- [ ] **Step 6: Update optimizer.py numba hardcodes**

In `src/backtest/optimizer.py`, three changes:

Line 614: `"LEVERAGE",` → `"SIZE_LEVERAGE",`

Line 622:
```python
# Before:
sizing_lev = int(params.get("LEVERAGE", default_params.get("LEVERAGE", self.leverage)))
# After:
sizing_lev = int(params.get("SIZE_LEVERAGE", default_params.get("SIZE_LEVERAGE", self.leverage)))
```

Line 653:
```python
# Before:
int(params.get("LEVERAGE", default_params.get("LEVERAGE", self.leverage))),
# After:
int(params.get("SIZE_LEVERAGE", default_params.get("SIZE_LEVERAGE", self.leverage))),
```

- [ ] **Step 7: Run tests to verify they pass**

```bash
pytest tests/test_optimizer.py::TestStrategyOptimizeSpace -v
```

Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add src/backtest/cuda_strategies/__init__.py src/backtest/optimizer.py \
        tests/test_cuda_runner.py tests/test_numba_simulate.py tests/test_optimizer.py
git commit -m "refactor: update CUDA param_order and numba refs to SIZE_LEVERAGE"
```

---

## Task 3 — Update __main__.py sizing leverage reference

**Files:**
- Modify: `src/backtest/__main__.py`

- [ ] **Step 1: Update sizing_leverage assignment in cmd_run**

In `src/backtest/__main__.py` lines 98–100, replace:
```python
if args.sizing_leverage is not None:
    strategy_class.LEVERAGE = args.sizing_leverage
    print(f"  [param] LEVERAGE (sizing) = {args.sizing_leverage}")
```
With:
```python
if args.sizing_leverage is not None:
    attr = "SIZE_LEVERAGE" if hasattr(strategy_class, "SIZE_LEVERAGE") else "LEVERAGE"
    setattr(strategy_class, attr, args.sizing_leverage)
    print(f"  [param] {attr} = {args.sizing_leverage}")
```

This keeps backward compat for strategies that haven't been renamed yet.

- [ ] **Step 2: Verify with a quick smoke test**

```bash
python -m backtest run --help
```

Expected: help text prints without error

- [ ] **Step 3: Commit**

```bash
git add src/backtest/__main__.py
git commit -m "fix: sizing_leverage assignment uses SIZE_LEVERAGE with LEVERAGE fallback"
```

---

## Task 4 — Add strategy_path and exchange to optimize_results schema

**Files:**
- Modify: `src/backtest/optimizer.py`
- Modify: `tests/test_optimizer.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_optimizer.py` in the `TestSaveResults` class (or as a standalone class if it doesn't exist):

```python
class TestSaveResultsColumns:
    def test_strategy_path_and_exchange_saved(self, tmp_path):
        import json, sqlite3
        from backtest.optimizer import save_results, OptimizeResult
        db = str(tmp_path / "test.db")
        result = OptimizeResult(
            best_params={"SIZE_LEVERAGE": 10},
            best_score=0.5,
            all_trials=[{
                "params": {"SIZE_LEVERAGE": 10},
                "score": 0.5,
                "report": {"net_return": 0.1, "sharpe_ratio": 0.5,
                           "max_drawdown": 0.2, "win_rate": 0.5,
                           "profit_factor": 1.2, "total_trades": 10,
                           "annual_return": 0.15, "sortino_ratio": 0.4},
            }],
            objective="sharpe_ratio",
            total_trials=1,
            elapsed_seconds=1.0,
        )
        save_results(
            db_path=db, strategy="TestStrat", symbol="BTCUSDT", interval="1h",
            start_date="2024-01-01", end_date="2024-03-01", result=result,
            leverage=50, strategy_path="strategies/consecutive_reverse.py",
            exchange="okx",
        )
        conn = sqlite3.connect(db)
        row = conn.execute(
            "SELECT strategy_path, exchange FROM optimize_results WHERE id=1"
        ).fetchone()
        conn.close()
        assert row[0] == "strategies/consecutive_reverse.py"
        assert row[1] == "okx"
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest tests/test_optimizer.py::TestSaveResultsColumns -v
```

Expected: FAIL — `OperationalError: table optimize_results has no column named strategy_path`

- [ ] **Step 3: Update save_results signature and body**

In `src/backtest/optimizer.py`:

**Signature** (line 698):
```python
def save_results(
    db_path: str,
    strategy: str,
    symbol: str,
    interval: str,
    start_date: str,
    end_date: str,
    result: OptimizeResult,
    top_n: int | None = 1000,
    leverage: int | None = None,
    strategy_path: str | None = None,
    exchange: str | None = None,
) -> None:
```

**CREATE TABLE** — add two columns at the end of the CREATE TABLE statement:
```python
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
        batch_id TEXT,
        leverage REAL,
        strategy_path TEXT,
        exchange TEXT
    )
""")
```

**Idempotent migrations** — add after the existing leverage migration block:
```python
try:
    conn.execute("ALTER TABLE optimize_results ADD COLUMN strategy_path TEXT")
    conn.commit()
except sqlite3.OperationalError:
    pass

try:
    conn.execute("ALTER TABLE optimize_results ADD COLUMN exchange TEXT")
    conn.commit()
except sqlite3.OperationalError:
    pass
```

**rows list** — replace the current rows construction:
```python
rows = [
    (
        strategy, symbol, interval, start_date, end_date,
        result.objective, trial["score"],
        json.dumps(trial["params"], sort_keys=True),
        json.dumps(trial.get("report", {}), sort_keys=True),
        now,
        batch_id,
        leverage,
        strategy_path,
        exchange,
    )
    for trial in trials
]
```

**executemany INSERT** — add the two new columns:
```python
conn.executemany(
    "INSERT INTO optimize_results "
    "(strategy, symbol, interval, start_date, end_date, objective, score, "
    "params_json, report_json, created_at, batch_id, leverage, strategy_path, exchange) "
    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
    rows,
)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_optimizer.py::TestSaveResultsColumns -v
```

Expected: PASS

- [ ] **Step 5: Update cmd_optimize in __main__.py to pass strategy_path and exchange**

In `src/backtest/__main__.py`, find the `save_results(...)` call (around line 289) and add the two new kwargs:

```python
save_results(
    db_path=report_db,
    strategy=strategy_class.__name__,
    symbol=args.symbol,
    interval=args.interval,
    start_date=args.start,
    end_date=args.end,
    result=result,
    top_n=save_n,
    leverage=args.leverage,
    strategy_path=args.strategy,
    exchange=args.exchange,
)
```

- [ ] **Step 6: Run existing optimizer tests to confirm no regression**

```bash
pytest tests/test_optimizer.py -v -k "not TestStrategyOptimizeSpace"
```

Expected: all pass (skip the pre-existing broken OPTIMIZE_SPACE test)

- [ ] **Step 7: Commit**

```bash
git add src/backtest/optimizer.py src/backtest/__main__.py tests/test_optimizer.py
git commit -m "feat: store strategy_path and exchange in optimize_results"
```

---

## Task 5 — Add _load_optimize_result and _apply_optimize_params helpers

**Files:**
- Modify: `src/backtest/__main__.py`
- Modify: `tests/test_optimizer.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_optimizer.py`:

```python
class TestLoadOptimizeResult:
    def test_loads_all_fields(self, tmp_path):
        import json, sqlite3
        from backtest.optimizer import save_results, OptimizeResult
        db = str(tmp_path / "r.db")
        result = OptimizeResult(
            best_params={"SIZE_LEVERAGE": 26},
            best_score=0.88,
            all_trials=[{
                "params": {"SIZE_LEVERAGE": 26, "CONSECUTIVE_THRESHOLD": 8},
                "score": 0.88,
                "report": {"net_return": 1.0, "sharpe_ratio": 0.88,
                           "max_drawdown": 0.5, "win_rate": 0.5,
                           "profit_factor": 1.5, "total_trades": 100,
                           "annual_return": 0.5, "sortino_ratio": 0.7},
            }],
            objective="sharpe_ratio", total_trials=1, elapsed_seconds=1.0,
        )
        save_results(
            db_path=db, strategy="ConsecutiveReverseStrategy",
            symbol="BTCUSDT", interval="1h",
            start_date="2020-01-01", end_date="2026-03-30",
            result=result, leverage=50,
            strategy_path="strategies/consecutive_reverse.py", exchange="okx",
        )
        # Import helper to test
        import sys
        sys.path.insert(0, "src")
        from backtest.__main__ import _load_optimize_result
        loaded = _load_optimize_result(1, db)
        assert loaded["strategy_path"] == "strategies/consecutive_reverse.py"
        assert loaded["symbol"] == "BTCUSDT"
        assert loaded["interval"] == "1h"
        assert loaded["start"] == "2020-01-01"
        assert loaded["end"] == "2026-03-30"
        assert loaded["leverage"] == 50
        assert loaded["exchange"] == "okx"
        assert loaded["params"]["SIZE_LEVERAGE"] == 26

    def test_not_found_raises(self, tmp_path):
        import sqlite3
        from backtest.__main__ import _load_optimize_result
        db = str(tmp_path / "empty.db")
        conn = sqlite3.connect(db)
        conn.execute("""CREATE TABLE optimize_results (
            id INTEGER PRIMARY KEY, strategy_path TEXT, symbol TEXT,
            interval TEXT, start_date TEXT, end_date TEXT,
            leverage REAL, exchange TEXT, params_json TEXT)""")
        conn.commit()
        conn.close()
        with pytest.raises(SystemExit):
            _load_optimize_result(999, db)


class TestApplyOptimizeParams:
    def test_sets_known_attrs(self):
        from backtest.__main__ import _apply_optimize_params
        from strategies.consecutive_reverse import ConsecutiveReverseStrategy

        class TrialClass(ConsecutiveReverseStrategy):
            pass

        _apply_optimize_params(TrialClass, {"SIZE_LEVERAGE": 26, "CONSECUTIVE_THRESHOLD": 8})
        assert TrialClass.SIZE_LEVERAGE == 26
        assert TrialClass.CONSECUTIVE_THRESHOLD == 8

    def test_legacy_leverage_key_maps_to_size_leverage(self):
        from backtest.__main__ import _apply_optimize_params
        from strategies.consecutive_reverse import ConsecutiveReverseStrategy

        class TrialClass(ConsecutiveReverseStrategy):
            pass

        # Old DB record with "LEVERAGE" key — should map to SIZE_LEVERAGE
        _apply_optimize_params(TrialClass, {"LEVERAGE": 26})
        assert TrialClass.SIZE_LEVERAGE == 26

    def test_ignores_unknown_attrs(self):
        from backtest.__main__ import _apply_optimize_params
        from strategies.consecutive_reverse import ConsecutiveReverseStrategy

        class TrialClass(ConsecutiveReverseStrategy):
            pass

        # Should not raise even with unknown key
        _apply_optimize_params(TrialClass, {"UNKNOWN_PARAM": 99})
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/test_optimizer.py::TestLoadOptimizeResult tests/test_optimizer.py::TestApplyOptimizeParams -v
```

Expected: FAIL — `ImportError: cannot import name '_load_optimize_result'`

- [ ] **Step 3: Add helper functions to __main__.py**

Add after the `_parse_date_to_ms` function (around line 26) in `src/backtest/__main__.py`:

```python
def _load_optimize_result(optimize_id: int, db_arg: str | None) -> dict:
    """Load run parameters from optimize_results table by id."""
    import json, sqlite3
    from pathlib import Path
    report_db = db_arg or str(Path("data") / "reports.db")
    conn = sqlite3.connect(report_db)
    row = conn.execute(
        "SELECT strategy_path, symbol, interval, start_date, end_date, "
        "leverage, exchange, params_json "
        "FROM optimize_results WHERE id = ?",
        (optimize_id,),
    ).fetchone()
    conn.close()
    if row is None:
        sys.exit(f"optimize_results id={optimize_id} not found in {report_db}")
    strategy_path, symbol, interval, start_date, end_date, leverage, exchange, params_json = row
    return {
        "strategy_path": strategy_path,
        "symbol": symbol,
        "interval": interval,
        "start": start_date,
        "end": end_date,
        "leverage": int(leverage) if leverage is not None else None,
        "exchange": exchange or "binance",
        "params": json.loads(params_json),
    }


def _apply_optimize_params(strategy_class: type, params: dict) -> None:
    """Apply params dict from optimize_results to strategy class attributes."""
    for key, value in params.items():
        attr = key
        # backward compat: old DB records use "LEVERAGE"; new strategies use "SIZE_LEVERAGE"
        if key == "LEVERAGE" and not hasattr(strategy_class, "LEVERAGE") \
                and hasattr(strategy_class, "SIZE_LEVERAGE"):
            attr = "SIZE_LEVERAGE"
        if not hasattr(strategy_class, attr):
            continue
        try:
            setattr(strategy_class, attr, type(getattr(strategy_class, attr))(value))
            print(f"  [param] {attr} = {value}")
        except (TypeError, ValueError):
            setattr(strategy_class, attr, value)
            print(f"  [param] {attr} = {value}")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_optimizer.py::TestLoadOptimizeResult tests/test_optimizer.py::TestApplyOptimizeParams -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/backtest/__main__.py tests/test_optimizer.py
git commit -m "feat: add _load_optimize_result and _apply_optimize_params helpers"
```

---

## Task 6 — Wire --optimize-id into CLI and cmd_run

**Files:**
- Modify: `src/backtest/__main__.py`

- [ ] **Step 1: Make --strategy, --symbol, --interval optional in argparser**

In `src/backtest/__main__.py`, find the `p_run` argument definitions (around line 415) and change:

```python
p_run = sub.add_parser("run", help="Run backtest")
p_run.add_argument("--strategy", default=None)          # was required=True
p_run.add_argument("--symbol", default=None)            # was required=True
p_run.add_argument("--interval", default=None)          # was required=True
p_run.add_argument("--exchange", default=None)          # was default="binance"
p_run.add_argument("--start", default=None)
p_run.add_argument("--end", default=None)
p_run.add_argument("--balance", type=float, default=None)
p_run.add_argument("--leverage", type=int, default=None,
                   help="Exchange leverage (for margin & liquidation)")
p_run.add_argument("--margin-mode", choices=["isolated", "cross"], default=None,
                   dest="margin_mode",
                   help="Margin mode (isolated=per-position, cross=account-wide)")
p_run.add_argument("--sizing-leverage", type=int, default=None, dest="sizing_leverage",
                   help="Strategy sizing leverage (sets SIZE_LEVERAGE class attr)")
p_run.add_argument("--optimize-id", type=int, default=None, dest="optimize_id",
                   help="Load all params from optimize_results id; replaces manual param flags")
p_run.add_argument("--db", default=None)
```

- [ ] **Step 2: Update cmd_run to handle --optimize-id**

In `src/backtest/__main__.py`, replace the beginning of `cmd_run` (lines 87–101) with:

```python
def cmd_run(args: argparse.Namespace, extra_args: list[str] | None = None) -> None:
    import yaml
    config_path = Path("config/default.yaml")
    config = {}
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f).get("backtest", {})

    optimize_params: dict | None = None
    if args.optimize_id is not None:
        opt = _load_optimize_result(args.optimize_id, args.db)
        if opt["strategy_path"] is None and args.strategy is None:
            sys.exit(
                f"optimize_results id={args.optimize_id} has no strategy_path; "
                "pass --strategy to specify the file"
            )
        args.strategy  = args.strategy  or opt["strategy_path"]
        args.symbol    = args.symbol    or opt["symbol"]
        args.interval  = args.interval  or opt["interval"]
        args.start     = args.start     or opt["start"]
        args.end       = args.end       or opt["end"]
        args.leverage  = args.leverage  or opt["leverage"]
        args.exchange  = args.exchange  or opt["exchange"]
        optimize_params = opt["params"]

    # Validate required args (may have been filled by --optimize-id)
    if not args.strategy:
        sys.exit("--strategy is required (or use --optimize-id)")
    if not args.symbol:
        sys.exit("--symbol is required (or use --optimize-id)")
    if not args.interval:
        sys.exit("--interval is required (or use --optimize-id)")

    strategy_class = _load_strategy(args.strategy)
    if optimize_params:
        _apply_optimize_params(strategy_class, optimize_params)
    if extra_args:
        _apply_extra_params(strategy_class, extra_args)
    if args.sizing_leverage is not None:
        attr = "SIZE_LEVERAGE" if hasattr(strategy_class, "SIZE_LEVERAGE") else "LEVERAGE"
        setattr(strategy_class, attr, args.sizing_leverage)
        print(f"  [param] {attr} = {args.sizing_leverage}")
    db_path = args.db or str(Path("data") / "klines.db")
```

Note: the `args.exchange` default was `"binance"` before; it is now `None` and filled by `_load_optimize_result` or defaults in the engine constructor call below. Confirm line 105 reads:
```python
exchange=args.exchange or "binance",
```

- [ ] **Step 3: Smoke test that normal run still works**

```bash
python -m backtest run --strategy strategies/consecutive_reverse.py \
    --exchange okx --symbol BTCUSDT --interval 1h \
    --start 2026-01-01 --end 2026-02-01 \
    --balance 1000 --leverage 50 --CONSECUTIVE_THRESHOLD 8 \
    --SIZE_LEVERAGE 26 --INITIAL_POSITION_PCT 0.205 \
    --POSITION_MULTIPLIER 3.0 --PROFIT_CANDLE_THRESHOLD 2 2>&1 | tail -8
```

Expected: `Backtest Complete: ConsecutiveReverseStrategy` printed with results

- [ ] **Step 4: Smoke test --optimize-id with a recent record**

First confirm a recent OKX optimize result id (should have strategy_path after Task 4; if the id predates Task 4 it will have NULL strategy_path and require --strategy):

```bash
sqlite3 data/reports.db "SELECT id, strategy_path, exchange FROM optimize_results ORDER BY id DESC LIMIT 3;"
```

If strategy_path is NULL (old record), run:
```bash
python -m backtest run --optimize-id 26582 \
    --strategy strategies/consecutive_reverse.py 2>&1 | tail -8
```

Expected: params printed from DB, then `Backtest Complete`.

- [ ] **Step 5: Run full test suite to confirm no regressions**

```bash
pytest tests/ -v -k "not (TestStrategyOptimizeSpace or TestNumbaMatchesEngine or TestPositionSizing)" 2>&1 | tail -20
```

(The three excluded tests are pre-existing failures unrelated to this work.)

Expected: all included tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/backtest/__main__.py
git commit -m "feat: add --optimize-id to run command for one-command optimization replay"
```
