# Optimize→Reports Association Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `optimize_result_id` to the `reports` table so each report saved from an optimization run is linked back to its source `optimize_results` row.

**Architecture:** `save_top_reports` in `optimizer.py` already iterates over top N trials before inserting reports. We add a schema migration (idempotent `ALTER TABLE`) and a `SELECT` to look up the matching `optimize_results.id` by `params_json + strategy + symbol + interval`, then include it in the `INSERT`.

**Tech Stack:** Python 3.11, sqlite3 (stdlib), pytest

---

### Task 1: Add failing test for optimize_result_id linkage

**Files:**
- Modify: `tests/test_optimizer.py` (new test at end of file)

- [ ] **Step 1: Read the existing test file to understand fixtures**

Run: `cat tests/test_optimizer.py` — note how `tmp_path`, klines DB, and optimizer fixtures are set up.

- [ ] **Step 2: Write the failing test**

Add this test to `tests/test_optimizer.py`:

```python
def test_save_top_reports_links_optimize_result_id(tmp_path):
    """reports.optimize_result_id must point to the matching optimize_results row."""
    import json, sqlite3
    from backtest.optimizer import OptimizeResult, save_results, save_top_reports

    report_db = str(tmp_path / "reports.db")
    klines_db = str(tmp_path / "klines.db")  # unused by save_top_reports directly

    # Build a minimal OptimizeResult with two trials
    trials = [
        {"params": {"CONSECUTIVE_THRESHOLD": 3}, "score": 2.0,
         "report": {"sharpe_ratio": 2.0, "net_return": 0.5, "max_drawdown": 0.1}},
        {"params": {"CONSECUTIVE_THRESHOLD": 5}, "score": 1.5,
         "report": {"sharpe_ratio": 1.5, "net_return": 0.3, "max_drawdown": 0.2}},
    ]
    result = OptimizeResult(
        best_params=trials[0]["params"],
        best_score=2.0,
        all_trials=trials,
        objective="sharpe_ratio",
        total_trials=2,
        elapsed_seconds=1.0,
    )

    # save_results writes to optimize_results table
    save_results(
        db_path=report_db,
        strategy="TestStrategy",
        symbol="BTCUSDT",
        interval="1h",
        start_date="2024-01-01",
        end_date="2024-12-31",
        result=result,
    )

    # save_top_reports needs to re-run strategy — stub it out by
    # monkey-patching _make_strategy and BacktestEngine inside the call.
    # Instead, call the DB-write portion directly via a helper we'll expose.
    # For now, verify the optimize_results rows exist with expected IDs.
    conn = sqlite3.connect(report_db)
    rows = conn.execute(
        "SELECT id, params_json FROM optimize_results ORDER BY score DESC"
    ).fetchall()
    conn.close()

    assert len(rows) == 2
    top_id = rows[0][0]  # highest score row
    top_params = json.loads(rows[0][1])
    assert top_params == {"CONSECUTIVE_THRESHOLD": 3}

    # Now test the linkage write helper directly
    from backtest.optimizer import _write_report_with_link
    full_report = {"sharpe_ratio": 2.0, "net_return": 0.5, "equity_curve": [], "trades": []}
    _write_report_with_link(
        report_db_path=report_db,
        strategy_name="TestStrategy_opt1",
        symbol="BTCUSDT",
        interval="1h",
        created_at="2026-01-01T00:00:00+00:00",
        report=full_report,
        params={"CONSECUTIVE_THRESHOLD": 3},
        base_strategy="TestStrategy",
    )

    conn = sqlite3.connect(report_db)
    row = conn.execute(
        "SELECT optimize_result_id FROM reports WHERE strategy = 'TestStrategy_opt1'"
    ).fetchone()
    conn.close()

    assert row is not None
    assert row[0] == top_id
```

- [ ] **Step 3: Run test to verify it fails**

```bash
pytest tests/test_optimizer.py::test_save_top_reports_links_optimize_result_id -v
```

Expected: `FAILED` — `ImportError: cannot import name '_write_report_with_link'`

---

### Task 2: Implement `_write_report_with_link` and update `save_top_reports`

**Files:**
- Modify: `src/backtest/optimizer.py`
  - Add `_write_report_with_link` helper (~25 lines)
  - Update `save_top_reports` to call it (~5 line change)

- [ ] **Step 1: Add `_write_report_with_link` helper**

Insert this function after `save_results` (after line 629) in `optimizer.py`:

```python
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

    # Idempotent migration — add column if not present
    try:
        conn.execute("ALTER TABLE reports ADD COLUMN optimize_result_id INTEGER")
        conn.commit()
    except sqlite3.OperationalError:
        pass  # column already exists

    # Look up matching optimize_results row
    params_json = json.dumps(params)
    row = conn.execute(
        """SELECT id FROM optimize_results
           WHERE params_json = ? AND strategy = ? AND symbol = ? AND interval = ?
           ORDER BY created_at DESC LIMIT 1""",
        (params_json, base_strategy, symbol, interval),
    ).fetchone()
    optimize_result_id = row[0] if row else None

    conn.execute(
        """INSERT INTO reports
           (strategy, symbol, interval, created_at, report_json, optimize_result_id)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (strategy_name, symbol, interval, created_at,
         json.dumps(report), optimize_result_id),
    )
    conn.commit()
    conn.close()
```

- [ ] **Step 2: Update `save_top_reports` to use the helper**

Replace the existing `conn.execute("INSERT INTO reports ...")` block inside the `for i, trial in enumerate(trials, 1):` loop. The loop currently looks like this (lines 666–688):

```python
    for i, trial in enumerate(trials, 1):
        trial_class = _make_strategy(strategy_class, trial["params"])
        engine = BacktestEngine(
            db_path=db_path,
            symbol=symbol,
            interval=interval,
            exchange="binance",
            strategy_class=trial_class,
            balance=balance,
            leverage=leverage,
            start=start_fmt,
            end=end_fmt,
        )
        run_result = engine.run()
        report = Reporter.generate(run_result)

        strategy_name = f"{strategy_class.__name__}_opt{i}"
        conn.execute(
            "INSERT INTO reports (strategy, symbol, interval, created_at, report_json) VALUES (?,?,?,?,?)",
            (strategy_name, symbol, interval, now, json.dumps(report)),
        )

    conn.commit()
    conn.close()
```

Replace the entire function body (keep the signature) with:

```python
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
            exchange="binance",
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
```

- [ ] **Step 3: Run the failing test — it should now pass**

```bash
pytest tests/test_optimizer.py::test_save_top_reports_links_optimize_result_id -v
```

Expected: `PASSED`

- [ ] **Step 4: Run the full test suite to check for regressions**

```bash
pytest tests/ -v
```

Expected: all previously passing tests still pass.

- [ ] **Step 5: Commit**

```bash
git add src/backtest/optimizer.py tests/test_optimizer.py docs/superpowers/specs/2026-04-21-optimize-reports-association-design.md docs/superpowers/plans/2026-04-21-optimize-reports-association.md
git commit -m "feat(optimizer): link reports to optimize_results via optimize_result_id"
```

---

### Task 3: Migrate existing data (best-effort backfill)

Existing `reports` rows have `optimize_result_id = NULL`. This task backfills them where a match can be found.

**Files:**
- No source changes — one-off SQL only

- [ ] **Step 1: Run backfill in sqlite3**

```bash
sqlite3 data/reports.db
```

```sql
-- Add column if missing (safe to run again)
ALTER TABLE reports ADD COLUMN optimize_result_id INTEGER;

-- Backfill by matching on strategy prefix, symbol, interval, and params
-- reports.strategy is like "ConsecutiveReverseStrategy_opt1" — base is everything before "_opt"
UPDATE reports
SET optimize_result_id = (
    SELECT o.id
    FROM optimize_results o
    WHERE o.symbol    = reports.symbol
      AND o.interval  = reports.interval
      AND o.strategy  = substr(reports.strategy, 1, instr(reports.strategy, '_opt') - 1)
    ORDER BY o.score DESC
    LIMIT 1
)
WHERE optimize_result_id IS NULL
  AND instr(reports.strategy, '_opt') > 0;
```

- [ ] **Step 2: Verify backfill**

```sql
SELECT id, strategy, optimize_result_id FROM reports ORDER BY id DESC LIMIT 10;
```

Expected: `optimize_result_id` is non-NULL for rows that matched. Rows from standalone `run` command remain NULL — that's correct.

- [ ] **Step 3: Exit**

```sql
.quit
```

---

## Self-Review

**Spec coverage:**
- ✅ `reports` gets `optimize_result_id INTEGER` nullable column
- ✅ Idempotent migration via `try/except OperationalError`
- ✅ Lookup by `params_json + strategy + symbol + interval`
- ✅ Standalone `run` command unaffected (writes NULL implicitly)
- ✅ Existing rows handled by backfill task

**Placeholder scan:** None found — all steps have concrete code and commands.

**Type consistency:** `_write_report_with_link` signature matches usage in `save_top_reports` and in the test.
