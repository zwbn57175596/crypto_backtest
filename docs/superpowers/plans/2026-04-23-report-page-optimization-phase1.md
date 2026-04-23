# Report & Optimize Page Optimization — Phase 1 (P0) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add batch_id support to optimize_results, batch selector UI on optimize page, enhanced optimize params card on report page with batch context, new risk metrics, Buy & Hold baseline, monthly heatmap, virtual scroll for trade table, adaptive downsampling, error handling, 3D scatter, parameter range sliders, and column customization.

**Architecture:** Backend changes center on adding `batch_id` column to `optimize_results` table and extending FastAPI routes. Frontend changes are in two vanilla JS+ECharts single-file pages (`index.html` and `optimize.html`). No frameworks — all changes use native DOM manipulation and ECharts APIs. TDD with `pytest` on backend; manual verification for frontend.

**Tech Stack:** Python 3.11+, FastAPI, SQLite3, vanilla JS, ECharts 5, ECharts GL (for 3D scatter)

**Spec:** `docs/prd/report-page-optimization.md` — this plan covers Phase 1 (P0) items only.

---

## File Structure

### Files to Modify

| File | Responsibility |
|------|---------------|
| `src/backtest/optimizer.py` | Add `batch_id` to schema, `save_results()` generates batch_id, migration for old data |
| `src/backtest/web/routes.py` | New API endpoints: batches list, batch_ids filter, report batch context, monthly, benchmark |
| `src/backtest/web/static/index.html` | Report page: new metrics, grouped cards, baseline chart, monthly heatmap, virtual scroll, error handling, adaptive downsample, enhanced opt-params card |
| `src/backtest/web/static/optimize.html` | Optimize page: batch selector, 3D scatter, range sliders, column customization |
| `tests/test_web.py` | Tests for new/modified API routes |
| `tests/test_optimizer.py` | Tests for batch_id in save_results() |

### No New Files

All changes go into existing files. The project uses single-file HTML pages — we keep that pattern.

---

## Task 1: Add batch_id Column to optimize_results

**Files:**
- Modify: `src/backtest/optimizer.py:580-630` (save_results function + CREATE TABLE)
- Test: `tests/test_optimizer.py`

This task adds the `batch_id` TEXT column to the `optimize_results` table schema, generates a batch_id in `save_results()`, and migrates old data.

- [ ] **Step 1: Write failing test for batch_id in save_results**

Add to `tests/test_optimizer.py`:

```python
class TestBatchId:
    def test_save_results_generates_batch_id(self, tmp_path):
        """save_results should auto-generate batch_id for all trials in one call."""
        db = str(tmp_path / "test.db")
        result = OptimizeResult(
            best_params={"x": 1},
            best_score=1.5,
            all_trials=[
                {"params": {"x": 1}, "score": 1.5, "report": {"net_return": 0.1}},
                {"params": {"x": 2}, "score": 1.0, "report": {"net_return": 0.05}},
            ],
            objective="sharpe_ratio",
            total_trials=2,
            elapsed_seconds=1.0,
        )
        save_results(db, "MaCross", "BTCUSDT", "1h", "2026-01-01", "2026-06-30", result)

        conn = sqlite3.connect(db)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT batch_id FROM optimize_results").fetchall()
        conn.close()

        assert len(rows) == 2
        # All rows share the same batch_id
        assert rows[0]["batch_id"] == rows[1]["batch_id"]
        # batch_id contains strategy and symbol
        batch_id = rows[0]["batch_id"]
        assert "MaCross" in batch_id
        assert "BTCUSDT" in batch_id

    def test_save_results_different_calls_different_batch_ids(self, tmp_path):
        """Two separate save_results calls should produce different batch_ids."""
        import time
        db = str(tmp_path / "test.db")
        result = OptimizeResult(
            best_params={"x": 1},
            best_score=1.5,
            all_trials=[{"params": {"x": 1}, "score": 1.5, "report": {"net_return": 0.1}}],
            objective="sharpe_ratio",
            total_trials=1,
            elapsed_seconds=1.0,
        )
        save_results(db, "MaCross", "BTCUSDT", "1h", "2026-01-01", "2026-06-30", result)
        time.sleep(0.01)  # ensure different timestamp
        save_results(db, "MaCross", "BTCUSDT", "1h", "2026-01-01", "2026-06-30", result)

        conn = sqlite3.connect(db)
        rows = conn.execute("SELECT DISTINCT batch_id FROM optimize_results").fetchall()
        conn.close()

        assert len(rows) == 2

    def test_batch_id_migration_for_old_data(self, tmp_path):
        """Old rows without batch_id should get backfilled."""
        db = str(tmp_path / "test.db")
        conn = sqlite3.connect(db)
        # Create old schema without batch_id
        conn.execute("""
            CREATE TABLE optimize_results (
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
            )
        """)
        # Insert old data
        conn.execute(
            "INSERT INTO optimize_results (strategy,symbol,interval,start_date,end_date,objective,score,params_json,report_json,created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            ("MaCross", "BTCUSDT", "1h", "2026-01-01", "2026-06-30", "sharpe_ratio", 1.5, '{"x":1}', '{"net_return":0.1}', "2026-04-20T09:30:00+00:00"),
        )
        conn.commit()
        conn.close()

        # Now call save_results which should trigger migration
        result = OptimizeResult(
            best_params={"x": 2},
            best_score=2.0,
            all_trials=[{"params": {"x": 2}, "score": 2.0, "report": {"net_return": 0.2}}],
            objective="sharpe_ratio",
            total_trials=1,
            elapsed_seconds=1.0,
        )
        save_results(db, "MaCross", "BTCUSDT", "1h", "2026-01-01", "2026-06-30", result)

        conn = sqlite3.connect(db)
        rows = conn.execute("SELECT batch_id FROM optimize_results WHERE id = 1").fetchall()
        conn.close()

        # Old row should have been backfilled
        assert rows[0][0] is not None
        assert "MaCross" in rows[0][0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_optimizer.py::TestBatchId -v`
Expected: FAIL — `batch_id` column doesn't exist

- [ ] **Step 3: Implement batch_id in optimizer.py**

In `src/backtest/optimizer.py`, modify the `save_results()` function:

1. Update CREATE TABLE to include `batch_id TEXT`:

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
        batch_id TEXT
    )
""")
```

2. Add idempotent migration + backfill after CREATE TABLE:

```python
# Idempotent migration — add batch_id column if not present
try:
    conn.execute("ALTER TABLE optimize_results ADD COLUMN batch_id TEXT")
    conn.commit()
except sqlite3.OperationalError:
    pass  # column already exists

# Backfill old rows without batch_id
conn.execute("""
    UPDATE optimize_results
    SET batch_id = strftime('%Y%m%dT%H%M%S', created_at) || '_' || strategy || '_' || symbol
    WHERE batch_id IS NULL
""")
conn.commit()
```

3. Generate batch_id before the INSERT:

```python
now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")
batch_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + strategy + "_" + symbol
```

4. Update the INSERT to include batch_id:

```python
conn.executemany(
    """INSERT INTO optimize_results
       (strategy, symbol, interval, start_date, end_date,
        objective, score, params_json, report_json, created_at, batch_id)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
    [
        (strategy, symbol, interval, start_date, end_date,
         result.objective, t["score"],
         json.dumps(t["params"]), json.dumps(t["report"]),
         now, batch_id)
        for t in trials
    ],
)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_optimizer.py::TestBatchId -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Run all existing optimizer tests to check no regression**

Run: `pytest tests/test_optimizer.py -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/backtest/optimizer.py tests/test_optimizer.py
git commit -m "feat: add batch_id column to optimize_results table

Generate batch_id in save_results() with format {timestamp}_{strategy}_{symbol}.
Idempotent migration backfills old rows using created_at + strategy + symbol."
```

---

## Task 2: Batch List API Endpoint

**Files:**
- Modify: `src/backtest/web/routes.py:54-88`
- Test: `tests/test_web.py`

New endpoint: `GET /api/optimize_results/batches?strategy=X&symbol=Y`

- [ ] **Step 1: Write failing test for batches endpoint**

Add to `tests/test_web.py`. First, create a fixture that includes batch_id data:

```python
@pytest.fixture
def db_with_batches(tmp_path):
    """Database with optimize_results that have batch_ids."""
    db = str(tmp_path / "reports.db")
    conn = sqlite3.connect(db)
    conn.execute("""
        CREATE TABLE optimize_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy TEXT NOT NULL, symbol TEXT NOT NULL, interval TEXT NOT NULL,
            start_date TEXT NOT NULL, end_date TEXT NOT NULL,
            objective TEXT NOT NULL, score REAL NOT NULL,
            params_json TEXT NOT NULL, report_json TEXT NOT NULL,
            created_at TEXT NOT NULL, batch_id TEXT
        )
    """)
    # Batch 1: 2 trials
    conn.execute(
        "INSERT INTO optimize_results (strategy,symbol,interval,start_date,end_date,objective,score,params_json,report_json,created_at,batch_id) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        ("MaCross","BTCUSDT","1h","2026-01-01","2026-03-31","sharpe_ratio",1.5,'{"x":1}','{"net_return":0.1}',
         "2026-04-15T14:00:00+00:00","20260415T140000_MaCross_BTCUSDT"),
    )
    conn.execute(
        "INSERT INTO optimize_results (strategy,symbol,interval,start_date,end_date,objective,score,params_json,report_json,created_at,batch_id) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        ("MaCross","BTCUSDT","1h","2026-01-01","2026-03-31","sharpe_ratio",1.2,'{"x":2}','{"net_return":0.05}',
         "2026-04-15T14:00:00+00:00","20260415T140000_MaCross_BTCUSDT"),
    )
    # Batch 2: 1 trial
    conn.execute(
        "INSERT INTO optimize_results (strategy,symbol,interval,start_date,end_date,objective,score,params_json,report_json,created_at,batch_id) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        ("MaCross","BTCUSDT","1h","2026-01-01","2026-06-30","sharpe_ratio",2.3,'{"x":3}','{"net_return":0.2}',
         "2026-04-23T12:00:00+00:00","20260423T120000_MaCross_BTCUSDT"),
    )
    conn.commit()
    conn.close()
    return db


def test_get_batches(db_with_batches):
    app = create_app(db_with_batches)
    client = TestClient(app)
    resp = client.get("/api/optimize_results/batches?strategy=MaCross&symbol=BTCUSDT")
    assert resp.status_code == 200
    batches = resp.json()
    assert len(batches) == 2
    # Ordered by created_at DESC — newest batch first
    assert batches[0]["batch_id"] == "20260423T120000_MaCross_BTCUSDT"
    assert batches[0]["count"] == 1
    assert batches[0]["best_score"] == 2.3
    assert batches[0]["start_date"] == "2026-01-01"
    assert batches[0]["end_date"] == "2026-06-30"
    assert batches[0]["objective"] == "sharpe_ratio"
    assert batches[0]["batch_number"] == 2
    # Older batch
    assert batches[1]["batch_id"] == "20260415T140000_MaCross_BTCUSDT"
    assert batches[1]["count"] == 2
    assert batches[1]["best_score"] == 1.5
    assert batches[1]["batch_number"] == 1


def test_get_batches_empty(db_with_batches):
    app = create_app(db_with_batches)
    client = TestClient(app)
    resp = client.get("/api/optimize_results/batches?strategy=NoExist&symbol=BTCUSDT")
    assert resp.status_code == 200
    assert resp.json() == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_web.py::test_get_batches -v`
Expected: FAIL — 404 (route doesn't exist)

- [ ] **Step 3: Implement batches endpoint in routes.py**

Add to `src/backtest/web/routes.py` — **IMPORTANT:** this route must be defined BEFORE the existing `/api/optimize_results` route to avoid path conflicts:

```python
@router.get("/api/optimize_results/batches")
def list_optimize_batches(
    request: Request,
    strategy: str = Query(...),
    symbol: str = Query(...),
):
    conn = sqlite3.connect(_get_db(request))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT batch_id,
               MIN(created_at) AS created_at,
               COUNT(*) AS count,
               MAX(score) AS best_score,
               objective,
               start_date,
               end_date
        FROM optimize_results
        WHERE strategy = ? AND symbol = ? AND batch_id IS NOT NULL
        GROUP BY batch_id
        ORDER BY created_at ASC
    """, (strategy, symbol)).fetchall()
    conn.close()

    batches = []
    for i, row in enumerate(rows):
        batches.append({
            "batch_id": row["batch_id"],
            "batch_number": i + 1,
            "created_at": row["created_at"],
            "count": row["count"],
            "best_score": row["best_score"],
            "objective": row["objective"],
            "start_date": row["start_date"],
            "end_date": row["end_date"],
        })
    # Return newest first for display
    batches.reverse()
    return batches
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_web.py::test_get_batches tests/test_web.py::test_get_batches_empty -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/backtest/web/routes.py tests/test_web.py
git commit -m "feat: add GET /api/optimize_results/batches endpoint

Returns batch list grouped by batch_id with count, best_score, date range,
and sequential batch_number for a given strategy+symbol."
```

---

## Task 3: Add batch_ids Filter to optimize_results API

**Files:**
- Modify: `src/backtest/web/routes.py` (the existing `list_optimize_results` function)
- Test: `tests/test_web.py`

- [ ] **Step 1: Write failing test for batch_ids filter**

Add to `tests/test_web.py`:

```python
def test_get_optimize_results_filtered_by_batch_ids(db_with_batches):
    app = create_app(db_with_batches)
    client = TestClient(app)
    resp = client.get(
        "/api/optimize_results?strategy=MaCross&symbol=BTCUSDT&batch_ids=20260423T120000_MaCross_BTCUSDT"
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["score"] == 2.3


def test_get_optimize_results_without_batch_ids_returns_all(db_with_batches):
    app = create_app(db_with_batches)
    client = TestClient(app)
    resp = client.get("/api/optimize_results?strategy=MaCross&symbol=BTCUSDT")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 3  # all rows from both batches
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_web.py::test_get_optimize_results_filtered_by_batch_ids -v`
Expected: FAIL — batch_ids param not handled, returns all 3 rows

- [ ] **Step 3: Implement batch_ids filter in list_optimize_results**

Modify `list_optimize_results` in `src/backtest/web/routes.py`:

```python
@router.get("/api/optimize_results")
def list_optimize_results(
    request: Request,
    strategy: str | None = Query(None),
    symbol: str | None = Query(None),
    batch_ids: str | None = Query(None),
):
    conn = sqlite3.connect(_get_db(request))
    conn.row_factory = sqlite3.Row
    query = "SELECT * FROM optimize_results WHERE 1=1"
    params = []
    if strategy:
        query += " AND strategy = ?"
        params.append(strategy)
    if symbol:
        query += " AND symbol = ?"
        params.append(symbol)
    if batch_ids:
        ids = [b.strip() for b in batch_ids.split(",") if b.strip()]
        placeholders = ",".join("?" * len(ids))
        query += f" AND batch_id IN ({placeholders})"
        params.extend(ids)
    query += " ORDER BY score DESC"
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_web.py::test_get_optimize_results_filtered_by_batch_ids tests/test_web.py::test_get_optimize_results_without_batch_ids_returns_all -v`
Expected: PASS

- [ ] **Step 5: Run all web tests to check no regression**

Run: `pytest tests/test_web.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add src/backtest/web/routes.py tests/test_web.py
git commit -m "feat: add batch_ids filter param to GET /api/optimize_results

Accepts comma-separated batch_ids to return only results from specified batches.
Omitting the param returns all results (backward compatible)."
```

---

## Task 4: Report API — Batch Context for Optimize Params

**Files:**
- Modify: `src/backtest/web/routes.py` (the existing `get_report` function)
- Test: `tests/test_web.py`

Extend `GET /api/reports/{id}` to return `optimize_batch_id`, `optimize_batch_created_at`, `optimize_start_date`, `optimize_end_date`, `optimize_rank`, `optimize_batch_total`.

- [ ] **Step 1: Write failing test for batch context in report**

Add to `tests/test_web.py`. Create a fixture with batch_id data and linked reports:

```python
@pytest.fixture
def db_with_batch_linked_report(tmp_path):
    """Database with reports linked to optimize_results that have batch_ids."""
    db = str(tmp_path / "reports.db")
    conn = sqlite3.connect(db)
    conn.execute("""
        CREATE TABLE optimize_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy TEXT NOT NULL, symbol TEXT NOT NULL, interval TEXT NOT NULL,
            start_date TEXT NOT NULL, end_date TEXT NOT NULL,
            objective TEXT NOT NULL, score REAL NOT NULL,
            params_json TEXT NOT NULL, report_json TEXT NOT NULL,
            created_at TEXT NOT NULL, batch_id TEXT
        )
    """)
    # 3 trials in same batch
    for i, (score, params) in enumerate([(2.3, '{"x":3}'), (1.8, '{"x":2}'), (1.5, '{"x":1}')]):
        conn.execute(
            """INSERT INTO optimize_results
               (strategy,symbol,interval,start_date,end_date,objective,score,params_json,report_json,created_at,batch_id)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            ("MaCross","BTCUSDT","1h","2026-01-01","2026-06-30","sharpe_ratio",score,params,
             '{"net_return":0.1}',
             "2026-04-23T12:00:00+00:00","20260423T120000_MaCross_BTCUSDT"),
        )
    # Report linked to the second-best trial (id=2, score=1.8)
    report_json = json.dumps({
        "net_return": 0.1, "annual_return": 0.2, "max_drawdown": 0.05,
        "max_dd_duration": 0, "sharpe_ratio": 1.8, "sortino_ratio": 2.0,
        "win_rate": 0.6, "profit_factor": 1.5, "total_trades": 10,
        "long_trades": 5, "short_trades": 5, "avg_hold_time": 3600000,
        "total_commission": 10.0, "total_funding": 0.0,
        "equity_curve": [[1000000, 10000]], "trades": [],
    })
    conn.execute("""
        CREATE TABLE reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy TEXT, symbol TEXT, interval TEXT,
            created_at TEXT, report_json TEXT,
            optimize_result_id INTEGER
        )
    """)
    conn.execute(
        "INSERT INTO reports (strategy,symbol,interval,created_at,report_json,optimize_result_id) VALUES (?,?,?,?,?,?)",
        ("MaCross","BTCUSDT","1h","2026-04-23T12:00:00+00:00", report_json, 2),
    )
    conn.commit()
    conn.close()
    return db


def test_get_report_includes_batch_context(db_with_batch_linked_report):
    app = create_app(db_with_batch_linked_report)
    client = TestClient(app)
    resp = client.get("/api/reports/1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["optimize_batch_id"] == "20260423T120000_MaCross_BTCUSDT"
    assert data["optimize_batch_created_at"] == "2026-04-23T12:00:00+00:00"
    assert data["optimize_start_date"] == "2026-01-01"
    assert data["optimize_end_date"] == "2026-06-30"
    assert data["optimize_rank"] == 2  # second best in batch
    assert data["optimize_batch_total"] == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_web.py::test_get_report_includes_batch_context -v`
Expected: FAIL — `optimize_batch_id` key missing

- [ ] **Step 3: Implement batch context in get_report**

Modify the `get_report` function in `src/backtest/web/routes.py`:

```python
@router.get("/api/reports/{report_id}")
def get_report(report_id: int, request: Request):
    conn = sqlite3.connect(_get_db(request))
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        """SELECT r.report_json, r.strategy, r.symbol, r.interval, r.created_at,
                  o.params_json AS optimize_params_json,
                  o.score       AS optimize_score,
                  o.objective   AS optimize_objective,
                  o.batch_id    AS optimize_batch_id,
                  o.created_at  AS optimize_batch_created_at,
                  o.start_date  AS optimize_start_date,
                  o.end_date    AS optimize_end_date,
                  (SELECT COUNT(*) + 1 FROM optimize_results o2
                   WHERE o2.batch_id = o.batch_id AND o2.score > o.score) AS optimize_rank,
                  (SELECT COUNT(*) FROM optimize_results o3
                   WHERE o3.batch_id = o.batch_id) AS optimize_batch_total
           FROM reports r
           LEFT JOIN optimize_results o ON r.optimize_result_id = o.id
           WHERE r.id = ?""",
        (report_id,),
    ).fetchone()
    conn.close()
    if row is None:
        raise HTTPException(status_code=404, detail="Report not found")
    report = json.loads(row["report_json"])
    report["id"] = report_id
    report["strategy"] = row["strategy"]
    report["symbol"] = row["symbol"]
    report["interval"] = row["interval"]
    report["created_at"] = row["created_at"]
    report["optimize_params"] = json.loads(row["optimize_params_json"]) if row["optimize_params_json"] else None
    report["optimize_score"] = row["optimize_score"]
    report["optimize_objective"] = row["optimize_objective"]
    report["optimize_batch_id"] = row["optimize_batch_id"]
    report["optimize_batch_created_at"] = row["optimize_batch_created_at"]
    report["optimize_start_date"] = row["optimize_start_date"]
    report["optimize_end_date"] = row["optimize_end_date"]
    report["optimize_rank"] = row["optimize_rank"]
    report["optimize_batch_total"] = row["optimize_batch_total"]
    return report
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_web.py::test_get_report_includes_batch_context -v`
Expected: PASS

- [ ] **Step 5: Run all web tests to check no regression**

Run: `pytest tests/test_web.py -v`
Expected: All PASS (existing tests for unlinked reports should still return None for new fields)

- [ ] **Step 6: Commit**

```bash
git add src/backtest/web/routes.py tests/test_web.py
git commit -m "feat: add batch context to GET /api/reports/{id} response

Includes optimize_batch_id, batch_created_at, start/end_date,
rank_in_batch, and batch_total via SQL subqueries."
```

---

## Task 5: Optimize Page — Batch Selector UI

**Files:**
- Modify: `src/backtest/web/static/optimize.html`

Add a batch multi-select checkbox list between the strategy/symbol filters and the charts. When batches are selected, pass `batch_ids` to the data API.

- [ ] **Step 1: Add batch selector HTML structure**

In `optimize.html`, add after the existing `filter-bar` div (after line 69):

```html
<div id="batch-panel" class="batch-panel" style="display:none;">
  <div class="batch-header">
    <span style="color:#aaa;font-size:13px;">优化批次</span>
    <label style="font-size:12px;color:#888;cursor:pointer;">
      <input type="checkbox" id="batch-select-all"> 全选
    </label>
  </div>
  <div id="batch-list" class="batch-list"></div>
</div>
```

- [ ] **Step 2: Add CSS for batch panel**

Add to the `<style>` block:

```css
.batch-panel { padding: 0 24px 12px; }
.batch-header { display: flex; justify-content: space-between; align-items: center;
                margin-bottom: 8px; }
.batch-list { display: flex; flex-direction: column; gap: 6px; }
.batch-item { display: flex; align-items: center; gap: 10px; background: #1e293b;
              border-radius: 6px; padding: 8px 12px; font-size: 12px; cursor: pointer; }
.batch-item:hover { background: #253347; }
.batch-item input { accent-color: #00d4aa; }
.batch-item .batch-info { flex: 1; display: flex; gap: 16px; align-items: center; }
.batch-item .batch-num { color: #00d4aa; font-weight: 600; min-width: 30px; }
.batch-item .batch-date { color: #888; }
.batch-item .batch-count { color: #888; }
.batch-item .batch-best { color: #22c55e; }
.batch-item .batch-range { color: #666; }
```

- [ ] **Step 3: Implement batch loading and selection JS logic**

Replace the existing `loadData()` function and add batch management functions. The full JS section in `optimize.html` should be updated:

```javascript
let allBatches = [];
let selectedBatchIds = [];

// After strategy/symbol change, load batches
document.getElementById('filter-strategy').onchange = onFilterChange;
document.getElementById('filter-symbol').onchange = onFilterChange;

function onFilterChange() {
  loadBatches();
}

function loadBatches() {
  const stratVal = document.getElementById('filter-strategy').value;
  if (!stratVal) {
    document.getElementById('batch-panel').style.display = 'none';
    selectedBatchIds = [];
    loadData();
    return;
  }
  const [strat, sym] = stratVal.split('|');
  fetch(`/api/optimize_results/batches?strategy=${encodeURIComponent(strat)}&symbol=${encodeURIComponent(sym)}`)
    .then(r => r.json())
    .then(batches => {
      allBatches = batches;
      if (batches.length === 0) {
        document.getElementById('batch-panel').style.display = 'none';
        selectedBatchIds = [];
        loadData();
        return;
      }
      renderBatchList(batches);
      document.getElementById('batch-panel').style.display = '';

      // Read batch selection from URL params
      const urlParams = new URLSearchParams(window.location.search);
      const urlBatches = urlParams.get('batches');
      if (urlBatches) {
        selectedBatchIds = urlBatches.split(',');
      } else {
        // Default: select newest batch only
        selectedBatchIds = [batches[0].batch_id];
      }
      updateBatchCheckboxes();
      loadData();
    });
}

function renderBatchList(batches) {
  const list = document.getElementById('batch-list');
  list.innerHTML = '';
  batches.forEach(b => {
    const div = document.createElement('div');
    div.className = 'batch-item';
    div.innerHTML = `
      <input type="checkbox" value="${b.batch_id}" onchange="onBatchToggle()">
      <div class="batch-info">
        <span class="batch-num">#${b.batch_number}</span>
        <span class="batch-date">${b.created_at.slice(0,16).replace('T',' ')}</span>
        <span class="batch-count">(${b.count}组)</span>
        <span class="batch-best">best=${b.best_score.toFixed(2)}</span>
        <span class="batch-range">${b.start_date.slice(5)} ~ ${b.end_date.slice(5)}</span>
      </div>`;
    list.appendChild(div);
  });

  document.getElementById('batch-select-all').onchange = function() {
    selectedBatchIds = this.checked ? allBatches.map(b => b.batch_id) : [];
    updateBatchCheckboxes();
    updateBatchUrl();
    loadData();
  };
}

function onBatchToggle() {
  const checkboxes = document.querySelectorAll('#batch-list input[type=checkbox]');
  selectedBatchIds = [];
  checkboxes.forEach(cb => { if (cb.checked) selectedBatchIds.push(cb.value); });
  document.getElementById('batch-select-all').checked = selectedBatchIds.length === allBatches.length;
  updateBatchUrl();
  loadData();
}

function updateBatchCheckboxes() {
  const checkboxes = document.querySelectorAll('#batch-list input[type=checkbox]');
  checkboxes.forEach(cb => { cb.checked = selectedBatchIds.includes(cb.value); });
  document.getElementById('batch-select-all').checked = selectedBatchIds.length === allBatches.length;
}

function updateBatchUrl() {
  const url = new URL(window.location);
  if (selectedBatchIds.length > 0 && selectedBatchIds.length < allBatches.length) {
    url.searchParams.set('batches', selectedBatchIds.join(','));
  } else {
    url.searchParams.delete('batches');
  }
  history.replaceState(null, '', url);
}
```

- [ ] **Step 4: Update loadData() to use batch_ids**

```javascript
function loadData() {
  const stratVal = document.getElementById('filter-strategy').value;
  const symVal = document.getElementById('filter-symbol').value;
  let url = '/api/optimize_results?';
  if (stratVal) {
    const [strat, sym] = stratVal.split('|');
    url += `strategy=${encodeURIComponent(strat)}&symbol=${encodeURIComponent(sym)}`;
  } else if (symVal) {
    url += `symbol=${encodeURIComponent(symVal)}`;
  }
  if (selectedBatchIds.length > 0) {
    url += `&batch_ids=${selectedBatchIds.map(encodeURIComponent).join(',')}`;
  }
  fetch(url).then(r => r.json()).then(data => {
    allData = data.map(d => ({
      ...d,
      params: JSON.parse(d.params_json),
      report: JSON.parse(d.report_json),
    }));
    if (allData.length === 0) {
      document.getElementById('no-data').style.display = '';
      document.querySelector('.charts-row').style.display = 'none';
      document.querySelector('.table-wrap').style.display = 'none';
      return;
    }
    document.getElementById('no-data').style.display = 'none';
    document.querySelector('.charts-row').style.display = '';
    document.querySelector('.table-wrap').style.display = '';

    document.getElementById('badge-objective').textContent = '目标: ' + allData[0].objective;
    document.getElementById('badge-count').textContent = allData.length + ' 组合';

    initHeatmapControls();
    renderScatter();
    renderHeatmap();
    renderTable();
  });
}
```

- [ ] **Step 5: Change initial load to call loadBatches instead of loadData**

In the strategies fetch callback, replace `loadData()` with `loadBatches()`:

```javascript
fetch('/api/optimize_results/strategies').then(r => r.json()).then(strategies => {
  // ... existing select population code ...
  loadBatches();  // was: loadData()
});
```

- [ ] **Step 6: Verify manually — start web server and test**

Run: `python -m backtest web --port 8000`
Navigate to `http://localhost:8000/optimize`
Expected: batch selector appears when a strategy+symbol is selected, data filters by selected batches.

- [ ] **Step 7: Commit**

```bash
git add src/backtest/web/static/optimize.html
git commit -m "feat: add batch selector UI to optimize page

Checkbox list for batch selection with auto-select newest batch.
Passes batch_ids filter to API. Persists selection in URL params."
```

---

## Task 6: Report Page — Enhanced Optimize Params Card

**Files:**
- Modify: `src/backtest/web/static/index.html`

Update the optimize params card to show batch context: batch number, run time, backtest date range, rank in batch, and a link to the optimize page for that batch.

- [ ] **Step 1: Add CSS for enhanced card layout**

Add to the `<style>` block in `index.html`:

```css
.opt-params-card .batch-context { font-size: 12px; color: #888; padding: 8px 0;
                                   border-bottom: 1px solid #2a3a4a; margin-bottom: 8px; }
.opt-params-card .batch-context div { padding: 2px 0; }
.opt-params-card .batch-context .highlight { color: #00d4aa; }
.opt-params-card .batch-link { display: block; text-align: center; padding: 8px;
                                margin-top: 8px; color: #00d4aa; text-decoration: none;
                                font-size: 12px; border: 1px solid #2a3a4a; border-radius: 6px; }
.opt-params-card .batch-link:hover { background: #00d4aa22; }
```

- [ ] **Step 2: Update the optimize params card rendering in JS**

In the `loadReport` function, replace the optimize params card rendering block (lines 211-226):

```javascript
// 优化参数卡片 (enhanced with batch context)
const optCard = document.getElementById('opt-params-card');
if (d.optimize_params) {
  const scoreLabel = document.getElementById('opt-score-label');
  scoreLabel.textContent = `score: ${d.optimize_score.toFixed(4)} | ${d.optimize_objective}`;
  const body = document.getElementById('opt-params-body');
  body.innerHTML = '';

  // Batch context section
  if (d.optimize_batch_id) {
    const ctx = document.createElement('div');
    ctx.className = 'batch-context';
    const batchTime = d.optimize_batch_created_at ? d.optimize_batch_created_at.slice(0,16).replace('T',' ') : '';
    ctx.innerHTML = `
      <div>批次: <span class="highlight">${d.optimize_batch_id.split('_')[0]}</span>  ${batchTime}</div>
      <div>回测区间: ${d.optimize_start_date || '?'} ~ ${d.optimize_end_date || '?'}</div>
      <div>批次排名: <span class="highlight">${d.optimize_rank}</span> / ${d.optimize_batch_total}</div>`;
    body.appendChild(ctx);
  }

  // Param key-value rows
  Object.entries(d.optimize_params).forEach(([k, v]) => {
    const row = document.createElement('div');
    row.className = 'param-row';
    row.innerHTML = `<span class="param-name">${k}</span><span class="param-value">${v}</span>`;
    body.appendChild(row);
  });

  // Link to optimize page with batch selected
  if (d.optimize_batch_id) {
    const link = document.createElement('a');
    link.className = 'batch-link';
    link.href = `/optimize?batches=${encodeURIComponent(d.optimize_batch_id)}`;
    link.textContent = '查看该批次全部结果 →';
    body.appendChild(link);
  }

  optCard.style.display = '';
} else {
  optCard.style.display = 'none';
}
```

- [ ] **Step 3: Verify manually — start web server and test**

Run: `python -m backtest web --port 8000`
Navigate to `http://localhost:8000/`
Select a report that has optimization params.
Expected: card shows batch context (batch time, date range, rank) and "查看该批次全部结果" link.

- [ ] **Step 4: Commit**

```bash
git add src/backtest/web/static/index.html
git commit -m "feat: enhance optimize params card with batch context

Shows batch timestamp, backtest date range, rank in batch,
and link to optimize page filtered by that batch."
```

---

## Task 7: Report Page — New Risk Metrics + Grouped Cards

**Files:**
- Modify: `src/backtest/web/static/index.html`

Add Calmar Ratio, max consecutive win/loss streak, daily avg return, return volatility. Group all metrics into 3 sections.

- [ ] **Step 1: Compute new metrics from existing report data in JS**

In `loadReport()`, after `const d = ...`, add helper computations:

```javascript
// Compute additional metrics client-side from equity_curve and trades
const closingTrades = (d.trades || []).filter(t => t.pnl !== 0);

// Max consecutive win/loss streaks
let maxWinStreak = 0, maxLossStreak = 0, curWin = 0, curLoss = 0;
closingTrades.forEach(t => {
  if (t.pnl > 0) { curWin++; curLoss = 0; maxWinStreak = Math.max(maxWinStreak, curWin); }
  else { curLoss++; curWin = 0; maxLossStreak = Math.max(maxLossStreak, curLoss); }
});

// Calmar ratio: annual_return / max_drawdown
const calmar = d.max_drawdown > 0 ? d.annual_return / d.max_drawdown : 0;

// Daily returns from equity curve
const ec = d.equity_curve || [];
const dailyReturns = [];
if (ec.length > 1) {
  // Group by day, use last equity per day
  const dayMap = new Map();
  ec.forEach(p => {
    const day = new Date(p[0]).toISOString().slice(0,10);
    dayMap.set(day, p[1]);
  });
  const days = [...dayMap.values()];
  for (let i = 1; i < days.length; i++) {
    dailyReturns.push((days[i] - days[i-1]) / days[i-1]);
  }
}
const avgDailyReturn = dailyReturns.length > 0 ? dailyReturns.reduce((a,b)=>a+b,0) / dailyReturns.length : 0;
const returnVol = dailyReturns.length > 1
  ? Math.sqrt(dailyReturns.reduce((s,r) => s + (r - avgDailyReturn)**2, 0) / (dailyReturns.length - 1)) * Math.sqrt(365)
  : 0;
```

- [ ] **Step 2: Replace the flat metrics grid with 3 grouped sections**

Replace the metrics-grid HTML and its rendering. First, update the HTML structure (replace the single `<div class="metrics-grid" id="metrics-grid">` with):

```html
<!-- 指标分组 -->
<div id="metrics-grouped" style="padding: 0 24px 16px;">
  <div style="margin-bottom: 8px;">
    <div style="font-size: 12px; color: #666; margin-bottom: 6px;">收益类</div>
    <div class="metrics-grid" id="metrics-return"></div>
  </div>
  <div style="margin-bottom: 8px;">
    <div style="font-size: 12px; color: #666; margin-bottom: 6px;">风险类</div>
    <div class="metrics-grid" id="metrics-risk"></div>
  </div>
  <div>
    <div style="font-size: 12px; color: #666; margin-bottom: 6px;">交易类</div>
    <div class="metrics-grid" id="metrics-trade"></div>
  </div>
</div>
```

Update the `.metrics-grid` CSS to handle variable column counts:

```css
.metrics-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 10px; }
```

- [ ] **Step 3: Render grouped metrics in JS**

Replace the old `metricsData` rendering with:

```javascript
// Grouped metrics
function renderMetricGroup(containerId, items) {
  const grid = document.getElementById(containerId);
  grid.innerHTML = '';
  items.forEach(m => {
    const div = document.createElement('div');
    div.className = 'metric';
    div.innerHTML = `<div class="label">${m.label}</div><div class="value" style="color:${m.color}">${m.value}</div>`;
    grid.appendChild(div);
  });
}

renderMetricGroup('metrics-return', [
  { label: '总收益率', value: (d.net_return * 100).toFixed(2) + '%', color: d.net_return >= 0 ? '#22c55e' : '#ef4444' },
  { label: '年化收益率', value: (d.annual_return * 100).toFixed(2) + '%', color: d.annual_return >= 0 ? '#22c55e' : '#ef4444' },
  { label: '日均收益率', value: (avgDailyReturn * 100).toFixed(4) + '%', color: avgDailyReturn >= 0 ? '#22c55e' : '#ef4444' },
]);

renderMetricGroup('metrics-risk', [
  { label: '最大回撤', value: (d.max_drawdown * 100).toFixed(2) + '%', color: '#ef4444' },
  { label: '最大回撤持续', value: formatDuration(d.max_dd_duration || 0), color: '#ef4444' },
  { label: '收益波动率', value: (returnVol * 100).toFixed(2) + '%', color: '#f59e0b' },
  { label: '夏普比率', value: d.sharpe_ratio.toFixed(2), color: d.sharpe_ratio >= 1 ? '#22c55e' : d.sharpe_ratio >= 0 ? '#00d4aa' : '#ef4444' },
  { label: '索提诺比率', value: d.sortino_ratio === Infinity ? 'Inf' : (d.sortino_ratio || 0).toFixed(2), color: '#00d4aa' },
  { label: 'Calmar', value: calmar.toFixed(2), color: calmar >= 1 ? '#22c55e' : '#f59e0b' },
]);

renderMetricGroup('metrics-trade', [
  { label: '胜率', value: (d.win_rate * 100).toFixed(1) + '%', color: d.win_rate >= 0.5 ? '#22c55e' : '#ef4444' },
  { label: '盈亏比', value: d.profit_factor === Infinity ? 'Inf' : (d.profit_factor || 0).toFixed(2), color: d.profit_factor >= 1 ? '#22c55e' : '#ef4444' },
  { label: '总交易次数', value: d.total_trades, color: '#e0e0e0' },
  { label: '多头交易', value: d.long_trades, color: '#22c55e' },
  { label: '空头交易', value: d.short_trades, color: '#ef4444' },
  { label: '平均持仓时间', value: formatDuration(d.avg_hold_time || 0), color: '#e0e0e0' },
  { label: '连续盈利', value: maxWinStreak + ' 次', color: '#22c55e' },
  { label: '连续亏损', value: maxLossStreak + ' 次', color: '#ef4444' },
  { label: '总手续费', value: (d.total_commission || 0).toFixed(2) + ' USDT', color: '#f59e0b' },
  { label: '总资金费用', value: (d.total_funding || 0).toFixed(2) + ' USDT', color: '#f59e0b' },
]);
```

- [ ] **Step 4: Remove the old `<div class="metrics-grid" id="metrics-grid"></div>`** and the old `metricsData` rendering code. Also remove the duplicate KPI row since total return and sharpe are now in groups. Keep the 4 KPI cards at top for at-a-glance view.

- [ ] **Step 5: Verify manually**

Run: `python -m backtest web --port 8000`
Expected: 3 grouped metric sections with new Calmar, streak, daily return, volatility metrics.

- [ ] **Step 6: Commit**

```bash
git add src/backtest/web/static/index.html
git commit -m "feat: add risk metrics and grouped metric cards to report page

New metrics: Calmar ratio, max consecutive win/loss streak,
daily avg return, annualized return volatility.
Metrics organized into Return/Risk/Trade groups."
```

---

## Task 8: Report Page — Buy & Hold Baseline on Equity Chart

**Files:**
- Modify: `src/backtest/web/static/index.html`

Overlay a Buy & Hold baseline curve on the equity chart. Compute it client-side from the first and last equity_curve data points, scaling linearly by the first bar's close price vs each bar's close price.

- [ ] **Step 1: Compute Buy & Hold baseline from equity curve**

The equity_curve already has `[timestamp, equity]` pairs. We don't have raw bar close prices in the report JSON. A simpler approach: use the equity curve's initial value and scale by the ratio of each timestamp's equity to a hypothetical buy-and-hold. Since we don't have close prices, we'll compute a pseudo-baseline: start with initial equity, and derive the buy-and-hold as `initial_equity * (close_price / first_close_price)`.

Actually, the report JSON doesn't include close prices. The simplest approach is to compute it from the trades data — but that's complex. Let's add a backend endpoint that computes it.

Add to `routes.py` a new endpoint:

```python
@router.get("/api/reports/{report_id}/benchmark")
def get_benchmark(report_id: int, request: Request):
    """Return Buy & Hold equity curve for the same period."""
    conn = sqlite3.connect(_get_db(request))
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT report_json, symbol, interval FROM reports WHERE id = ?",
        (report_id,),
    ).fetchone()
    if row is None:
        conn.close()
        raise HTTPException(status_code=404, detail="Report not found")

    report = json.loads(row["report_json"])
    equity_curve = report.get("equity_curve", [])
    if len(equity_curve) < 2:
        conn.close()
        return {"benchmark": []}

    # Get initial balance from first equity point
    initial = equity_curve[0][1]
    first_ts = equity_curve[0][0]
    last_ts = equity_curve[-1][0]

    # Look up kline close prices for the period
    db_path = _get_db(request)
    # The klines DB might be separate — check if klines table exists
    klines_db = str(Path(db_path).parent / "klines.db")
    try:
        kconn = sqlite3.connect(klines_db)
        kconn.row_factory = sqlite3.Row
        klines = kconn.execute(
            """SELECT timestamp, close FROM klines
               WHERE symbol = ? AND interval = ? AND timestamp >= ? AND timestamp <= ?
               ORDER BY timestamp""",
            (row["symbol"], row["interval"], first_ts, last_ts),
        ).fetchall()
        kconn.close()
    except Exception:
        conn.close()
        return {"benchmark": []}
    conn.close()

    if not klines:
        return {"benchmark": []}

    first_close = klines[0]["close"]
    benchmark = [[k["timestamp"], initial * k["close"] / first_close] for k in klines]
    return {"benchmark": benchmark}
```

- [ ] **Step 2: Write test for benchmark endpoint**

Add to `tests/test_web.py`:

```python
@pytest.fixture
def db_with_klines(tmp_path):
    """Database with reports and a separate klines.db."""
    db = str(tmp_path / "reports.db")
    conn = sqlite3.connect(db)
    report_json = json.dumps({
        "net_return": 0.1, "annual_return": 0.2, "max_drawdown": 0.05,
        "max_dd_duration": 0, "sharpe_ratio": 1.8, "sortino_ratio": 2.0,
        "win_rate": 0.6, "profit_factor": 1.5, "total_trades": 10,
        "long_trades": 5, "short_trades": 5, "avg_hold_time": 3600000,
        "total_commission": 10.0, "total_funding": 0.0,
        "equity_curve": [[1000, 10000], [2000, 10500], [3000, 10200]],
        "trades": [],
    })
    conn.execute("""
        CREATE TABLE reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy TEXT, symbol TEXT, interval TEXT,
            created_at TEXT, report_json TEXT, optimize_result_id INTEGER
        )
    """)
    conn.execute(
        "INSERT INTO reports (strategy,symbol,interval,created_at,report_json) VALUES (?,?,?,?,?)",
        ("MaCross", "BTCUSDT", "1h", "2026-04-23", report_json),
    )
    conn.commit()
    conn.close()

    # Create klines.db
    klines_db = str(tmp_path / "klines.db")
    kconn = sqlite3.connect(klines_db)
    kconn.execute("""
        CREATE TABLE klines (
            exchange TEXT, symbol TEXT, interval TEXT, timestamp INTEGER,
            open REAL, high REAL, low REAL, close REAL, volume REAL,
            PRIMARY KEY (exchange, symbol, interval, timestamp)
        )
    """)
    kconn.execute(
        "INSERT INTO klines VALUES (?,?,?,?,?,?,?,?,?)",
        ("binance", "BTCUSDT", "1h", 1000, 100, 105, 95, 102, 1000),
    )
    kconn.execute(
        "INSERT INTO klines VALUES (?,?,?,?,?,?,?,?,?)",
        ("binance", "BTCUSDT", "1h", 2000, 102, 110, 100, 108, 1000),
    )
    kconn.execute(
        "INSERT INTO klines VALUES (?,?,?,?,?,?,?,?,?)",
        ("binance", "BTCUSDT", "1h", 3000, 108, 112, 105, 99, 1000),
    )
    kconn.commit()
    kconn.close()

    return db


def test_get_benchmark(db_with_klines):
    app = create_app(db_with_klines)
    client = TestClient(app)
    resp = client.get("/api/reports/1/benchmark")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["benchmark"]) == 3
    # First point: initial * close/first_close = 10000 * 102/102 = 10000
    assert data["benchmark"][0][1] == 10000.0
    # Second point: 10000 * 108/102 ≈ 10588.24
    assert abs(data["benchmark"][1][1] - 10588.24) < 1
```

- [ ] **Step 3: Run test to verify it fails, then passes after adding the route**

Run: `pytest tests/test_web.py::test_get_benchmark -v`

- [ ] **Step 4: Add benchmark overlay to equity chart in index.html**

In `loadReport()`, after the equity chart `setOption` call, add:

```javascript
// Fetch and overlay Buy & Hold baseline
fetch(`/api/reports/${id}/benchmark`).then(r => r.json()).then(bm => {
  if (!bm.benchmark || bm.benchmark.length === 0) return;
  const bmDates = bm.benchmark.map(p => {
    const dt = new Date(p[0]);
    return dt.getFullYear() + '-' + String(dt.getMonth()+1).padStart(2,'0') + '-' + String(dt.getDate()).padStart(2,'0');
  });
  const bmVals = bm.benchmark.map(p => p[1]);
  const bmSampled = downsample(bmDates, bmVals, 500);

  const currentOption = eqChart.getOption();
  eqChart.setOption({
    legend: { data: ['策略权益', 'Buy & Hold'], textStyle: { color: '#888' }, top: 0, right: 20 },
    series: [
      { name: '策略权益', data: currentOption.series[0].data, type: 'line', smooth: true, symbol: 'none',
        lineStyle: { color: '#22c55e', width: 1.5 },
        areaStyle: { color: new echarts.graphic.LinearGradient(0,0,0,1,
          [{offset:0,color:'rgba(34,197,94,0.3)'},{offset:1,color:'rgba(34,197,94,0.02)'}]) }},
      { name: 'Buy & Hold', data: bmSampled.vals, type: 'line', smooth: true, symbol: 'none',
        lineStyle: { color: '#f59e0b', width: 1.5, type: 'dashed' } },
    ],
  });
});
```

- [ ] **Step 5: Verify manually**

Run: `python -m backtest web --port 8000`
Expected: equity chart shows a dashed orange Buy & Hold line alongside the green strategy equity.

- [ ] **Step 6: Commit**

```bash
git add src/backtest/web/routes.py src/backtest/web/static/index.html tests/test_web.py
git commit -m "feat: add Buy & Hold baseline to equity chart

New endpoint GET /api/reports/{id}/benchmark computes buy-and-hold
equity from klines close prices. Frontend overlays as dashed line."
```

---

## Task 9: Report Page — Monthly Returns Heatmap

**Files:**
- Modify: `src/backtest/web/static/index.html`

Add a calendar-style monthly returns heatmap computed client-side from equity_curve data.

- [ ] **Step 1: Add heatmap chart container**

In `index.html`, add after the equity chart-box (after line 95):

```html
<div class="chart-box"><h3>月度收益</h3><div id="chart-monthly" style="height:200px;"></div></div>
```

- [ ] **Step 2: Initialize the chart**

Add `monthlyChart` to the chart variables and `initCharts()`:

```javascript
let eqChart, ddChart, pnlChart, monthlyChart;
function initCharts() {
  if (eqChart) return;
  eqChart = echarts.init(document.getElementById('chart-equity'));
  ddChart = echarts.init(document.getElementById('chart-dd'));
  pnlChart = echarts.init(document.getElementById('chart-pnl'));
  monthlyChart = echarts.init(document.getElementById('chart-monthly'));
}
window.addEventListener('resize', () => {
  if (eqChart) { eqChart.resize(); ddChart.resize(); pnlChart.resize(); monthlyChart.resize(); }
});
```

- [ ] **Step 3: Compute monthly returns and render heatmap**

In `loadReport()`, add after the equity chart code:

```javascript
// Monthly returns heatmap
const monthMap = new Map();  // "YYYY-MM" -> {first: equity, last: equity}
ec.forEach(p => {
  const dt = new Date(p[0]);
  const key = dt.getFullYear() + '-' + String(dt.getMonth()+1).padStart(2,'0');
  if (!monthMap.has(key)) monthMap.set(key, { first: p[1], last: p[1] });
  else monthMap.get(key).last = p[1];
});

const monthlyData = [];
const months = [...monthMap.keys()].sort();
let prevLast = null;
months.forEach(m => {
  const entry = monthMap.get(m);
  const base = prevLast !== null ? prevLast : entry.first;
  const ret = base > 0 ? (entry.last - base) / base : 0;
  monthlyData.push({ month: m, return: ret });
  prevLast = entry.last;
});

if (monthlyData.length > 0) {
  // Use ECharts heatmap with x=month, y=year
  const years = [...new Set(monthlyData.map(d => d.month.slice(0,4)))].sort();
  const monthNames = ['01','02','03','04','05','06','07','08','09','10','11','12'];
  const heatData = [];
  let minVal = 0, maxVal = 0;
  monthlyData.forEach(d => {
    const yi = years.indexOf(d.month.slice(0,4));
    const mi = monthNames.indexOf(d.month.slice(5));
    if (mi >= 0) {
      heatData.push([mi, yi, +(d.return * 100).toFixed(2)]);
      minVal = Math.min(minVal, d.return * 100);
      maxVal = Math.max(maxVal, d.return * 100);
    }
  });

  monthlyChart.setOption({
    tooltip: { formatter: p => `${years[p.data[1]]}-${monthNames[p.data[0]]}<br/>收益: ${p.data[2].toFixed(2)}%` },
    xAxis: { type: 'category', data: monthNames, axisLabel: { color: '#888' } },
    yAxis: { type: 'category', data: years, axisLabel: { color: '#888' } },
    visualMap: { min: Math.min(minVal, -1), max: Math.max(maxVal, 1), calculable: true,
                 orient: 'horizontal', left: 'center', bottom: 0, textStyle: { color: '#888' },
                 inRange: { color: ['#ef4444', '#1e293b', '#22c55e'] } },
    series: [{ type: 'heatmap', data: heatData,
               label: { show: true, color: '#e0e0e0', fontSize: 11, formatter: p => p.data[2].toFixed(1) + '%' },
               emphasis: { itemStyle: { shadowBlur: 10, shadowColor: 'rgba(0,0,0,0.5)' } } }],
    grid: { left: 50, right: 20, top: 10, bottom: 50 },
  }, true);
}
```

- [ ] **Step 4: Verify manually**

Run: `python -m backtest web --port 8000`
Expected: monthly heatmap shows green/red cells for positive/negative months.

- [ ] **Step 5: Commit**

```bash
git add src/backtest/web/static/index.html
git commit -m "feat: add monthly returns heatmap to report page

Calendar-style heatmap showing monthly returns by year.
Computed client-side from equity_curve data."
```

---

## Task 10: Report Page — Virtual Scroll for Trade Table

**Files:**
- Modify: `src/backtest/web/static/index.html`

Replace the current full-DOM trade table rendering with a virtual scroll implementation that only renders visible rows.

- [ ] **Step 1: Replace trade table rendering with virtual scroll**

Update the trade-wrap container to use a fixed-height scrollable area with a tall inner div for scroll height, and a visible-rows container that repositions via `transform: translateY`:

```html
<div class="chart-box">
  <h3>交易明细 <span id="trade-count" style="color:#888;font-size:12px;"></span></h3>
  <div class="trade-wrap" id="trade-scroll" style="position:relative;">
    <table class="trade-table">
      <thead><tr><th>#</th><th>时间</th><th>方向</th><th>价格</th><th>数量</th><th>盈亏</th><th>手续费</th></tr></thead>
    </table>
    <div id="trade-virtual" style="overflow-y:auto;max-height:360px;position:relative;">
      <div id="trade-spacer" style="width:100%;"></div>
      <table class="trade-table" style="position:absolute;top:0;left:0;width:100%;">
        <tbody id="trade-body"></tbody>
      </table>
    </div>
  </div>
</div>
```

- [ ] **Step 2: Implement virtual scroll JS**

Replace the trade table rendering code in `loadReport()`:

```javascript
// Virtual scroll for trade table
const allTrades = d.trades || [];
document.getElementById('trade-count').textContent = `(${allTrades.length} 笔)`;
const ROW_HEIGHT = 38;
const VISIBLE_COUNT = 10;
const BUFFER = 5;
const totalHeight = allTrades.length * ROW_HEIGHT;
const spacer = document.getElementById('trade-spacer');
spacer.style.height = totalHeight + 'px';
const tbody = document.getElementById('trade-body');
const virtualDiv = document.getElementById('trade-virtual');

function renderVisibleTrades() {
  const scrollTop = virtualDiv.scrollTop;
  const startIdx = Math.max(0, Math.floor(scrollTop / ROW_HEIGHT) - BUFFER);
  const endIdx = Math.min(allTrades.length, startIdx + VISIBLE_COUNT + 2 * BUFFER);
  tbody.innerHTML = '';
  const table = tbody.parentElement;
  table.style.transform = `translateY(${startIdx * ROW_HEIGHT}px)`;
  for (let i = startIdx; i < endIdx; i++) {
    const t = allTrades[i];
    const tr = document.createElement('tr');
    const sideClass = t.side === 'buy' ? 'long' : 'short';
    const sideText = t.side === 'buy' ? '买入' : '卖出';
    const pnlClass = t.pnl > 0 ? 'long' : t.pnl < 0 ? 'short' : '';
    tr.innerHTML = `<td>${i+1}</td>
      <td>${new Date(t.timestamp).toLocaleString('zh-CN')}</td>
      <td class="${sideClass}">${sideText}</td>
      <td>${t.price.toFixed(2)}</td><td>${t.quantity.toFixed(2)}</td>
      <td class="${pnlClass}">${t.pnl.toFixed(4)}</td><td>${t.commission.toFixed(2)}</td>`;
    tbody.appendChild(tr);
  }
}

virtualDiv.onscroll = renderVisibleTrades;
renderVisibleTrades();
```

- [ ] **Step 3: Verify manually with a report that has many trades**

Run: `python -m backtest web --port 8000`
Expected: trade table scrolls smoothly, only ~20 DOM rows exist at any time.

- [ ] **Step 4: Commit**

```bash
git add src/backtest/web/static/index.html
git commit -m "feat: implement virtual scroll for trade table

Only renders visible rows plus buffer. Handles 10k+ trades
without DOM performance issues."
```

---

## Task 11: Report Page — Adaptive Downsampling + Error Handling

**Files:**
- Modify: `src/backtest/web/static/index.html`

- [ ] **Step 1: Replace hardcoded 500 with screen-width adaptive value**

Update the downsample call:

```javascript
// Adaptive downsampling based on chart container width
const chartWidth = document.getElementById('chart-equity').clientWidth || 800;
const maxPoints = Math.max(200, Math.min(2000, chartWidth));
const eq = downsample(allDates, allVals, maxPoints);
```

- [ ] **Step 2: Add error handling wrapper for API calls**

Add a utility function and wrap all fetch calls:

```javascript
function apiFetch(url) {
  return fetch(url).then(r => {
    if (!r.ok) throw new Error(`API error: ${r.status}`);
    return r.json();
  }).catch(err => {
    console.error('API fetch failed:', url, err);
    const loading = document.getElementById('loading');
    loading.style.display = '';
    loading.innerHTML = `<div style="color:#ef4444;">加载失败: ${err.message}</div>
      <button onclick="location.reload()" style="margin-top:12px;padding:8px 16px;background:#1e293b;color:#e0e0e0;border:1px solid #333;border-radius:6px;cursor:pointer;">重试</button>`;
    throw err;
  });
}
```

Replace `fetch(...)` calls with `apiFetch(...)` for the reports list and report detail calls.

- [ ] **Step 3: Add loading skeleton state**

Add a CSS class for skeleton loading:

```css
.skeleton { background: linear-gradient(90deg, #1e293b 25%, #253347 50%, #1e293b 75%);
            background-size: 200% 100%; animation: shimmer 1.5s infinite;
            border-radius: 8px; height: 80px; }
@keyframes shimmer { 0% { background-position: 200% 0; } 100% { background-position: -200% 0; } }
```

- [ ] **Step 4: Verify manually**

Run: `python -m backtest web --port 8000`
Expected: charts sample adaptively, error states show retry button.

- [ ] **Step 5: Commit**

```bash
git add src/backtest/web/static/index.html
git commit -m "feat: adaptive downsampling and error handling for report page

Downsample points based on chart container width.
API errors show user-friendly message with retry button."
```

---

## Task 12: Optimize Page — 3D Scatter Plot

**Files:**
- Modify: `src/backtest/web/static/optimize.html`

Add ECharts GL for 3D scatter when >= 3 params are present.

- [ ] **Step 1: Add ECharts GL script tag**

In `optimize.html` `<head>`, add after the ECharts script:

```html
<script src="https://cdn.jsdelivr.net/npm/echarts-gl@2/dist/echarts-gl.min.js"></script>
```

- [ ] **Step 2: Add 3D chart container and controls**

Add after the existing `charts-row` div:

```html
<div id="chart-3d-wrap" class="charts-row" style="display:none;">
  <div class="chart-box" style="grid-column: span 2;">
    <h3>3D 参数空间</h3>
    <div class="chart-controls">
      <select id="scatter3d-x"></select>
      <select id="scatter3d-y"></select>
      <select id="scatter3d-z"></select>
    </div>
    <div id="chart-scatter3d" style="height:400px;"></div>
  </div>
</div>
```

- [ ] **Step 3: Implement 3D scatter rendering**

Add to the JS section:

```javascript
let scatter3dChart;

function init3dControls() {
  const names = getParamNames();
  if (names.length < 3) {
    document.getElementById('chart-3d-wrap').style.display = 'none';
    return;
  }
  document.getElementById('chart-3d-wrap').style.display = '';
  ['scatter3d-x','scatter3d-y','scatter3d-z'].forEach((id, i) => {
    const sel = document.getElementById(id);
    sel.innerHTML = '';
    names.forEach((n, j) => {
      sel.innerHTML += `<option value="${n}" ${j===i?'selected':''}>${n}</option>`;
    });
    sel.onchange = render3dScatter;
  });
  render3dScatter();
}

function render3dScatter() {
  if (!scatter3dChart) scatter3dChart = echarts.init(document.getElementById('chart-scatter3d'));
  const xP = document.getElementById('scatter3d-x').value;
  const yP = document.getElementById('scatter3d-y').value;
  const zP = document.getElementById('scatter3d-z').value;
  if (!xP || !yP || !zP) return;

  const scores = allData.map(d => d.score);
  const minScore = Math.min(...scores);
  const maxScore = Math.max(...scores);

  const data = allData.map(d => [d.params[xP], d.params[yP], d.params[zP], d.score]);

  scatter3dChart.setOption({
    tooltip: {},
    xAxis3D: { name: xP, type: 'value' },
    yAxis3D: { name: yP, type: 'value' },
    zAxis3D: { name: zP, type: 'value' },
    visualMap: { min: minScore, max: maxScore, dimension: 3, inRange: { color: ['#ef4444','#f59e0b','#22c55e'] },
                 textStyle: { color: '#888' } },
    grid3D: { viewControl: { autoRotate: false }, light: { main: { intensity: 1.2 } },
              environment: '#0f1117' },
    series: [{ type: 'scatter3D', data: data, symbolSize: 6,
               itemStyle: { opacity: 0.8 } }],
  }, true);
}
```

- [ ] **Step 4: Call `init3dControls()` from `loadData()` after `renderTable()`**

```javascript
// In loadData() success handler, add after renderTable():
init3dControls();
```

- [ ] **Step 5: Handle resize for 3D chart**

```javascript
window.addEventListener('resize', () => {
  if (scatterChart) scatterChart.resize();
  if (heatmapChart) heatmapChart.resize();
  if (scatter3dChart) scatter3dChart.resize();
});
```

- [ ] **Step 6: Verify manually**

Run: `python -m backtest web --port 8000`
Navigate to optimize page, select a strategy with >= 3 params.
Expected: 3D scatter appears with rotatable view, color-mapped by score.

- [ ] **Step 7: Commit**

```bash
git add src/backtest/web/static/optimize.html
git commit -m "feat: add 3D parameter space scatter plot

Uses ECharts GL for 3D scatter when >=3 params exist.
X/Y/Z axes selectable, color maps to score."
```

---

## Task 13: Optimize Page — Parameter Range Sliders

**Files:**
- Modify: `src/backtest/web/static/optimize.html`

Add range slider filters for each optimization parameter.

- [ ] **Step 1: Add slider container HTML**

Add after the batch-panel div:

```html
<div id="param-sliders" class="param-sliders" style="display:none;padding:0 24px 12px;">
  <div style="font-size:13px;color:#aaa;margin-bottom:8px;">参数范围筛选</div>
  <div id="slider-list" style="display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:8px;"></div>
</div>
```

- [ ] **Step 2: Add CSS for sliders**

```css
.slider-item { background: #1e293b; border-radius: 6px; padding: 8px 12px; }
.slider-item label { font-size: 11px; color: #888; display: block; margin-bottom: 4px; }
.slider-item .range-values { display: flex; justify-content: space-between; font-size: 11px; color: #00d4aa; }
.slider-item input[type=range] { width: 100%; accent-color: #00d4aa; }
.slider-row { display: flex; gap: 8px; align-items: center; }
```

- [ ] **Step 3: Implement slider rendering and filtering**

```javascript
let paramRanges = {};  // { paramName: { min, max, curMin, curMax } }
let unfilteredData = [];  // allData before param range filtering

function initParamSliders() {
  const names = getParamNames();
  if (names.length === 0) {
    document.getElementById('param-sliders').style.display = 'none';
    return;
  }
  document.getElementById('param-sliders').style.display = '';
  const list = document.getElementById('slider-list');
  list.innerHTML = '';
  paramRanges = {};

  names.forEach(name => {
    const values = unfilteredData.map(d => d.params[name]).filter(v => typeof v === 'number');
    if (values.length === 0) return;
    const min = Math.min(...values);
    const max = Math.max(...values);
    if (min === max) return;  // single value, no slider needed
    paramRanges[name] = { min, max, curMin: min, curMax: max };

    const step = (max - min) / 100;
    const div = document.createElement('div');
    div.className = 'slider-item';
    div.innerHTML = `
      <label>${name}</label>
      <div class="slider-row">
        <input type="range" min="${min}" max="${max}" step="${step}" value="${min}"
               id="slider-min-${name}" oninput="onSliderChange('${name}')">
        <input type="range" min="${min}" max="${max}" step="${step}" value="${max}"
               id="slider-max-${name}" oninput="onSliderChange('${name}')">
      </div>
      <div class="range-values">
        <span id="slider-val-min-${name}">${min}</span>
        <span id="slider-val-max-${name}">${max}</span>
      </div>`;
    list.appendChild(div);
  });
}

function onSliderChange(name) {
  const minVal = parseFloat(document.getElementById(`slider-min-${name}`).value);
  const maxVal = parseFloat(document.getElementById(`slider-max-${name}`).value);
  paramRanges[name].curMin = Math.min(minVal, maxVal);
  paramRanges[name].curMax = Math.max(minVal, maxVal);
  document.getElementById(`slider-val-min-${name}`).textContent = paramRanges[name].curMin.toFixed(2);
  document.getElementById(`slider-val-max-${name}`).textContent = paramRanges[name].curMax.toFixed(2);
  applyParamFilters();
}

function applyParamFilters() {
  allData = unfilteredData.filter(d => {
    for (const [name, range] of Object.entries(paramRanges)) {
      const v = d.params[name];
      if (typeof v === 'number' && (v < range.curMin || v > range.curMax)) return false;
    }
    return true;
  });
  document.getElementById('badge-count').textContent = allData.length + ' 组合';
  renderScatter();
  renderHeatmap();
  renderTable();
  if (scatter3dChart) render3dScatter();
}
```

- [ ] **Step 4: Integrate into loadData()**

In `loadData()`, after parsing the API response data:

```javascript
// Store unfiltered data and init sliders
unfilteredData = allData;
initParamSliders();
```

- [ ] **Step 5: Verify manually**

Run: `python -m backtest web --port 8000`
Expected: range sliders appear for each parameter, moving them filters charts + table in real time.

- [ ] **Step 6: Commit**

```bash
git add src/backtest/web/static/optimize.html
git commit -m "feat: add parameter range slider filters to optimize page

Dual range sliders per parameter for real-time filtering.
Updates scatter, heatmap, 3D scatter, and table simultaneously."
```

---

## Task 14: Optimize Page — Column Customization

**Files:**
- Modify: `src/backtest/web/static/optimize.html`

Allow users to show/hide table columns, persisted in localStorage.

- [ ] **Step 1: Add column toggle UI**

Add a button + dropdown above the table:

```html
<!-- Add inside table-wrap div, before table-container -->
<div style="display:flex;justify-content:flex-end;margin-bottom:8px;">
  <div style="position:relative;">
    <button id="col-toggle-btn" onclick="toggleColMenu()"
            style="background:#1e293b;color:#aaa;border:1px solid #333;padding:6px 12px;border-radius:6px;font-size:12px;cursor:pointer;">
      列设置 ▾
    </button>
    <div id="col-menu" style="display:none;position:absolute;right:0;top:100%;background:#1e293b;border:1px solid #333;border-radius:6px;padding:8px;z-index:10;min-width:180px;">
      <div id="col-menu-items"></div>
    </div>
  </div>
</div>
```

- [ ] **Step 2: Implement column toggle logic**

```javascript
const DEFAULT_VISIBLE_COLS = ['rank','score','net_return','max_drawdown','sharpe_ratio','win_rate'];
let visibleCols = JSON.parse(localStorage.getItem('opt_visible_cols') || 'null') || null;

function getVisibleCols() {
  if (visibleCols) return visibleCols;
  const paramNames = getParamNames();
  return [...DEFAULT_VISIBLE_COLS, ...paramNames.map(p => 'param_' + p)];
}

function toggleColMenu() {
  const menu = document.getElementById('col-menu');
  menu.style.display = menu.style.display === 'none' ? '' : 'none';
  if (menu.style.display !== 'none') renderColMenu();
}

function renderColMenu() {
  const allCols = getAllColumnDefs();
  const visible = getVisibleCols();
  const container = document.getElementById('col-menu-items');
  container.innerHTML = '';
  allCols.forEach(col => {
    const label = document.createElement('label');
    label.style.cssText = 'display:block;padding:3px 0;font-size:12px;color:#e0e0e0;cursor:pointer;';
    const checked = visible.includes(col.key) ? 'checked' : '';
    label.innerHTML = `<input type="checkbox" ${checked} onchange="onColToggle('${col.key}', this.checked)"
                        style="accent-color:#00d4aa;margin-right:6px;"> ${col.label}`;
    container.appendChild(label);
  });
}

function getAllColumnDefs() {
  const paramNames = getParamNames();
  const cols = [{ key: 'rank', label: 'Rank' }];
  paramNames.forEach(p => cols.push({ key: 'param_' + p, label: p }));
  cols.push({ key: 'score', label: 'Score' });
  cols.push({ key: 'net_return', label: 'Return' });
  cols.push({ key: 'max_drawdown', label: 'MaxDD' });
  cols.push({ key: 'sharpe_ratio', label: 'Sharpe' });
  cols.push({ key: 'win_rate', label: 'Win Rate' });
  cols.push({ key: 'sortino_ratio', label: 'Sortino' });
  cols.push({ key: 'profit_factor', label: 'Profit Factor' });
  cols.push({ key: 'total_trades', label: 'Trades' });
  cols.push({ key: 'annual_return', label: 'Annual Return' });
  return cols;
}

function onColToggle(key, checked) {
  let cols = getVisibleCols();
  if (checked && !cols.includes(key)) cols.push(key);
  if (!checked) cols = cols.filter(c => c !== key);
  visibleCols = cols;
  localStorage.setItem('opt_visible_cols', JSON.stringify(cols));
  renderTable();
}

// Close menu on outside click
document.addEventListener('click', e => {
  if (!e.target.closest('#col-toggle-btn') && !e.target.closest('#col-menu')) {
    document.getElementById('col-menu').style.display = 'none';
  }
});
```

- [ ] **Step 3: Update renderTable() to respect visible columns**

Modify `renderTable()` to only render columns in `getVisibleCols()`:

```javascript
function renderTable() {
  const paramNames = getParamNames();
  const thead = document.getElementById('table-head');
  const tbody = document.getElementById('table-body');
  const visible = getVisibleCols();

  const allCols = getAllColumnDefs();
  let headerHtml = '<tr>';
  allCols.filter(c => visible.includes(c.key)).forEach(c => {
    headerHtml += mkTh(c.label, c.key);
  });
  headerHtml += '</tr>';
  thead.innerHTML = headerHtml;

  const sorted = [...allData];
  sorted.sort((a, b) => {
    let av, bv;
    if (sortCol === 'rank') { av = allData.indexOf(a); bv = allData.indexOf(b); }
    else if (sortCol.startsWith('param_')) { const p = sortCol.slice(6); av = a.params[p]; bv = b.params[p]; }
    else if (sortCol === 'score') { av = a.score; bv = b.score; }
    else { av = a.report[sortCol] || 0; bv = b.report[sortCol] || 0; }
    return sortDir === 'asc' ? av - bv : bv - av;
  });

  tbody.innerHTML = '';
  sorted.forEach((d, i) => {
    const origIdx = allData.indexOf(d);
    const tr = document.createElement('tr');
    tr.dataset.idx = origIdx;
    tr.onclick = () => highlightRow(origIdx);

    let html = '';
    allCols.filter(c => visible.includes(c.key)).forEach(c => {
      if (c.key === 'rank') { html += `<td>${i+1}</td>`; }
      else if (c.key.startsWith('param_')) { html += `<td>${d.params[c.key.slice(6)]}</td>`; }
      else if (c.key === 'score') { html += `<td><b>${d.score.toFixed(4)}</b></td>`; }
      else if (c.key === 'net_return') {
        const ret = (d.report.net_return || 0) * 100;
        html += `<td class="${ret >= 0 ? 'positive' : 'negative'}">${ret >= 0 ? '+' : ''}${ret.toFixed(1)}%</td>`;
      } else if (c.key === 'max_drawdown') {
        html += `<td class="negative">${((d.report.max_drawdown || 0) * 100).toFixed(1)}%</td>`;
      } else if (c.key === 'sharpe_ratio') {
        html += `<td>${(d.report.sharpe_ratio || 0).toFixed(2)}</td>`;
      } else if (c.key === 'win_rate') {
        html += `<td>${((d.report.win_rate || 0) * 100).toFixed(1)}%</td>`;
      } else if (c.key === 'sortino_ratio') {
        html += `<td>${(d.report.sortino_ratio || 0).toFixed(2)}</td>`;
      } else if (c.key === 'profit_factor') {
        html += `<td>${(d.report.profit_factor || 0).toFixed(2)}</td>`;
      } else if (c.key === 'total_trades') {
        html += `<td>${d.report.total_trades || 0}</td>`;
      } else if (c.key === 'annual_return') {
        html += `<td>${((d.report.annual_return || 0) * 100).toFixed(1)}%</td>`;
      }
    });
    tr.innerHTML = html;
    tbody.appendChild(tr);

    // Expand row
    const expandTr = document.createElement('tr');
    expandTr.className = 'expand-row';
    expandTr.style.display = 'none';
    expandTr.dataset.expandFor = origIdx;
    const colSpan = visible.length;
    let expandHtml = `<td colspan="${colSpan}"><div class="metrics">`;
    const metrics = [
      ['Sortino', (d.report.sortino_ratio || 0).toFixed(2)],
      ['Profit Factor', (d.report.profit_factor || 0).toFixed(2)],
      ['Total Trades', d.report.total_trades || 0],
      ['Annual Return', ((d.report.annual_return || 0) * 100).toFixed(1) + '%'],
    ];
    metrics.forEach(([label, val]) => {
      expandHtml += `<div class="metric-item"><span class="label">${label}</span><br/><span class="val">${val}</span></div>`;
    });
    expandHtml += '</div></td>';
    expandTr.innerHTML = expandHtml;
    tbody.appendChild(expandTr);
  });
}
```

- [ ] **Step 4: Verify manually**

Run: `python -m backtest web --port 8000`
Expected: "列设置" button opens column toggle dropdown. Preferences persist on reload.

- [ ] **Step 5: Commit**

```bash
git add src/backtest/web/static/optimize.html
git commit -m "feat: add column customization to optimize results table

Toggle visible columns via dropdown menu.
Preferences saved in localStorage."
```

---

## Task 15: Run Full Test Suite + Final Verification

**Files:** None (verification only)

- [ ] **Step 1: Run all backend tests**

Run: `pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 2: Start web server and manually verify all features**

Run: `python -m backtest web --port 8000`

Checklist:
- [ ] Report page: 3 grouped metric sections with new Calmar, streaks, volatility
- [ ] Report page: Buy & Hold dashed baseline on equity chart
- [ ] Report page: Monthly returns heatmap
- [ ] Report page: Virtual scroll trade table
- [ ] Report page: Enhanced optimize params card with batch context
- [ ] Report page: Adaptive downsampling
- [ ] Report page: Error handling with retry
- [ ] Optimize page: Batch selector with checkbox list
- [ ] Optimize page: batch_ids filter in API
- [ ] Optimize page: 3D scatter plot
- [ ] Optimize page: Parameter range sliders
- [ ] Optimize page: Column customization

- [ ] **Step 3: Commit any final fixes if needed**

- [ ] **Step 4: Final commit message summary**

```bash
git log --oneline feat/report-page-optimization ^master
```
