# Report Optimize Params Display Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show the optimization parameters (params + score + objective) that generated a report, as a card on the report detail page.

**Architecture:** Backend extends `GET /api/reports/{id}` with a LEFT JOIN on `optimize_results` to return `optimize_params`, `optimize_score`, and `optimize_objective`. Frontend adds a hidden card that renders when these fields are present. Reports from standalone `run` command (no linkage) simply don't show the card.

**Tech Stack:** Python 3.11, FastAPI, sqlite3, vanilla JS, no new dependencies

---

## File Map

| File | Change |
|------|--------|
| `src/backtest/web/routes.py` | Extend `get_report` with LEFT JOIN, return 3 new fields |
| `src/backtest/web/static/index.html` | Add CSS + HTML card + JS render logic |
| `tests/test_web.py` | Add 2 tests: linked report returns params, unlinked returns nulls |

---

### Task 1: Backend — extend `get_report` with optimize params

**Files:**
- Modify: `src/backtest/web/routes.py` — `get_report` function
- Test: `tests/test_web.py`

- [ ] **Step 1: Write the failing tests**

Add these two tests at the end of `tests/test_web.py`.

First, add a new fixture `db_with_linked_report` after the existing `db_with_optimize` fixture:

```python
@pytest.fixture
def db_with_linked_report(tmp_path):
    """DB with a report linked to an optimize_results row via optimize_result_id."""
    db_path = str(tmp_path / "reports.db")
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE optimize_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy TEXT NOT NULL, symbol TEXT NOT NULL,
            interval TEXT NOT NULL, start_date TEXT NOT NULL,
            end_date TEXT NOT NULL, objective TEXT NOT NULL,
            score REAL NOT NULL, params_json TEXT NOT NULL,
            report_json TEXT NOT NULL, created_at TEXT NOT NULL
        )
    """)
    conn.execute(
        "INSERT INTO optimize_results (strategy,symbol,interval,start_date,end_date,objective,score,params_json,report_json,created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
        ("TestStrategy", "BTCUSDT", "1h", "2024-01-01", "2024-12-31",
         "sharpe_ratio", 2.04,
         json.dumps({"CONSECUTIVE_THRESHOLD": 5, "POSITION_MULTIPLIER": 1.1}, sort_keys=True),
         json.dumps({}),
         "2026-04-21T00:00:00+00:00"),
    )
    conn.execute("""
        CREATE TABLE reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy TEXT, symbol TEXT, interval TEXT,
            created_at TEXT, report_json TEXT,
            optimize_result_id INTEGER
        )
    """)
    report = {
        "net_return": 0.5, "max_drawdown": 0.1, "sharpe_ratio": 2.04,
        "win_rate": 0.6, "total_trades": 20, "equity_curve": [], "trades": [],
    }
    conn.execute(
        "INSERT INTO reports (strategy,symbol,interval,created_at,report_json,optimize_result_id) VALUES (?,?,?,?,?,?)",
        ("TestStrategy_opt1", "BTCUSDT", "1h", "2026-04-21T00:00:00+00:00",
         json.dumps(report), 1),
    )
    # Unlinked report (standalone run)
    conn.execute(
        "INSERT INTO reports (strategy,symbol,interval,created_at,report_json,optimize_result_id) VALUES (?,?,?,?,?,?)",
        ("TestStrategy", "BTCUSDT", "1h", "2026-04-21T01:00:00+00:00",
         json.dumps(report), None),
    )
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def linked_client(db_with_linked_report):
    app = create_app(db_with_linked_report)
    return TestClient(app)
```

Then add the two tests:

```python
def test_get_report_includes_optimize_params_when_linked(linked_client):
    resp = linked_client.get("/api/reports/1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["optimize_params"] == {"CONSECUTIVE_THRESHOLD": 5, "POSITION_MULTIPLIER": 1.1}
    assert data["optimize_score"] == pytest.approx(2.04)
    assert data["optimize_objective"] == "sharpe_ratio"


def test_get_report_optimize_params_null_when_unlinked(linked_client):
    resp = linked_client.get("/api/reports/2")
    assert resp.status_code == 200
    data = resp.json()
    assert data["optimize_params"] is None
    assert data["optimize_score"] is None
    assert data["optimize_objective"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/zhaowei/GitHub/crypto_backtest && source .venv/bin/activate && pytest tests/test_web.py::test_get_report_includes_optimize_params_when_linked tests/test_web.py::test_get_report_optimize_params_null_when_unlinked -v
```

Expected: both `FAILED` — `KeyError: 'optimize_params'`

- [ ] **Step 3: Update `get_report` in `routes.py`**

Replace the entire `get_report` function (currently lines 25–39) with:

```python
@router.get("/api/reports/{report_id}")
def get_report(report_id: int, request: Request):
    conn = sqlite3.connect(_get_db(request))
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        """SELECT r.report_json, r.strategy, r.symbol, r.interval, r.created_at,
                  o.params_json AS optimize_params_json,
                  o.score       AS optimize_score,
                  o.objective   AS optimize_objective
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
    return report
```

- [ ] **Step 4: Run the new tests — both must pass**

```bash
cd /Users/zhaowei/GitHub/crypto_backtest && source .venv/bin/activate && pytest tests/test_web.py::test_get_report_includes_optimize_params_when_linked tests/test_web.py::test_get_report_optimize_params_null_when_unlinked -v
```

Expected: both `PASSED`

- [ ] **Step 5: Run the full test suite**

```bash
cd /Users/zhaowei/GitHub/crypto_backtest && source .venv/bin/activate && pytest tests/ -v
```

Expected: all previously passing tests still pass.

- [ ] **Step 6: Commit**

```bash
cd /Users/zhaowei/GitHub/crypto_backtest && git add src/backtest/web/routes.py tests/test_web.py && git commit -m "feat(web): include optimize_params in report detail API"
```

---

### Task 2: Frontend — add optimize params card to `index.html`

**Files:**
- Modify: `src/backtest/web/static/index.html`

No automated test for HTML/JS — manual verification steps provided.

- [ ] **Step 1: Add CSS for the params card**

Inside the `<style>` block, after the last rule (`.report-select { ... }`), add:

```css
  .opt-params-card { background: #1e293b; border-radius: 8px; padding: 16px;
                     margin: 0 24px 12px; }
  .opt-params-card .card-header { display: flex; justify-content: space-between;
                                   align-items: center; margin-bottom: 12px; }
  .opt-params-card .card-title { font-size: 13px; color: #aaa; }
  .opt-params-card .card-score { font-size: 13px; color: #00d4aa; }
  .opt-params-card .param-row { display: flex; justify-content: space-between;
                                 padding: 5px 0; border-bottom: 1px solid #2a3a4a; }
  .opt-params-card .param-row:last-child { border-bottom: none; }
  .opt-params-card .param-name { font-size: 12px; color: #888; }
  .opt-params-card .param-value { font-size: 13px; font-weight: 600; color: #e0e0e0; }
```

- [ ] **Step 2: Add the card HTML**

Inside `<div id="content" style="display:none;">`, after `<div class="metrics-grid" id="metrics-grid"></div>` and before `<!-- 图表 -->`, insert:

```html
  <!-- 优化参数 -->
  <div id="opt-params-card" class="opt-params-card" style="display:none;">
    <div class="card-header">
      <span class="card-title">优化参数</span>
      <span class="card-score" id="opt-score-label"></span>
    </div>
    <div id="opt-params-body"></div>
  </div>
```

- [ ] **Step 3: Add JS render logic in `loadReport()`**

Inside the `loadReport()` function, after the metrics grid block (after the `metricsData.forEach(...)` loop, before the `// 权益曲线` comment), add:

```javascript
    // 优化参数卡片
    const optCard = document.getElementById('opt-params-card');
    if (d.optimize_params) {
      const scoreLabel = document.getElementById('opt-score-label');
      scoreLabel.textContent = `score: ${d.optimize_score.toFixed(4)} | ${d.optimize_objective}`;
      const body = document.getElementById('opt-params-body');
      body.innerHTML = '';
      Object.entries(d.optimize_params).forEach(([k, v]) => {
        const row = document.createElement('div');
        row.className = 'param-row';
        row.innerHTML = `<span class="param-name">${k}</span><span class="param-value">${v}</span>`;
        body.appendChild(row);
      });
      optCard.style.display = '';
    } else {
      optCard.style.display = 'none';
    }
```

- [ ] **Step 4: Manual verification**

Start the server:
```bash
cd /Users/zhaowei/GitHub/crypto_backtest && source .venv/bin/activate && python -m backtest web --port 8000
```

Open `http://localhost:8000` in a browser:
1. Select a report that was generated from optimization (strategy name ends with `_opt1/2/3`) → the **优化参数** card should appear between the metrics grid and charts, showing params and score
2. Select a report generated from standalone `run` command → the card should be hidden

- [ ] **Step 5: Commit**

```bash
cd /Users/zhaowei/GitHub/crypto_backtest && git add src/backtest/web/static/index.html && git commit -m "feat(web): add optimize params card to report detail page"
```

---

## Self-Review

**Spec coverage:**
- ✅ LEFT JOIN optimize_results in get_report
- ✅ Returns `optimize_params` (dict or null), `optimize_score` (float or null), `optimize_objective` (str or null)
- ✅ Card hidden when no linkage (standalone run)
- ✅ Card shows params as key/value rows + score + objective in header
- ✅ No breaking changes to existing API consumers

**Placeholder scan:** None found.

**Type consistency:** `optimize_params`, `optimize_score`, `optimize_objective` — consistent across backend, tests, and frontend JS.
