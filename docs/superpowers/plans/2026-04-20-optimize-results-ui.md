# Optimization Results UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a web page for visualizing parameter optimization results with interactive charts and sortable table, plus auto-save Top N results as regular reports.

**Architecture:** New `/optimize` page served by existing FastAPI app, with 2 new API routes reading from `optimize_results` table. ECharts for scatter plot and heatmap. Top 3 auto-save re-runs backtest and inserts into `reports` table.

**Tech Stack:** FastAPI, ECharts 5, SQLite, vanilla JavaScript

---

## File Structure

| File | Responsibility |
|------|---------------|
| `src/backtest/web/routes.py` | Add 3 new routes: `/optimize`, `/api/optimize_results`, `/api/optimize_results/strategies` |
| `src/backtest/web/static/optimize.html` | New page: filter bar + scatter + heatmap + table |
| `src/backtest/web/static/index.html` | Add navigation bar at top |
| `src/backtest/__main__.py` | Add Top 3 auto-save logic in `cmd_optimize` |
| `tests/test_web.py` | Tests for new API routes |
| `README.md` | API documentation section |

---

### Task 1: API Routes — Backend

**Files:**
- Modify: `src/backtest/web/routes.py`
- Modify: `tests/test_web.py`

- [ ] **Step 1: Write failing tests for new API routes**

Append to `tests/test_web.py`:

```python
@pytest.fixture
def db_with_optimize(tmp_path):
    """DB with both reports and optimize_results tables."""
    db_path = str(tmp_path / "reports.db")
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy TEXT, symbol TEXT, interval TEXT,
            created_at TEXT, report_json TEXT
        )
    """)
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
    # Insert sample optimization results
    conn.execute(
        "INSERT INTO optimize_results (strategy,symbol,interval,start_date,end_date,objective,score,params_json,report_json,created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
        ("ShadowPower", "BTCUSDT", "15m", "2024-01-01", "2024-06-30", "sharpe_ratio", 2.31,
         json.dumps({"DECISION_LEN": 40, "SHADOW_FACTOR": 2.5}),
         json.dumps({"net_return": 1.85, "max_drawdown": 0.12, "sharpe_ratio": 2.31, "win_rate": 0.55, "sortino_ratio": 3.1}),
         "2026-04-20T10:00:00+00:00"),
    )
    conn.execute(
        "INSERT INTO optimize_results (strategy,symbol,interval,start_date,end_date,objective,score,params_json,report_json,created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
        ("ShadowPower", "BTCUSDT", "15m", "2024-01-01", "2024-06-30", "sharpe_ratio", 1.85,
         json.dumps({"DECISION_LEN": 50, "SHADOW_FACTOR": 3.0}),
         json.dumps({"net_return": 1.2, "max_drawdown": 0.15, "sharpe_ratio": 1.85, "win_rate": 0.50, "sortino_ratio": 2.5}),
         "2026-04-20T10:00:00+00:00"),
    )
    conn.execute(
        "INSERT INTO optimize_results (strategy,symbol,interval,start_date,end_date,objective,score,params_json,report_json,created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
        ("MaCross", "ETHUSDT", "1h", "2024-01-01", "2024-03-01", "sharpe_ratio", 0.95,
         json.dumps({"short_period": 7, "long_period": 25}),
         json.dumps({"net_return": 0.3, "max_drawdown": 0.08, "sharpe_ratio": 0.95, "win_rate": 0.45, "sortino_ratio": 1.2}),
         "2026-04-19T10:00:00+00:00"),
    )
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def opt_client(db_with_optimize):
    app = create_app(db_with_optimize)
    return TestClient(app)


def test_get_optimize_results_all(opt_client):
    resp = opt_client.get("/api/optimize_results")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 3
    # Sorted by score DESC
    assert data[0]["score"] == 2.31
    assert data[2]["score"] == 0.95


def test_get_optimize_results_filter_strategy(opt_client):
    resp = opt_client.get("/api/optimize_results?strategy=ShadowPower")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 2
    assert all(d["strategy"] == "ShadowPower" for d in data)


def test_get_optimize_results_filter_symbol(opt_client):
    resp = opt_client.get("/api/optimize_results?symbol=ETHUSDT")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["strategy"] == "MaCross"


def test_get_optimize_strategies(opt_client):
    resp = opt_client.get("/api/optimize_results/strategies")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 2
    # Find ShadowPower entry
    sp = next(d for d in data if d["strategy"] == "ShadowPower")
    assert sp["symbol"] == "BTCUSDT"
    assert sp["count"] == 2
    assert sp["best_score"] == 2.31


def test_optimize_page(opt_client):
    resp = opt_client.get("/optimize")
    assert resp.status_code == 200
    assert "echarts" in resp.text.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_web.py::test_get_optimize_results_all -v`
Expected: FAIL (404 — route not defined)

- [ ] **Step 3: Implement API routes**

Replace `src/backtest/web/routes.py` with:

```python
import json
import sqlite3
from fastapi import APIRouter, Request, HTTPException, Query
from fastapi.responses import HTMLResponse
from pathlib import Path

router = APIRouter()


def _get_db(request: Request) -> str:
    return request.app.state.db_path


@router.get("/api/reports")
def list_reports(request: Request):
    conn = sqlite3.connect(_get_db(request))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, strategy, symbol, interval, created_at FROM reports ORDER BY id DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@router.get("/api/reports/{report_id}")
def get_report(report_id: int, request: Request):
    conn = sqlite3.connect(_get_db(request))
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM reports WHERE id = ?", (report_id,)).fetchone()
    conn.close()
    if row is None:
        raise HTTPException(status_code=404, detail="Report not found")
    report = json.loads(row["report_json"])
    report["id"] = row["id"]
    report["strategy"] = row["strategy"]
    report["symbol"] = row["symbol"]
    report["interval"] = row["interval"]
    report["created_at"] = row["created_at"]
    return report


@router.get("/api/optimize_results")
def list_optimize_results(
    request: Request,
    strategy: str | None = Query(None),
    symbol: str | None = Query(None),
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
    query += " ORDER BY score DESC"
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@router.get("/api/optimize_results/strategies")
def list_optimize_strategies(request: Request):
    conn = sqlite3.connect(_get_db(request))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT strategy, symbol, COUNT(*) as count,
               MAX(score) as best_score, MAX(created_at) as latest_date
        FROM optimize_results
        GROUP BY strategy, symbol
        ORDER BY best_score DESC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@router.get("/optimize", response_class=HTMLResponse)
def optimize_page():
    html_path = Path(__file__).parent / "static" / "optimize.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Optimize page not found")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@router.get("/", response_class=HTMLResponse)
def index():
    html_path = Path(__file__).parent / "static" / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
```

- [ ] **Step 4: Create a minimal optimize.html placeholder** (so the `/optimize` route test passes)

Create `src/backtest/web/static/optimize.html`:

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>参数优化结果</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
</head>
<body>
<p>Placeholder — will be replaced in Task 4</p>
</body>
</html>
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_web.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add src/backtest/web/routes.py src/backtest/web/static/optimize.html tests/test_web.py
git commit -m "feat(web): add optimization results API routes"
```

---

### Task 2: Top N Auto-Save to Reports

**Files:**
- Modify: `src/backtest/__main__.py`
- Modify: `tests/test_optimizer.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_optimizer.py`:

```python
class TestTopNAutoSave:
    @pytest.fixture
    def db_with_data(self):
        """Create a temp DB with 1h bars + reports table."""
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        conn = sqlite3.connect(tmp.name)
        conn.execute("""
            CREATE TABLE klines (
                exchange TEXT, symbol TEXT, interval TEXT, timestamp INTEGER,
                open REAL, high REAL, low REAL, close REAL, volume REAL,
                PRIMARY KEY (exchange, symbol, interval, timestamp)
            )
        """)
        base_ts = 1704067200000
        for i in range(200):
            ts = base_ts + i * 3600000
            price = 40000 + i * 10
            conn.execute(
                "INSERT INTO klines VALUES (?,?,?,?,?,?,?,?,?)",
                ("binance", "BTCUSDT", "1h", ts, price, price + 50, price - 50, price + 5, 1000.0),
            )
        conn.commit()
        conn.close()
        yield tmp.name
        os.unlink(tmp.name)

    def test_save_top_reports(self, db_with_data):
        from backtest.optimizer import GridSearchOptimizer, ParamSpace, save_top_reports

        space = ParamSpace({"short_period": [5, 7], "long_period": [20, 25]})
        optimizer = GridSearchOptimizer(
            db_path=db_with_data,
            strategy_path="strategies/example_ma_cross.py",
            symbol="BTCUSDT",
            interval="1h",
            start="2024-01-01",
            end="2024-01-08",
            balance=10000,
            leverage=10,
            param_space=space,
            objective="sharpe_ratio",
            n_jobs=1,
        )
        result = optimizer.run()

        report_db = db_with_data.replace(".db", "_reports.db")
        save_top_reports(
            result=result,
            top_n=2,
            db_path=db_with_data,
            report_db_path=report_db,
            strategy_path="strategies/example_ma_cross.py",
            symbol="BTCUSDT",
            interval="1h",
            start="2024-01-01",
            end="2024-01-08",
            balance=10000,
            leverage=10,
        )

        conn = sqlite3.connect(report_db)
        rows = conn.execute("SELECT strategy, report_json FROM reports").fetchall()
        conn.close()
        os.unlink(report_db)

        assert len(rows) == 2
        assert "MaCrossStrategy_opt1" in rows[0][0]
        report = json.loads(rows[0][1])
        assert "equity_curve" in report
        assert "trades" in report
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_optimizer.py::TestTopNAutoSave -v`
Expected: FAIL (ImportError — save_top_reports not defined)

- [ ] **Step 3: Implement save_top_reports**

Add to `src/backtest/optimizer.py`:

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

    conn = sqlite3.connect(report_db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy TEXT, symbol TEXT, interval TEXT,
            created_at TEXT, report_json TEXT
        )
    """)

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
        conn.execute(
            "INSERT INTO reports (strategy, symbol, interval, created_at, report_json) VALUES (?,?,?,?,?)",
            (strategy_name, symbol, interval, now, json.dumps(report)),
        )

    conn.commit()
    conn.close()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_optimizer.py::TestTopNAutoSave -v`
Expected: PASS

- [ ] **Step 5: Integrate into cmd_optimize**

In `src/backtest/__main__.py`, add at the end of `cmd_optimize()` (after the existing `save_results` call):

```python
    # Auto-save top 3 as full reports
    from backtest.optimizer import save_top_reports
    save_top_reports(
        result=result,
        top_n=min(3, len(result.all_trials)),
        db_path=args.db or str(Path("data") / "klines.db"),
        report_db_path=report_db,
        strategy_path=args.strategy,
        symbol=args.symbol,
        interval=args.interval,
        start=args.start,
        end=args.end,
        balance=args.balance,
        leverage=args.leverage,
    )
    print(f"Top 3 results saved as reports. View with: python -m backtest web")
```

- [ ] **Step 6: Run all tests**

Run: `pytest tests/ -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add src/backtest/optimizer.py src/backtest/__main__.py tests/test_optimizer.py
git commit -m "feat(optimizer): auto-save top N results as full reports"
```

---

### Task 3: Navigation Bar on index.html

**Files:**
- Modify: `src/backtest/web/static/index.html`

- [ ] **Step 1: Add navigation to index.html**

Replace the existing `<div class="header">` block (lines 48-51) with:

```html
<div class="header">
  <div style="display:flex;align-items:center;gap:24px;">
    <h1>Crypto Backtest</h1>
    <nav style="display:flex;gap:16px;">
      <a href="/" style="color:#00d4aa;text-decoration:none;font-size:14px;border-bottom:2px solid #00d4aa;padding-bottom:2px;">回测报告</a>
      <a href="/optimize" style="color:#888;text-decoration:none;font-size:14px;">参数优化</a>
    </nav>
  </div>
  <div class="meta" id="meta-info"></div>
</div>
```

- [ ] **Step 2: Verify page still renders**

Run: `pytest tests/test_web.py::test_index_page -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add src/backtest/web/static/index.html
git commit -m "feat(web): add navigation bar to index.html"
```

---

### Task 4: Optimization Results Page (optimize.html)

**Files:**
- Modify: `src/backtest/web/static/optimize.html` (replace placeholder)

- [ ] **Step 1: Write the full optimize.html page**

Replace `src/backtest/web/static/optimize.html` with:

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>参数优化结果</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #0f1117; color: #e0e0e0; }
  .header { background: #1a1a2e; padding: 16px 24px; display: flex;
            justify-content: space-between; align-items: center;
            border-bottom: 1px solid #333; }
  .header h1 { font-size: 18px; color: #00d4aa; }
  .nav { display: flex; gap: 16px; }
  .nav a { color: #888; text-decoration: none; font-size: 14px; padding-bottom: 2px; }
  .nav a.active { color: #00d4aa; border-bottom: 2px solid #00d4aa; }
  .filter-bar { display: flex; gap: 12px; padding: 16px 24px; flex-wrap: wrap; align-items: center; }
  .filter-bar select { background: #1e293b; color: #e0e0e0; border: 1px solid #333;
                       padding: 8px 12px; border-radius: 6px; font-size: 13px; }
  .filter-bar .badge { background: #00d4aa22; color: #00d4aa; padding: 4px 10px;
                       border-radius: 12px; font-size: 12px; }
  .charts-row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; padding: 0 24px 12px; }
  .chart-box { background: #1e293b; border-radius: 8px; padding: 16px; }
  .chart-box h3 { font-size: 13px; color: #aaa; margin-bottom: 8px; }
  .chart-controls { display: flex; gap: 8px; margin-bottom: 8px; }
  .chart-controls select { background: #0f1117; color: #e0e0e0; border: 1px solid #333;
                           padding: 4px 8px; border-radius: 4px; font-size: 12px; }
  .table-wrap { padding: 0 24px 24px; }
  .table-container { background: #1e293b; border-radius: 8px; overflow: hidden; }
  .opt-table { width: 100%; border-collapse: collapse; font-size: 13px; }
  .opt-table th { padding: 12px 10px; text-align: left; color: #888;
                  border-bottom: 1px solid #333; cursor: pointer; user-select: none;
                  position: sticky; top: 0; background: #1e293b; }
  .opt-table th:hover { color: #00d4aa; }
  .opt-table th .sort-arrow { margin-left: 4px; font-size: 10px; }
  .opt-table td { padding: 10px; border-bottom: 1px solid #0f1117; }
  .opt-table tr:hover { background: #253347; }
  .opt-table tr.selected { background: #1a3a4a; }
  .opt-table .positive { color: #22c55e; }
  .opt-table .negative { color: #ef4444; }
  .expand-row td { padding: 12px 20px; background: #162032; }
  .expand-row .metrics { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; }
  .expand-row .metric-item { font-size: 12px; }
  .expand-row .metric-item .label { color: #888; }
  .expand-row .metric-item .val { color: #e0e0e0; font-weight: 600; }
  .table-scroll { max-height: 500px; overflow-y: auto; }
  .no-data { text-align: center; padding: 60px; color: #888; }
  @media (max-width: 900px) { .charts-row { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<div class="header">
  <div style="display:flex;align-items:center;gap:24px;">
    <h1>Crypto Backtest</h1>
    <nav class="nav">
      <a href="/">回测报告</a>
      <a href="/optimize" class="active">参数优化</a>
    </nav>
  </div>
</div>

<!-- Filter Bar -->
<div class="filter-bar">
  <select id="filter-strategy"><option value="">全部策略</option></select>
  <select id="filter-symbol"><option value="">全部交易对</option></select>
  <span class="badge" id="badge-objective"></span>
  <span class="badge" id="badge-count"></span>
</div>

<!-- Charts -->
<div class="charts-row">
  <div class="chart-box">
    <h3>风险-收益散点图</h3>
    <div id="chart-scatter" style="height:300px;"></div>
  </div>
  <div class="chart-box">
    <h3>参数热力图</h3>
    <div class="chart-controls">
      <select id="heatmap-x"></select>
      <select id="heatmap-y"></select>
    </div>
    <div id="chart-heatmap" style="height:270px;"></div>
  </div>
</div>

<!-- Table -->
<div class="table-wrap">
  <div class="table-container">
    <div class="table-scroll">
      <table class="opt-table">
        <thead id="table-head"></thead>
        <tbody id="table-body"></tbody>
      </table>
    </div>
  </div>
</div>

<div class="no-data" id="no-data" style="display:none;">暂无优化结果，请先运行参数优化</div>

<script>
let allData = [];
let scatterChart, heatmapChart;
let sortCol = 'score', sortDir = 'desc';
let selectedRow = null;

// Init
fetch('/api/optimize_results/strategies').then(r => r.json()).then(strategies => {
  const sel = document.getElementById('filter-strategy');
  const symSet = new Set();
  strategies.forEach(s => {
    const opt = document.createElement('option');
    opt.value = s.strategy + '|' + s.symbol;
    opt.textContent = `${s.strategy} (${s.symbol}) — ${s.count}组 best=${s.best_score.toFixed(2)}`;
    sel.appendChild(opt);
    symSet.add(s.symbol);
  });
  const symSel = document.getElementById('filter-symbol');
  symSet.forEach(sym => {
    const opt = document.createElement('option');
    opt.value = sym;
    opt.textContent = sym;
    symSel.appendChild(opt);
  });
  loadData();
});

document.getElementById('filter-strategy').onchange = loadData;
document.getElementById('filter-symbol').onchange = loadData;

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

function getParamNames() {
  if (!allData.length) return [];
  return Object.keys(allData[0].params);
}

// Scatter Plot
function renderScatter() {
  if (!scatterChart) scatterChart = echarts.init(document.getElementById('chart-scatter'));
  const data = allData.map((d, i) => ({
    value: [(d.report.net_return || 0) * 100, (d.report.max_drawdown || 0) * 100, d.score, i],
    params: d.params,
    report: d.report,
  }));
  const scores = data.map(d => d.value[2]);
  const minScore = Math.min(...scores);
  const maxScore = Math.max(...scores);

  scatterChart.setOption({
    tooltip: {
      formatter: p => {
        const d = p.data;
        let s = `<b>Score: ${d.value[2].toFixed(4)}</b><br/>`;
        s += `收益: ${d.value[0].toFixed(1)}% | 回撤: ${d.value[1].toFixed(1)}%<br/>`;
        Object.entries(d.params).forEach(([k,v]) => { s += `${k}: ${v}<br/>`; });
        return s;
      }
    },
    xAxis: { name: 'Return %', nameLocation: 'center', nameGap: 25,
             axisLabel: { color: '#888' }, splitLine: { lineStyle: { color: '#222' }} },
    yAxis: { name: 'MaxDD %', nameLocation: 'center', nameGap: 35,
             axisLabel: { color: '#888' }, splitLine: { lineStyle: { color: '#222' }} },
    series: [{
      type: 'scatter',
      data: data,
      symbolSize: d => {
        const norm = maxScore > minScore ? (d[2] - minScore) / (maxScore - minScore) : 0.5;
        return 8 + norm * 20;
      },
      itemStyle: {
        color: p => {
          const norm = maxScore > minScore ? (p.data.value[2] - minScore) / (maxScore - minScore) : 0.5;
          const r = Math.round(239 - norm * 217);
          const g = Math.round(68 + norm * 129);
          const b = Math.round(68 - norm * 24);
          return `rgb(${r},${g},${b})`;
        }
      }
    }],
    grid: { left: 50, right: 20, top: 20, bottom: 40 },
  }, true);

  scatterChart.off('click');
  scatterChart.on('click', p => {
    const idx = p.data.value[3];
    highlightRow(idx);
  });
}

// Heatmap
function initHeatmapControls() {
  const names = getParamNames();
  const xSel = document.getElementById('heatmap-x');
  const ySel = document.getElementById('heatmap-y');
  xSel.innerHTML = '';
  ySel.innerHTML = '';
  names.forEach((n, i) => {
    xSel.innerHTML += `<option value="${n}" ${i===0?'selected':''}>${n}</option>`;
    ySel.innerHTML += `<option value="${n}" ${i===1?'selected':''}>${n}</option>`;
  });
  xSel.onchange = renderHeatmap;
  ySel.onchange = renderHeatmap;
}

function renderHeatmap() {
  if (!heatmapChart) heatmapChart = echarts.init(document.getElementById('chart-heatmap'));
  const xParam = document.getElementById('heatmap-x').value;
  const yParam = document.getElementById('heatmap-y').value;
  if (!xParam || !yParam || xParam === yParam) {
    heatmapChart.clear();
    return;
  }

  // Get unique values for each axis
  const xVals = [...new Set(allData.map(d => d.params[xParam]))].sort((a,b) => a-b);
  const yVals = [...new Set(allData.map(d => d.params[yParam]))].sort((a,b) => a-b);

  // Build heatmap data: for each (x,y) pair, take best score
  const heatData = [];
  let minVal = Infinity, maxVal = -Infinity;
  xVals.forEach((xv, xi) => {
    yVals.forEach((yv, yi) => {
      const matches = allData.filter(d => d.params[xParam] === xv && d.params[yParam] === yv);
      if (matches.length > 0) {
        const best = Math.max(...matches.map(m => m.score));
        heatData.push([xi, yi, best]);
        if (best < minVal) minVal = best;
        if (best > maxVal) maxVal = best;
      }
    });
  });

  heatmapChart.setOption({
    tooltip: {
      formatter: p => `${xParam}=${xVals[p.data[0]]}, ${yParam}=${yVals[p.data[1]]}<br/>Score: ${p.data[2].toFixed(4)}`
    },
    xAxis: { type: 'category', data: xVals.map(String), axisLabel: { color: '#888', fontSize: 11 },
             name: xParam, nameLocation: 'center', nameGap: 25 },
    yAxis: { type: 'category', data: yVals.map(String), axisLabel: { color: '#888', fontSize: 11 },
             name: yParam, nameLocation: 'center', nameGap: 40 },
    visualMap: { min: minVal, max: maxVal, calculable: true, orient: 'vertical', right: 0, top: 'center',
                 textStyle: { color: '#888' }, inRange: { color: ['#ef4444', '#f59e0b', '#22c55e'] } },
    series: [{ type: 'heatmap', data: heatData, label: { show: heatData.length <= 50, color: '#fff', fontSize: 10,
               formatter: p => p.data[2].toFixed(2) },
               emphasis: { itemStyle: { shadowBlur: 10, shadowColor: 'rgba(0,0,0,0.5)' } } }],
    grid: { left: 60, right: 80, top: 10, bottom: 40 },
  }, true);
}

// Table
function renderTable() {
  const paramNames = getParamNames();
  const thead = document.getElementById('table-head');
  const tbody = document.getElementById('table-body');

  // Header
  let headerHtml = '<tr>';
  headerHtml += mkTh('Rank', 'rank');
  paramNames.forEach(p => { headerHtml += mkTh(p, 'param_' + p); });
  headerHtml += mkTh('Score', 'score');
  headerHtml += mkTh('Return', 'net_return');
  headerHtml += mkTh('MaxDD', 'max_drawdown');
  headerHtml += mkTh('Sharpe', 'sharpe_ratio');
  headerHtml += mkTh('Win Rate', 'win_rate');
  headerHtml += '</tr>';
  thead.innerHTML = headerHtml;

  // Sort data
  const sorted = [...allData];
  sorted.sort((a, b) => {
    let av, bv;
    if (sortCol === 'rank') { av = allData.indexOf(a); bv = allData.indexOf(b); }
    else if (sortCol.startsWith('param_')) { const p = sortCol.slice(6); av = a.params[p]; bv = b.params[p]; }
    else if (sortCol === 'score') { av = a.score; bv = b.score; }
    else { av = a.report[sortCol] || 0; bv = b.report[sortCol] || 0; }
    return sortDir === 'asc' ? av - bv : bv - av;
  });

  // Body
  tbody.innerHTML = '';
  sorted.forEach((d, i) => {
    const origIdx = allData.indexOf(d);
    const tr = document.createElement('tr');
    tr.dataset.idx = origIdx;
    tr.onclick = () => highlightRow(origIdx);

    let html = `<td>${i+1}</td>`;
    paramNames.forEach(p => { html += `<td>${d.params[p]}</td>`; });
    html += `<td><b>${d.score.toFixed(4)}</b></td>`;
    const ret = (d.report.net_return || 0) * 100;
    html += `<td class="${ret >= 0 ? 'positive' : 'negative'}">${ret >= 0 ? '+' : ''}${ret.toFixed(1)}%</td>`;
    const dd = (d.report.max_drawdown || 0) * 100;
    html += `<td class="negative">${dd.toFixed(1)}%</td>`;
    html += `<td>${(d.report.sharpe_ratio || 0).toFixed(2)}</td>`;
    const wr = (d.report.win_rate || 0) * 100;
    html += `<td>${wr.toFixed(1)}%</td>`;
    tr.innerHTML = html;
    tbody.appendChild(tr);

    // Expand row (hidden by default)
    const expandTr = document.createElement('tr');
    expandTr.className = 'expand-row';
    expandTr.style.display = 'none';
    expandTr.dataset.expandFor = origIdx;
    const colSpan = paramNames.length + 6;
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

function mkTh(label, col) {
  const arrow = sortCol === col ? (sortDir === 'asc' ? ' ▲' : ' ▼') : '';
  return `<th onclick="sortBy('${col}')">${label}<span class="sort-arrow">${arrow}</span></th>`;
}

function sortBy(col) {
  if (sortCol === col) { sortDir = sortDir === 'asc' ? 'desc' : 'asc'; }
  else { sortCol = col; sortDir = 'desc'; }
  renderTable();
}

function highlightRow(idx) {
  // Table highlight
  document.querySelectorAll('.opt-table tr.selected').forEach(tr => tr.classList.remove('selected'));
  document.querySelectorAll('.expand-row').forEach(tr => tr.style.display = 'none');

  const row = document.querySelector(`.opt-table tr[data-idx="${idx}"]`);
  if (row) {
    row.classList.add('selected');
    row.scrollIntoView({ block: 'nearest' });
    const expandRow = document.querySelector(`.expand-row[data-expand-for="${idx}"]`);
    if (expandRow) expandRow.style.display = '';
  }

  // Scatter highlight
  if (scatterChart) {
    scatterChart.dispatchAction({ type: 'downplay', seriesIndex: 0 });
    scatterChart.dispatchAction({ type: 'highlight', seriesIndex: 0, dataIndex: idx });
  }
  selectedRow = idx;
}

// Resize
window.addEventListener('resize', () => {
  if (scatterChart) scatterChart.resize();
  if (heatmapChart) heatmapChart.resize();
});
</script>
</body>
</html>
```

- [ ] **Step 2: Verify page loads**

Run: `pytest tests/test_web.py::test_optimize_page -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add src/backtest/web/static/optimize.html
git commit -m "feat(web): add optimization results page with charts and table"
```

---

### Task 5: README API Documentation

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Read current README.md to find insertion point**

The README is at `/Users/zhaowei/GitHub/crypto_backtest/README.md`. Find the end of the file or an appropriate section to add API docs.

- [ ] **Step 2: Append API Reference section to README.md**

Add at the end of the file:

```markdown
## API Reference

The web server (`python -m backtest web`) exposes the following endpoints:

### Pages

| Endpoint | Description |
|----------|-------------|
| `GET /` | 回测报告查看器 |
| `GET /optimize` | 参数优化结果页面 |

### Backtest Reports

| Endpoint | Description |
|----------|-------------|
| `GET /api/reports` | 列出所有回测报告 (id, strategy, symbol, interval, created_at) |
| `GET /api/reports/{id}` | 获取单个报告详情 (含 equity_curve, trades, 所有指标) |

### Optimization Results

| Endpoint | Parameters | Description |
|----------|-----------|-------------|
| `GET /api/optimize_results` | `?strategy=X&symbol=Y` | 列出优化结果，按 score 降序。支持策略名和交易对筛选 |
| `GET /api/optimize_results/strategies` | — | 列出所有策略/交易对组合 (含 count, best_score, latest_date) |

### Response Examples

**GET /api/optimize_results**

```json
[
  {
    "id": 1,
    "strategy": "ShadowPowerStrategy",
    "symbol": "BTCUSDT",
    "interval": "15m",
    "start_date": "2024-01-01",
    "end_date": "2024-06-30",
    "objective": "sharpe_ratio",
    "score": 2.31,
    "params_json": "{\"DECISION_LEN\": 40, \"SHADOW_FACTOR\": 2.5}",
    "report_json": "{\"net_return\": 1.85, \"max_drawdown\": 0.12, ...}",
    "created_at": "2026-04-20T10:00:00+00:00"
  }
]
```

**GET /api/optimize_results/strategies**

```json
[
  {
    "strategy": "ShadowPowerStrategy",
    "symbol": "BTCUSDT",
    "count": 120,
    "best_score": 2.31,
    "latest_date": "2026-04-20T10:00:00+00:00"
  }
]
```
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: add API reference section to README"
```

---

### Task 6: Integration Smoke Test

**Files:**
- No new files; manual verification

- [ ] **Step 1: Run full test suite**

Run: `pytest tests/ -v`
Expected: All PASS

- [ ] **Step 2: Start web server and verify pages load**

Run: `python -m backtest web --port 8001 &`

Verify:
- `curl http://localhost:8001/` returns HTML with navigation links
- `curl http://localhost:8001/optimize` returns HTML with echarts
- `curl http://localhost:8001/api/optimize_results` returns JSON array
- `curl http://localhost:8001/api/optimize_results/strategies` returns JSON array

Kill server after verification.

- [ ] **Step 3: Run a real optimization and verify end-to-end**

```bash
python -m backtest optimize \
    --strategy strategies/example_ma_cross.py \
    --symbol BTCUSDT --interval 1h \
    --start 2024-01-01 --end 2024-03-01 \
    --balance 10000 --leverage 10 \
    --params "short_period=5:9:2,long_period=20:30:5" \
    --objective sharpe_ratio --method grid --n-jobs 4
```

Then start web server and verify:
- `/optimize` page shows the results
- Scatter plot and heatmap render
- Table is sortable
- Top 3 appear in report selector on `/`

- [ ] **Step 4: Commit any fixes if needed**
