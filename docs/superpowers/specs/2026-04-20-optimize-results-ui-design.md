# Optimization Results UI Design

## Overview

Add a dedicated web page for visualizing parameter optimization results, with interactive charts (heatmap + scatter) and a sortable table. Integrate into the existing `python -m backtest web` service. Additionally, Top N results auto-save as regular reports for viewing in the existing report page.

## Scope

- New file: `src/backtest/web/static/optimize.html`
- Modify: `src/backtest/web/static/index.html` (add navigation)
- Modify: `src/backtest/web/routes.py` (add 3 routes)
- Modify: `src/backtest/__main__.py` (Top 3 auto-save to reports table)
- Modify: `README.md` (API documentation)

## Page Layout

```
┌─────────────────────────────────────────────────┐
│ Navigation: [回测报告]  [参数优化]              │
├─────────────────────────────────────────────────┤
│ Filter bar: Strategy | Symbol | Date | Objective│
├───────────────────────┬─────────────────────────┤
│ Scatter Plot          │ Heatmap                 │
│ X=net_return          │ X=paramA, Y=paramB      │
│ Y=max_drawdown        │ Color=score             │
│ Size=score            │                         │
├───────────────────────┴─────────────────────────┤
│ Sortable Table                                   │
│ Rank | Param1 | Param2 | Score | Return | DD    │
│ Click row → highlight point in charts           │
│ Expand row → full report metrics                │
└─────────────────────────────────────────────────┘
```

## API Routes

### GET /api/optimize_results

Returns optimization trial data for display.

**Query Parameters:**
- `strategy` (optional): Filter by strategy name
- `symbol` (optional): Filter by trading pair
- `start_date` (optional): Filter by optimization start date
- `end_date` (optional): Filter by optimization end date

**Response:** Array of trial objects, sorted by score DESC.

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
    "params_json": "{\"DECISION_LEN\":40,\"SHADOW_FACTOR\":2.5}",
    "report_json": "{\"net_return\":1.85,\"max_drawdown\":0.123,...}",
    "created_at": "2026-04-20T10:00:00+00:00"
  }
]
```

### GET /api/optimize_results/strategies

Returns distinct strategy/symbol combinations for the filter dropdowns.

**Response:**

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

### GET /optimize

Returns the `optimize.html` page.

## Frontend Components

### Scatter Plot (ECharts)

- X-axis: `net_return` (from report_json)
- Y-axis: `max_drawdown` (from report_json)
- Point size: proportional to `score`
- Color: green gradient (higher score = darker green)
- Hover tooltip: shows parameter combination + all metrics
- Click: highlights corresponding table row, scrolls into view

### Heatmap (ECharts)

- Two dropdowns above the chart to select X-axis parameter and Y-axis parameter
- Parameter options dynamically extracted from `params_json` keys
- Cell color: score value (green=high, red=low)
- When >2 parameters exist: other parameters are sliced at their best-performing values
- Hover tooltip: shows full parameter set + score

### Sortable Table

- Columns: Rank, dynamic parameter columns (from params_json keys), Score, Return, MaxDD, Sharpe, Win Rate
- Click column header to sort (asc/desc toggle)
- Click row: highlight corresponding point in scatter plot (bold border), flash corresponding cell in heatmap
- Expand row (click arrow): show full report metrics in a sub-row

### Filter Bar

- Strategy dropdown (populated from `/api/optimize_results/strategies`)
- Symbol dropdown
- Objective badge (display only, from data)
- On filter change: re-fetch data, re-render all charts and table

## Navigation

Both `index.html` and `optimize.html` share a top navigation bar:

```html
<nav class="top-nav">
  <a href="/" class="nav-link">回测报告</a>
  <a href="/optimize" class="nav-link active">参数优化</a>
</nav>
```

Active link highlighted with accent color (`#00d4aa`).

## Styling

- Dark theme consistent with existing `index.html`:
  - Background: `#0f1117`
  - Card background: `#1e293b`
  - Text: `#e0e0e0`
  - Accent: `#00d4aa`
  - Positive: `#22c55e`, Negative: `#ef4444`
- ECharts dark theme
- Responsive: charts stack vertically on narrow screens

## Top N Auto-Save to Reports

In `cmd_optimize()`, after optimization completes:

1. Take Top 3 trials (configurable via `--save-top`, default 3)
2. For each, run a full backtest with those params (to get equity_curve and trades)
3. Save as regular report in `reports` table with strategy name suffixed: `ShadowPowerStrategy_opt1`
4. User can then view these in the existing report page with full equity curves

This re-runs the backtest to capture the full equity_curve and trades data that was stripped during optimization (for storage efficiency).

## File Structure

```
src/backtest/web/
├── static/
│   ├── index.html        # Modified: add navigation bar
│   └── optimize.html     # New: optimization results page
├── routes.py             # Modified: add 3 new routes
└── app.py                # No changes

src/backtest/
└── __main__.py           # Modified: Top 3 auto-save logic in cmd_optimize

README.md                 # Modified: add API documentation section
```

## README API Documentation

Add an "API Reference" section to README.md documenting all routes:

- `GET /` — Report viewer page
- `GET /optimize` — Optimization results page
- `GET /api/reports` — List all backtest reports
- `GET /api/reports/{id}` — Get single report details
- `GET /api/optimize_results` — List optimization trials (with filters)
- `GET /api/optimize_results/strategies` — List distinct strategy/symbol combos
