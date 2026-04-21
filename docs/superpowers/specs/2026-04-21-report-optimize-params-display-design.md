# Design: Show Optimize Params on Report Page

**Date:** 2026-04-21
**Status:** Approved

## Problem

The report detail page shows backtest metrics and charts, but gives no indication of which optimization parameters produced the result. The `reports.optimize_result_id` foreign key (added 2026-04-21) enables precise linkage, but the web layer doesn't expose it yet.

## Solution

### Backend — `routes.py`

Extend `GET /api/reports/{id}` to LEFT JOIN `optimize_results` and return three new fields:

```sql
SELECT r.report_json, r.strategy, r.symbol, r.interval, r.created_at,
       r.optimize_result_id,
       o.params_json AS optimize_params_json,
       o.score       AS optimize_score,
       o.objective   AS optimize_objective
FROM reports r
LEFT JOIN optimize_results o ON r.optimize_result_id = o.id
WHERE r.id = ?
```

New fields in the JSON response:
- `optimize_params` — dict (parsed from `params_json`) or `null` if no linkage
- `optimize_score` — float or `null`
- `optimize_objective` — string or `null`

### Frontend — `index.html`

Insert a new card between the metrics grid and the charts section. Hidden when `optimize_params` is null.

**Layout:**
```
┌─ 优化参数  (score: 2.04 | sharpe_ratio) ──────────────┐
│  CONSECUTIVE_THRESHOLD    5                            │
│  POSITION_MULTIPLIER      1.10                         │
│  INITIAL_POSITION_PCT     0.01                         │
│  PROFIT_CANDLE_THRESHOLD  1                            │
└────────────────────────────────────────────────────────┘
```

Implementation:
- A `<div id="opt-params-card">` with `display:none` default
- `loadReport()` checks `d.optimize_params`; if present, renders the card and shows it
- Each param rendered as a two-column row: name (left, muted) / value (right, highlighted)
- Score and objective shown in the card header

## Scope

- **Modified:** `src/backtest/web/routes.py` — `get_report` function only
- **Modified:** `src/backtest/web/static/index.html` — add CSS + HTML block + JS render logic

## No breaking changes

- Reports without `optimize_result_id` (standalone `run` command): LEFT JOIN returns NULL, card stays hidden
- Existing API consumers: new fields are additive only
