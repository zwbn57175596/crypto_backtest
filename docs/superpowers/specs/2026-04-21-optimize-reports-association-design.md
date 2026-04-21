# Design: Link optimize_results → reports

**Date:** 2026-04-21
**Status:** Approved

## Problem

`save_top_reports` re-runs the top N optimization trials and writes full reports (with equity_curve + trades) to the `reports` table. However, no foreign key is recorded, so there's no way to know which `optimize_results` row a given `reports` row came from.

## Solution

### Database

Add a nullable column to `reports`:

```sql
ALTER TABLE reports ADD COLUMN optimize_result_id INTEGER;
```

Nullable so that reports created via `python -m backtest run` (standalone backtests) are unaffected.

### Code change (`optimizer.py` — `save_top_reports` only)

Before inserting each report, look up the matching `optimize_results.id` using `params_json` + `strategy` + `symbol` + `interval` as the key:

```sql
SELECT id FROM optimize_results
WHERE params_json = ? AND strategy = ? AND symbol = ? AND interval = ?
ORDER BY created_at DESC LIMIT 1
```

Include the returned `id` in the `INSERT INTO reports` statement.

### Schema migration

On startup of `save_top_reports`, run `ALTER TABLE reports ADD COLUMN optimize_result_id INTEGER` inside a `try/except` (SQLite raises `OperationalError` if column already exists — safe to ignore).

## Scope

- **Changed:** `optimizer.py` — `save_top_reports` function only (~10 lines)
- **Unchanged:** `__main__.py`, `reporter.py`, web layer, `cmd_run`

## No breaking changes

- Existing `reports` rows get `optimize_result_id = NULL`
- Standalone `run` command continues to write `optimize_result_id = NULL`
