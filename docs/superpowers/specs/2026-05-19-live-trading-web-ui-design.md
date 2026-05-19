# Live Trading Web UI — Design Spec

**Date:** 2026-05-19  
**Branch:** feat/live-trading-adapter  
**Status:** Approved

---

## Overview

Add a live trading monitoring page to the existing FastAPI web server. The page shows real-time positions, order history, trade history (with P&L), and strategy run logs. All pages are protected by optional password authentication via a single environment variable.

---

## 1. Authentication

### Mechanism
- FastAPI `AuthMiddleware` registered in `app.py`, applied globally to all routes.
- Controlled by environment variable `WEB_PASSWORD`:
  - **Not set** → auth disabled, all requests pass through.
  - **Set** → all requests must carry a valid session cookie.

### Session Cookie
- Cookie name: `session`
- Cookie value: `sha256(WEB_PASSWORD)` hex digest (compared server-side; no plaintext stored).
- Flags: `HttpOnly`, `Path=/`, `SameSite=Lax`.

### Login Flow
- Unauthed page requests → redirect to `/login`.
- Unauthed API requests → `401 {"detail": "Unauthorized"}`.
- `GET /login` → serves an inline minimal HTML form (no new static file needed; rendered from a string in `routes.py` or a new `auth_routes.py`).
- `POST /login` (form body `password=...`) → verify against `WEB_PASSWORD`, set cookie, redirect to `/live`. Wrong password → re-render login with error message.
- No explicit logout endpoint required (clear cookie in browser to log out).

### Implementation
- `AuthMiddleware` in `src/backtest/web/middleware.py`.
- Skips auth for `/login` and `/static/*` paths.
- Registered via `app.add_middleware(AuthMiddleware)` in `create_app()`.

---

## 2. Live Trading Page (`/live`, `live.html`)

### Navigation
All three pages share a top navbar:
```
[回测报告 /]   [优化结果 /optimize]   [实盘监控 /live]
```
Add this navbar to `index.html`, `optimize.html`, and the new `live.html`.

### Filter Bar (fixed at top)
- **Account** dropdown — populated from `GET /api/live/accounts`
- **Symbol** dropdown — populated from `GET /api/live/symbols?account_id=<id>`, updates when account changes
- All tab content reloads when either filter changes.

### Tab: 概览 (Overview)

- **Position card**: direction (long/short/flat), quantity (USDT), entry price, unrealized PnL, margin.
- **Balance / Equity** numeric display (latest row from `positions` table).
- **Equity curve chart**: ECharts line chart using `equity` column from `positions` table, x-axis = timestamp, y-axis = USDT. Same visual style as report page equity curve.

### Tab: 订单 (Orders)

Paginated table (50 rows/page), sorted by `ts DESC`.

| Column | Source |
|--------|--------|
| 时间 | `ts` (formatted) |
| 方向 | `side` |
| 类型 | `type` |
| 数量 (USDT) | `quantity` |
| 委托价 | `price` (—if market) |
| 成交价 | `filled_price` |
| 状态 | `status` (badge coloring: filled=green, pending=yellow, canceled=grey) |
| 手续费 | `commission` |

### Tab: 成交 (Trades)

Paginated table (50 rows/page), sorted by `ts DESC`. Summary row at bottom of each page.

| Column | Source |
|--------|--------|
| 时间 | `ts` (formatted) |
| 方向 | `side` |
| 成交价 | `price` |
| 数量 (USDT) | `notional` |
| 已实现PnL | `realized_pnl` (colored: positive=green, negative=red) |
| 手续费 | `commission` |

**Summary row** (below table): 累计 PnL = sum of `realized_pnl` for current filter, 总手续费 = sum of `commission`.

### Tab: 日志 (Logs)

- **Log file selector** dropdown at top: scans `--log-dir` directory for `*.log` files, sorted by mtime descending (most recent first). Shows filename + timestamp.
- **Content area**: fixed-height scrollable `<pre>` box, shows last 500 lines of selected file in **reverse chronological order** (newest line at top).
- **Keyword filter** input: client-side, hides non-matching lines instantly without re-fetching.
- **Refresh button**: re-fetches the selected file.

---

## 3. Log File Generation

### Changes to `live_engine.py`

- New constructor parameter: `log_file: str | None = None`
- New CLI parameter: `--log-file PATH` (e.g. `data/logs/live_BTCUSDT_20260519T143000.log`)
- If `log_file` is provided:
  - Configure Python `logging.FileHandler` + `StreamHandler` (stdout).
  - Log format: `%(asctime)s [%(levelname)s] %(message)s` with second precision.
  - Replace all `print()` calls in `live_engine.py` with `self._log.info()` / `self._log.warning()` / `self._log.error()`.

### Log File Naming Convention (recommended, not enforced by engine)
```
data/logs/live_<SYMBOL>_<YYYYMMDDTHHMMSS>.log
```

---

## 4. New API Routes

All live routes go in `src/backtest/web/live_routes.py`, registered as a separate APIRouter with prefix `/api/live`. The router receives `live_db_path` and `log_dir` via `app.state`.

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/live/accounts` | List distinct `(account_id, exchange)` pairs from all tables |
| GET | `/api/live/symbols` | `?account_id=` → distinct symbols for that account |
| GET | `/api/live/overview` | `?account_id=&symbol=` → latest position + balance/equity + equity curve array |
| GET | `/api/live/orders` | `?account_id=&symbol=&page=1` → paginated orders, desc |
| GET | `/api/live/trades` | `?account_id=&symbol=&page=1` → paginated trades + summary totals |
| GET | `/api/live/logs` | List log files from `log_dir` (name + mtime), newest first |
| GET | `/api/live/logs/{filename}` | `?tail=500` → last N lines of file, reversed (newest first) |

### Security note for log file serving
`filename` parameter must be validated: strip path separators, only allow `*.log` files within `log_dir`. Reject any `..` or absolute paths.

---

## 5. File Changes Summary

| File | Change |
|------|--------|
| `src/backtest/web/app.py` | Register `AuthMiddleware`; add `live_db_path` + `log_dir` to `app.state`; include `live_routes` router |
| `src/backtest/web/middleware.py` | New file — `AuthMiddleware` |
| `src/backtest/web/live_routes.py` | New file — all `/api/live/*` routes + `/live` page route + `/login` routes |
| `src/backtest/web/static/live.html` | New file — single-page tabs UI |
| `src/backtest/web/static/index.html` | Add navbar |
| `src/backtest/web/static/optimize.html` | Add navbar |
| `src/backtest/live_engine.py` | Add `log_file` param; replace `print()` with `logging` |
| `src/backtest/__main__.py` | Add `--log-file` to `live` subcommand; add `--live-db` + `--log-dir` to `web` subcommand |

---

## 6. Non-Goals

- No real-time WebSocket push (manual refresh only).
- No trade editing or order cancellation from the UI.
- No user management (single shared password).
- No log rotation (engine just appends; rotation left to OS/systemd).
