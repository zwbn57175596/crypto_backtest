# Live Trading Web UI — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a live trading monitoring page (`/live`) to the FastAPI web server, with password auth, tabs for positions/orders/trades/logs, and file-based logging in the live engine.

**Architecture:** FastAPI `AuthMiddleware` in a new `middleware.py` protects all routes globally; a new `live_routes.py` handles login + all `/api/live/*` routes + the `/live` page; `live_engine.py` gains a `--log-file` option that writes structured log output via Python's `logging` module; `live.html` is a single-page tab UI (ECharts + vanilla JS) matching the existing dark theme.

**Tech Stack:** FastAPI, Starlette `BaseHTTPMiddleware`, Python `logging`, ECharts 5, SQLite, vanilla JS.

---

## File Map

| Action | File |
|--------|------|
| Create | `src/backtest/web/middleware.py` |
| Create | `src/backtest/web/live_routes.py` |
| Create | `src/backtest/web/static/live.html` |
| Modify | `src/backtest/web/app.py` |
| Modify | `src/backtest/live_engine.py` |
| Modify | `src/backtest/__main__.py` |
| Modify | `src/backtest/web/static/index.html` |
| Modify | `src/backtest/web/static/optimize.html` |
| Create | `tests/test_web_auth.py` |
| Create | `tests/test_live_routes.py` |
| Modify | `tests/test_live_engine.py` |

---

## Task 1: Auth Middleware

**Files:**
- Create: `src/backtest/web/middleware.py`
- Create: `src/backtest/web/live_routes.py` (login routes only — expanded in Task 2)
- Modify: `src/backtest/web/app.py`
- Test: `tests/test_web_auth.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_web_auth.py`:

```python
import hashlib
import sqlite3

import pytest
from fastapi.testclient import TestClient

from backtest.web.app import create_app


@pytest.fixture
def bare_db(tmp_path):
    db_path = str(tmp_path / "reports.db")
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE reports (id INTEGER PRIMARY KEY, strategy TEXT, symbol TEXT,"
        " interval TEXT, created_at TEXT, report_json TEXT, optimize_result_id INTEGER)"
    )
    conn.execute(
        "CREATE TABLE optimize_results (id INTEGER PRIMARY KEY, strategy TEXT, symbol TEXT,"
        " interval TEXT, start_date TEXT, end_date TEXT, objective TEXT, score REAL,"
        " params_json TEXT, report_json TEXT, created_at TEXT, batch_id TEXT)"
    )
    conn.commit()
    conn.close()
    return db_path


def test_no_password_env_passes_all_requests(bare_db, monkeypatch):
    monkeypatch.delenv("WEB_PASSWORD", raising=False)
    app = create_app(bare_db)
    client = TestClient(app)
    assert client.get("/api/reports").status_code == 200


def test_password_set_api_without_cookie_returns_401(bare_db, monkeypatch):
    monkeypatch.setenv("WEB_PASSWORD", "secret")
    app = create_app(bare_db)
    client = TestClient(app, follow_redirects=False)
    resp = client.get("/api/reports")
    assert resp.status_code == 401
    assert resp.json()["detail"] == "Unauthorized"


def test_password_set_page_without_cookie_redirects_to_login(bare_db, monkeypatch):
    monkeypatch.setenv("WEB_PASSWORD", "secret")
    app = create_app(bare_db)
    client = TestClient(app, follow_redirects=False)
    resp = client.get("/")
    assert resp.status_code == 302
    assert resp.headers["location"] == "/login"


def test_static_files_skip_auth(bare_db, monkeypatch):
    monkeypatch.setenv("WEB_PASSWORD", "secret")
    app = create_app(bare_db)
    client = TestClient(app, follow_redirects=False)
    resp = client.get("/static/nonexistent.css")
    assert resp.status_code not in (401, 302)


def test_valid_cookie_passes(bare_db, monkeypatch):
    monkeypatch.setenv("WEB_PASSWORD", "secret")
    session = hashlib.sha256(b"secret").hexdigest()
    app = create_app(bare_db)
    client = TestClient(app, follow_redirects=False, cookies={"session": session})
    assert client.get("/api/reports").status_code == 200


def test_login_correct_password_sets_cookie_and_redirects(bare_db, monkeypatch):
    monkeypatch.setenv("WEB_PASSWORD", "secret")
    app = create_app(bare_db)
    client = TestClient(app, follow_redirects=False)
    resp = client.post("/login", data={"password": "secret"})
    assert resp.status_code == 302
    expected = hashlib.sha256(b"secret").hexdigest()
    assert resp.cookies.get("session") == expected


def test_login_wrong_password_returns_401(bare_db, monkeypatch):
    monkeypatch.setenv("WEB_PASSWORD", "secret")
    app = create_app(bare_db)
    client = TestClient(app, follow_redirects=False)
    resp = client.post("/login", data={"password": "wrong"})
    assert resp.status_code == 401
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_web_auth.py -v
```
Expected: all FAIL (modules don't exist yet).

- [ ] **Step 3: Create `src/backtest/web/middleware.py`**

```python
import hashlib
import os

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse, Response


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        password = os.environ.get("WEB_PASSWORD")
        if not password:
            return await call_next(request)

        path = request.url.path
        if path == "/login" or path.startswith("/static/"):
            return await call_next(request)

        expected = hashlib.sha256(password.encode()).hexdigest()
        if request.cookies.get("session") == expected:
            return await call_next(request)

        if "text/html" in request.headers.get("accept", ""):
            return RedirectResponse(url="/login", status_code=302)
        return JSONResponse({"detail": "Unauthorized"}, status_code=401)
```

- [ ] **Step 4: Create minimal `src/backtest/web/live_routes.py`** (login only — full routes added in Task 2)

```python
import hashlib
import os
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, RedirectResponse
from starlette.requests import Request

router = APIRouter()

_LOGIN_HTML = """\
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>登录</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #0f1117; color: #e0e0e0;
         display: flex; justify-content: center; align-items: center; height: 100vh; }}
  .card {{ background: #1e293b; border-radius: 12px; padding: 32px; width: 320px; }}
  h2 {{ font-size: 18px; color: #00d4aa; margin-bottom: 24px; text-align: center; }}
  input {{ width: 100%; padding: 10px 12px; background: #0f1117; color: #e0e0e0;
          border: 1px solid #333; border-radius: 6px; font-size: 14px; margin-bottom: 16px; }}
  button {{ width: 100%; padding: 10px; background: #00d4aa; color: #0f1117;
           border: none; border-radius: 6px; font-size: 14px; cursor: pointer; font-weight: 600; }}
  .error {{ color: #ef4444; font-size: 13px; margin-bottom: 12px; text-align: center; }}
</style>
</head>
<body>
<div class="card">
  <h2>实盘监控</h2>
  {error}
  <form method="POST" action="/login">
    <input type="password" name="password" placeholder="密码" autofocus>
    <button type="submit">登录</button>
  </form>
</div>
</body>
</html>"""


@router.get("/login", response_class=HTMLResponse)
def login_page():
    return HTMLResponse(_LOGIN_HTML.format(error=""))


@router.post("/login")
async def do_login(request: Request):
    form = await request.form()
    password = str(form.get("password", ""))
    expected_password = os.environ.get("WEB_PASSWORD", "")
    if not expected_password or password == expected_password:
        resp = RedirectResponse(url="/live", status_code=302)
        if expected_password:
            session_val = hashlib.sha256(expected_password.encode()).hexdigest()
            resp.set_cookie("session", session_val, httponly=True, samesite="lax")
        return resp
    return HTMLResponse(
        _LOGIN_HTML.format(error='<p class="error">密码错误</p>'),
        status_code=401,
    )
```

- [ ] **Step 5: Update `src/backtest/web/app.py`**

Replace the current content with:

```python
import sqlite3

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pathlib import Path


def _migrate_db(db_path: str) -> None:
    """Ensure optimize_results has batch_id and leverage columns; backfill old rows."""
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("ALTER TABLE optimize_results ADD COLUMN batch_id TEXT")
        conn.commit()
    except sqlite3.OperationalError:
        pass

    try:
        conn.execute("ALTER TABLE optimize_results ADD COLUMN leverage REAL")
        conn.commit()
    except sqlite3.OperationalError:
        pass

    try:
        conn.execute("""
            UPDATE optimize_results
            SET batch_id = strftime('%Y%m%dT%H%M%S', created_at)
                           || '_' || strategy || '_' || symbol
            WHERE batch_id IS NULL
        """)
        conn.commit()
    except sqlite3.OperationalError:
        pass
    conn.close()


def create_app(
    db_path: str,
    live_db_path: str | None = None,
    log_dir: str | None = None,
) -> FastAPI:
    app = FastAPI(title="Crypto Backtest")
    app.state.db_path = db_path
    app.state.live_db_path = live_db_path or ""
    app.state.log_dir = log_dir or ""
    _migrate_db(db_path)

    from backtest.web.middleware import AuthMiddleware
    from backtest.web.routes import router
    from backtest.web.live_routes import router as live_router

    app.add_middleware(AuthMiddleware)
    app.include_router(router)
    app.include_router(live_router)

    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    return app
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
pytest tests/test_web_auth.py -v
```
Expected: all PASS.

- [ ] **Step 7: Run full test suite to check no regressions**

```bash
pytest tests/test_web.py tests/test_web_auth.py -v
```
Expected: all PASS.

- [ ] **Step 8: Commit**

```bash
git add src/backtest/web/middleware.py src/backtest/web/live_routes.py \
        src/backtest/web/app.py tests/test_web_auth.py
git commit -m "feat: add global auth middleware with WEB_PASSWORD env var"
```

---

## Task 2: Live API Routes + CLI Web Extensions

**Files:**
- Modify: `src/backtest/web/live_routes.py` (add all `/api/live/*` routes + `/live` page)
- Modify: `src/backtest/__main__.py` (add `--live-db` + `--log-dir` to `web` command)
- Test: `tests/test_live_routes.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_live_routes.py`:

```python
import sqlite3
import pytest
from fastapi.testclient import TestClient
from backtest.web.app import create_app


def _make_reports_db(tmp_path):
    db_path = str(tmp_path / "reports.db")
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE reports (id INTEGER PRIMARY KEY, strategy TEXT, symbol TEXT,"
        " interval TEXT, created_at TEXT, report_json TEXT, optimize_result_id INTEGER)"
    )
    conn.execute(
        "CREATE TABLE optimize_results (id INTEGER PRIMARY KEY, strategy TEXT, symbol TEXT,"
        " interval TEXT, start_date TEXT, end_date TEXT, objective TEXT, score REAL,"
        " params_json TEXT, report_json TEXT, created_at TEXT, batch_id TEXT)"
    )
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def live_db(tmp_path):
    db_path = str(tmp_path / "live_history.db")
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            account_id TEXT NOT NULL, exchange TEXT NOT NULL, symbol TEXT NOT NULL,
            side TEXT, quantity REAL NOT NULL, entry_price REAL, leverage INTEGER,
            unrealized_pnl REAL, margin REAL, balance REAL NOT NULL,
            equity REAL NOT NULL, ts INTEGER NOT NULL
        );
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            account_id TEXT NOT NULL, exchange TEXT NOT NULL, symbol TEXT NOT NULL,
            order_id TEXT NOT NULL, client_order_id TEXT, side TEXT NOT NULL,
            type TEXT NOT NULL, quantity REAL NOT NULL, price REAL,
            status TEXT NOT NULL, filled_price REAL, filled_qty REAL,
            commission REAL, ts INTEGER NOT NULL, filled_at INTEGER,
            UNIQUE (exchange, account_id, order_id)
        );
        CREATE TABLE trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            account_id TEXT NOT NULL, exchange TEXT NOT NULL, symbol TEXT NOT NULL,
            trade_id TEXT NOT NULL, order_id TEXT NOT NULL, side TEXT NOT NULL,
            price REAL NOT NULL, qty REAL NOT NULL, notional REAL NOT NULL,
            commission REAL NOT NULL, realized_pnl REAL, ts INTEGER NOT NULL,
            UNIQUE (exchange, account_id, trade_id)
        );
        CREATE TABLE state_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            account_id TEXT NOT NULL, exchange TEXT NOT NULL, strategy TEXT NOT NULL,
            symbol TEXT NOT NULL, interval TEXT NOT NULL,
            state_json TEXT NOT NULL, ts INTEGER NOT NULL
        );
    """)
    conn.execute(
        "INSERT INTO positions (account_id, exchange, symbol, side, quantity, entry_price,"
        " leverage, unrealized_pnl, margin, balance, equity, ts) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        ("acc1", "binance", "BTCUSDT", "long", 1000.0, 50000.0, 10, 50.0, 100.0, 900.0, 1050.0, 1700000000000),
    )
    conn.execute(
        "INSERT INTO orders (account_id, exchange, symbol, order_id, side, type, quantity,"
        " status, filled_price, filled_qty, commission, ts) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        ("acc1", "binance", "BTCUSDT", "ord1", "buy", "market", 1000.0, "filled", 50000.0, 1000.0, 0.4, 1700000000000),
    )
    conn.execute(
        "INSERT INTO trades (account_id, exchange, symbol, trade_id, order_id, side,"
        " price, qty, notional, commission, realized_pnl, ts) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        ("acc1", "binance", "BTCUSDT", "trd1", "ord1", "buy", 50000.0, 0.02, 1000.0, 0.4, 0.0, 1700000000000),
    )
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def client(tmp_path, live_db):
    app = create_app(
        _make_reports_db(tmp_path),
        live_db_path=live_db,
        log_dir=str(tmp_path / "logs"),
    )
    return TestClient(app)


def test_list_accounts(client):
    resp = client.get("/api/live/accounts")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["account_id"] == "acc1"
    assert data[0]["exchange"] == "binance"


def test_list_symbols(client):
    resp = client.get("/api/live/symbols", params={"account_id": "acc1"})
    assert resp.status_code == 200
    assert resp.json() == ["BTCUSDT"]


def test_overview_returns_position_and_equity_curve(client):
    resp = client.get("/api/live/overview", params={"account_id": "acc1", "symbol": "BTCUSDT"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["balance"] == pytest.approx(900.0)
    assert data["equity"] == pytest.approx(1050.0)
    assert data["position"]["side"] == "long"
    assert len(data["equity_curve"]) == 1
    assert data["equity_curve"][0] == [1700000000000, pytest.approx(1050.0)]


def test_overview_no_position_returns_flat(tmp_path, live_db):
    db = sqlite3.connect(live_db)
    db.execute(
        "INSERT INTO positions (account_id, exchange, symbol, side, quantity, balance, equity, ts)"
        " VALUES (?,?,?,?,?,?,?,?)",
        ("acc1", "binance", "BTCUSDT", None, 0.0, 950.0, 950.0, 1700000001000),
    )
    db.commit()
    db.close()
    app = create_app(_make_reports_db(tmp_path), live_db_path=live_db)
    data = TestClient(app).get("/api/live/overview", params={"account_id": "acc1", "symbol": "BTCUSDT"}).json()
    assert data["position"] is None
    assert data["balance"] == pytest.approx(950.0)


def test_list_orders_pagination(client):
    resp = client.get("/api/live/orders", params={"account_id": "acc1", "symbol": "BTCUSDT"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1
    assert data["page"] == 1
    assert len(data["items"]) == 1
    assert data["items"][0]["order_id"] == "ord1"
    assert data["items"][0]["status"] == "filled"


def test_list_trades_with_summary(client):
    resp = client.get("/api/live/trades", params={"account_id": "acc1", "symbol": "BTCUSDT"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1
    assert len(data["items"]) == 1
    assert data["summary"]["total_pnl"] == pytest.approx(0.0)
    assert data["summary"]["total_commission"] == pytest.approx(0.4)


def test_list_logs_empty_when_dir_missing(client):
    resp = client.get("/api/live/logs")
    assert resp.status_code == 200
    assert resp.json() == []


def test_get_log_file_reversed(tmp_path, live_db):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "test.log").write_text("line1\nline2\nline3\n")
    app = create_app(_make_reports_db(tmp_path), live_db_path=live_db, log_dir=str(log_dir))
    resp = TestClient(app).get("/api/live/logs/test.log")
    assert resp.status_code == 200
    assert resp.json()["lines"] == ["line3", "line2", "line1"]


def test_get_log_file_invalid_name_rejected(tmp_path, live_db):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    app = create_app(_make_reports_db(tmp_path), live_db_path=live_db, log_dir=str(log_dir))
    resp = TestClient(app).get("/api/live/logs/notalog.txt")
    assert resp.status_code == 400


def test_list_logs_returns_files(tmp_path, live_db):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "run1.log").write_text("hello")
    (log_dir / "run2.log").write_text("world")
    app = create_app(_make_reports_db(tmp_path), live_db_path=live_db, log_dir=str(log_dir))
    resp = TestClient(app).get("/api/live/logs")
    assert resp.status_code == 200
    names = [f["filename"] for f in resp.json()]
    assert set(names) == {"run1.log", "run2.log"}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_live_routes.py -v
```
Expected: all FAIL (routes not implemented yet).

- [ ] **Step 3: Expand `src/backtest/web/live_routes.py` with all API routes**

Append to the existing `live_routes.py` (after the login routes):

```python
import re
import sqlite3
from pathlib import Path

from fastapi import HTTPException, Query
from starlette.requests import Request

PAGE_SIZE = 50


def _live_db(request: Request) -> str:
    return request.app.state.live_db_path


def _log_dir(request: Request) -> str:
    return request.app.state.log_dir


def _open_live_db(request: Request):
    path = _live_db(request)
    if not path or not Path(path).exists():
        return None
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


# ── Page route ──────────────────────────────────────────────────────────────

@router.get("/live", response_class=HTMLResponse)
def live_page():
    html_path = Path(__file__).parent / "static" / "live.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="live.html not found")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


# ── API routes ───────────────────────────────────────────────────────────────

@router.get("/api/live/accounts")
def list_accounts(request: Request):
    conn = _open_live_db(request)
    if conn is None:
        return []
    rows = conn.execute("""
        SELECT DISTINCT account_id, exchange FROM (
            SELECT account_id, exchange FROM positions
            UNION SELECT account_id, exchange FROM orders
            UNION SELECT account_id, exchange FROM trades
        ) ORDER BY account_id
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@router.get("/api/live/symbols")
def list_symbols(request: Request, account_id: str = Query(...)):
    conn = _open_live_db(request)
    if conn is None:
        return []
    rows = conn.execute("""
        SELECT DISTINCT symbol FROM (
            SELECT symbol FROM positions WHERE account_id = ?
            UNION SELECT symbol FROM orders WHERE account_id = ?
            UNION SELECT symbol FROM trades WHERE account_id = ?
        ) ORDER BY symbol
    """, (account_id, account_id, account_id)).fetchall()
    conn.close()
    return [r[0] for r in rows]


@router.get("/api/live/overview")
def get_overview(request: Request, account_id: str = Query(...), symbol: str = Query(...)):
    conn = _open_live_db(request)
    if conn is None:
        return {"position": None, "balance": 0.0, "equity": 0.0, "equity_curve": []}
    latest = conn.execute(
        "SELECT * FROM positions WHERE account_id=? AND symbol=? ORDER BY ts DESC LIMIT 1",
        (account_id, symbol),
    ).fetchone()
    curve_rows = conn.execute(
        "SELECT ts, equity FROM positions WHERE account_id=? AND symbol=? ORDER BY ts ASC",
        (account_id, symbol),
    ).fetchall()
    conn.close()

    position = None
    balance = 0.0
    equity = 0.0
    if latest:
        balance = latest["balance"]
        equity = latest["equity"]
        if latest["side"]:
            position = {
                "side": latest["side"],
                "quantity": latest["quantity"],
                "entry_price": latest["entry_price"],
                "leverage": latest["leverage"],
                "unrealized_pnl": latest["unrealized_pnl"],
                "margin": latest["margin"],
            }
    return {
        "position": position,
        "balance": balance,
        "equity": equity,
        "equity_curve": [[r["ts"], r["equity"]] for r in curve_rows],
    }


@router.get("/api/live/orders")
def list_orders(
    request: Request,
    account_id: str = Query(...),
    symbol: str = Query(...),
    page: int = Query(1, ge=1),
):
    conn = _open_live_db(request)
    if conn is None:
        return {"items": [], "total": 0, "page": page, "page_size": PAGE_SIZE}
    offset = (page - 1) * PAGE_SIZE
    total = conn.execute(
        "SELECT COUNT(*) FROM orders WHERE account_id=? AND symbol=?",
        (account_id, symbol),
    ).fetchone()[0]
    rows = conn.execute(
        "SELECT * FROM orders WHERE account_id=? AND symbol=? ORDER BY ts DESC LIMIT ? OFFSET ?",
        (account_id, symbol, PAGE_SIZE, offset),
    ).fetchall()
    conn.close()
    return {"items": [dict(r) for r in rows], "total": total, "page": page, "page_size": PAGE_SIZE}


@router.get("/api/live/trades")
def list_trades(
    request: Request,
    account_id: str = Query(...),
    symbol: str = Query(...),
    page: int = Query(1, ge=1),
):
    conn = _open_live_db(request)
    if conn is None:
        return {
            "items": [], "total": 0, "page": page, "page_size": PAGE_SIZE,
            "summary": {"total_pnl": 0.0, "total_commission": 0.0},
        }
    offset = (page - 1) * PAGE_SIZE
    total = conn.execute(
        "SELECT COUNT(*) FROM trades WHERE account_id=? AND symbol=?",
        (account_id, symbol),
    ).fetchone()[0]
    rows = conn.execute(
        "SELECT * FROM trades WHERE account_id=? AND symbol=? ORDER BY ts DESC LIMIT ? OFFSET ?",
        (account_id, symbol, PAGE_SIZE, offset),
    ).fetchall()
    summary = conn.execute(
        "SELECT COALESCE(SUM(realized_pnl),0) AS total_pnl,"
        "       COALESCE(SUM(commission),0)   AS total_commission"
        " FROM trades WHERE account_id=? AND symbol=?",
        (account_id, symbol),
    ).fetchone()
    conn.close()
    return {
        "items": [dict(r) for r in rows],
        "total": total,
        "page": page,
        "page_size": PAGE_SIZE,
        "summary": {"total_pnl": summary["total_pnl"], "total_commission": summary["total_commission"]},
    }


@router.get("/api/live/logs")
def list_log_files(request: Request):
    log_path = Path(_log_dir(request))
    if not log_path.exists():
        return []
    files = sorted(log_path.glob("*.log"), key=lambda f: f.stat().st_mtime, reverse=True)
    return [
        {"filename": f.name, "mtime": int(f.stat().st_mtime * 1000), "size": f.stat().st_size}
        for f in files
    ]


@router.get("/api/live/logs/{filename}")
def get_log_file(filename: str, request: Request, tail: int = Query(500, ge=1, le=10000)):
    if not re.fullmatch(r"[a-zA-Z0-9_\-\.]+\.log", filename):
        raise HTTPException(status_code=400, detail="Invalid filename")
    log_dir_path = Path(_log_dir(request))
    if not log_dir_path.exists():
        raise HTTPException(status_code=404, detail="Log directory not configured")
    try:
        filepath = (log_dir_path / filename).resolve()
        filepath.relative_to(log_dir_path.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid filename")
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Log file not found")
    lines = filepath.read_text(encoding="utf-8", errors="replace").splitlines()
    return {"lines": list(reversed(lines[-tail:]))}
```

Note: The imports (`re`, `sqlite3`, `Path`, `HTTPException`, `Query`) need to be at the top of `live_routes.py`. The full file top section should be:

```python
import hashlib
import os
import re
import sqlite3
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse, RedirectResponse
from starlette.requests import Request

router = APIRouter()
```

- [ ] **Step 4: Update `src/backtest/__main__.py` — add `--live-db` and `--log-dir` to the `web` subcommand**

Find the `p_web` argument parser block (around line 518) and add two arguments:

```python
    p_web = sub.add_parser("web", help="Start web report viewer")
    p_web.add_argument("--port", type=int, default=8000)
    p_web.add_argument("--db", default=None)
    p_web.add_argument("--live-db", default=None, dest="live_db",
                       help="SQLite path for live history (default: data/live_history.db)")
    p_web.add_argument("--log-dir", default=None, dest="log_dir",
                       help="Directory containing live engine log files (default: data/logs)")
```

Find `cmd_web` (around line 237) and update it:

```python
def cmd_web(args: argparse.Namespace) -> None:
    import uvicorn
    from backtest.web.app import create_app
    db_path = args.db or str(Path("data") / "reports.db")
    live_db_path = args.live_db or str(Path("data") / "live_history.db")
    log_dir = args.log_dir or str(Path("data") / "logs")
    app = create_app(db_path, live_db_path=live_db_path, log_dir=log_dir)
    print(f"Starting web server at http://localhost:{args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_live_routes.py -v
```
Expected: all PASS.

- [ ] **Step 6: Run full web test suite for regressions**

```bash
pytest tests/test_web.py tests/test_web_auth.py tests/test_live_routes.py -v
```
Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add src/backtest/web/live_routes.py src/backtest/__main__.py
git commit -m "feat: add /api/live/* routes and --live-db/--log-dir web CLI params"
```

Then add and commit the test:
```bash
git add tests/test_live_routes.py
git commit -m "test: add tests for live API routes"
```

---

## Task 3: live_engine File Logging + CLI Live Extension

**Files:**
- Modify: `src/backtest/live_engine.py`
- Modify: `src/backtest/__main__.py` (add `--log-file` to `live` subcommand and call `logging.basicConfig` in `cmd_live`)
- Modify: `tests/test_live_engine.py`

- [ ] **Step 1: Write failing tests — add to `tests/test_live_engine.py`**

At the top of the file, add `import logging` and `from pathlib import Path` to existing imports.

Add this new test class at the bottom of `tests/test_live_engine.py`:

```python
class TestLogging:
    def test_setup_logger_no_file_has_no_file_handler(self):
        from backtest.live_engine import LiveEngine
        log = LiveEngine._setup_logger(None)
        handler_types = {type(h).__name__ for h in log.handlers}
        assert "FileHandler" not in handler_types

    def test_setup_logger_with_file_adds_file_handler(self, tmp_path):
        from backtest.live_engine import LiveEngine
        log_path = str(tmp_path / "engine.log")
        log = LiveEngine._setup_logger(log_path)
        handler_types = {type(h).__name__ for h in log.handlers}
        assert "FileHandler" in handler_types

    def test_log_file_written(self, tmp_path):
        import tempfile
        from backtest.live_engine import LiveEngine
        log_path = str(tmp_path / "engine.log")
        with tempfile.TemporaryDirectory() as state_tmp:
            engine = LiveEngine(
                strategy_class=_NoOpStrategy,
                symbol="BTCUSDT",
                interval="1h",
                leverage=10,
                connector=_make_connector(),
                history_db=_make_history_db(),
                account_id="testacc",
                dry_run=True,
                state_db=f"{state_tmp}/state.db",
                log_file=log_path,
            )
        engine._log.info("test-marker")
        for h in engine._log.handlers:
            h.flush()
        content = Path(log_path).read_text()
        assert "test-marker" in content
```

Also update the existing test that uses `capsys` to use `caplog`:

Find and replace:
```python
    def test_process_bar_prints_status(self, capsys):
        engine, ex, strategy = self._make_engine()
        engine._process_bar(_make_bar(), strategy, ex)
        captured = capsys.readouterr()
        assert "balance=" in captured.out
```

Replace with:
```python
    def test_process_bar_logs_status(self, caplog):
        import logging
        engine, ex, strategy = self._make_engine()
        with caplog.at_level(logging.INFO, logger="backtest.live_engine"):
            engine._process_bar(_make_bar(), strategy, ex)
        assert any("balance=" in r.message for r in caplog.records)
```

- [ ] **Step 2: Run tests to verify new tests fail**

```bash
pytest tests/test_live_engine.py::TestLogging -v
```
Expected: FAIL (`_setup_logger` not a static method yet / `log_file` param missing).

- [ ] **Step 3: Update `src/backtest/live_engine.py`**

Add `import logging` to imports at the top.

Add `log_file: str | None = None` as the last parameter to `LiveEngine.__init__`:

```python
    def __init__(
        self,
        strategy_class: type[BaseStrategy],
        symbol: str,
        interval: str,
        leverage: int,
        connector: BaseExchangeConnector,
        history_db: LiveHistoryDB,
        account_id: str,
        commission_rate: float = 0.0004,
        dry_run: bool = False,
        state_db: str = "data/live_state.db",
        sync_interval: int = 300,
        max_order_notional: float | None = None,
        max_daily_notional: float | None = None,
        max_session_loss: float | None = None,
        order_rate: tuple[int, int] | None = None,
        log_file: str | None = None,
    ):
```

At the end of `__init__`, after `self._order_rate = order_rate`, add:

```python
        self._log = self._setup_logger(log_file)
```

Add `_setup_logger` as a static method (place it just before `run`):

```python
    @staticmethod
    def _setup_logger(log_file: str | None) -> logging.Logger:
        logger = logging.getLogger("backtest.live_engine")
        logger.setLevel(logging.DEBUG)
        logger.propagate = True
        logger.handlers.clear()
        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            fmt = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
            )
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(fmt)
            logger.addHandler(fh)
        return logger
```

Replace all `print(...)` and `print(..., file=sys.stderr)` calls in `LiveEngine` methods with `self._log` calls. Here are the exact replacements:

In `run()`:
```python
# Before:
print(f"[WARN] change_leverage: {e}", file=sys.stderr)
# After:
self._log.warning(f"change_leverage: {e}")
```

In `_process_bar()`:
```python
# Before:
print(f"[{ts}] balance={live_exchange.balance:.2f} equity={live_exchange.equity:.2f} pos={pos_str}")
# After:
self._log.info(f"balance={live_exchange.balance:.2f} equity={live_exchange.equity:.2f} pos={pos_str}")
```

```python
# Before:
self._alert(f"[ERROR] bar {ts}: {e}")
# After:
self._log.error(f"bar {ts}: {e}")
```

In `_sync_loop()`:
```python
# Before:
print(f"[WARN] background sync error: {e}", file=sys.stderr)
# After:
self._log.warning(f"background sync error: {e}")
```

In `_alert()`:
```python
# Before:
def _alert(self, msg: str) -> None:
    print(msg, file=sys.stderr)
# After:
def _alert(self, msg: str) -> None:
    self._log.error(msg)
```

In `_print_startup_summary()` — replace each `print(...)` with `self._log.info(...)`:
```python
def _print_startup_summary(self, live_exchange: LiveExchange) -> None:
    dry = " [DRY-RUN]" if self.dry_run else ""
    self._log.info(f"=== LiveEngine{dry} ===")
    self._log.info(
        f"Strategy: {self.strategy_class.__name__}  Symbol: {self.symbol}  "
        f"Interval: {self.interval}  Leverage: {self.leverage}x"
    )
    self._log.info(
        f"Account ID: {self._account_id}  Exchange: {self._connector.exchange_name}"
    )
    self._log.info(
        f"Balance: {live_exchange.balance:.2f} USDT  Equity: {live_exchange.equity:.2f} USDT"
    )
    pos = live_exchange.get_position(self.symbol)
    if pos:
        self._log.info(
            f"Position: {pos.side} {pos.quantity:.2f} USDT @ {pos.entry_price:.2f}  "
            f"PnL: {pos.unrealized_pnl:.2f}"
        )
    else:
        self._log.info("Position: flat")
    parts = []
    if self._max_order_notional is not None:
        parts.append(f"max-order={self._max_order_notional:.0f}")
    if self._max_daily_notional is not None:
        parts.append(f"max-daily={self._max_daily_notional:.0f}")
    if self._max_session_loss is not None:
        parts.append(f"max-loss={self._max_session_loss:.0f}")
    if self._order_rate is not None:
        parts.append(f"rate={self._order_rate[0]}/{self._order_rate[1]}s")
    risk_str = ", ".join(parts) if parts else "none configured"
    self._log.info(f"Risk limits: {risk_str}")
```

In `_on_exit()` — keep `print()` for the interactive prompt; replace status log lines:
```python
def _on_exit(self, live_exchange: LiveExchange) -> None:
    if self._stop_sync is not None:
        self._stop_sync.set()
    pos = live_exchange.get_position(self.symbol)
    self._log.info("=== Stopping LiveEngine ===")
    if pos:
        self._log.info(
            f"Open position: {pos.side} {pos.quantity:.2f} USDT @ {pos.entry_price:.2f}"
        )
        try:
            answer = input("Close position before exit? [y/N] ")
        except EOFError:
            answer = "n"
        if answer.strip().lower() == "y":
            side = "sell" if pos.side == "long" else "buy"
            live_exchange.submit_order(self.symbol, side, "market", pos.quantity)
            live_exchange.wait_fills(timeout=30.0)
            self._log.info("Position closed.")
    else:
        self._log.info("No open position.")
```

Also in `run()`, find the `print(f"[INFO] Restored state: ...")` line:
```python
# Before:
print(f"[INFO] Restored state: account={self._account_id} "
      f"{strategy_name}/{self.symbol}/{self.interval}")
# After:
self._log.info(
    f"Restored state: account={self._account_id} "
    f"{strategy_name}/{self.symbol}/{self.interval}"
)
```

- [ ] **Step 4: Update `src/backtest/__main__.py` — add `--log-file` and configure root logger in `cmd_live`**

In the `p_live` argument parser block, add after the existing `extra_params` argument:

```python
    p_live.add_argument("--log-file", default=None, dest="log_file",
                        help="Path to write live engine log (e.g. data/logs/live_BTCUSDT.log)")
```

In `cmd_live`, before `engine = LiveEngine(...)`, add:

```python
    import logging as _logging
    _logging.basicConfig(
        level=_logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
```

And add `log_file=args.log_file` to the `LiveEngine(...)` call:

```python
    engine = LiveEngine(
        strategy_class=strategy_class,
        symbol=args.symbol,
        interval=args.interval,
        leverage=args.leverage,
        connector=connector,
        history_db=history_db,
        account_id=account_id,
        commission_rate=args.commission_rate,
        dry_run=args.dry_run,
        state_db=args.state_db,
        sync_interval=args.sync_interval,
        max_order_notional=args.max_order_notional,
        max_daily_notional=args.max_daily_notional,
        max_session_loss=args.max_session_loss,
        order_rate=order_rate,
        log_file=args.log_file,
    )
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_live_engine.py -v
```
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/backtest/live_engine.py src/backtest/__main__.py tests/test_live_engine.py
git commit -m "feat: add --log-file to live engine with structured logging output"
```

---

## Task 4: live.html Frontend

**Files:**
- Create: `src/backtest/web/static/live.html`

No TDD for pure frontend. Manual verification: start the server and check each tab.

- [ ] **Step 1: Create `src/backtest/web/static/live.html`**

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>实盘监控</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #0f1117; color: #e0e0e0; }
  .header { background: #1a1a2e; padding: 16px 24px; display: flex;
            justify-content: space-between; align-items: center;
            border-bottom: 1px solid #333; }
  .header h1 { font-size: 18px; color: #00d4aa; }
  .filter-bar { display: flex; gap: 12px; padding: 12px 24px;
                background: #141824; border-bottom: 1px solid #1e293b;
                align-items: center; flex-wrap: wrap; }
  .filter-bar label { font-size: 12px; color: #888; }
  .filter-bar select { background: #1e293b; color: #e0e0e0; border: 1px solid #333;
                       padding: 6px 10px; border-radius: 6px; font-size: 13px; }
  .tab-bar { display: flex; gap: 4px; padding: 12px 24px 0;
             border-bottom: 1px solid #1e293b; }
  .tab-btn { padding: 8px 18px; background: none; color: #888; border: none;
             border-bottom: 2px solid transparent; cursor: pointer; font-size: 14px;
             transition: color 0.15s; }
  .tab-btn.active { color: #00d4aa; border-bottom-color: #00d4aa; }
  .tab-pane { display: none; padding: 20px 24px; }
  .tab-pane.active { display: block; }
  /* Cards */
  .kpi-row { display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
             gap: 12px; margin-bottom: 16px; }
  .kpi { background: #1e293b; border-radius: 8px; padding: 14px; }
  .kpi .label { font-size: 11px; color: #888; letter-spacing: 0.5px; }
  .kpi .value { font-size: 22px; font-weight: bold; margin-top: 4px; }
  .kpi .pos-value { color: #22c55e; }
  .kpi .neg-value { color: #ef4444; }
  .kpi .neu-value { color: #00d4aa; }
  /* Chart */
  .chart-box { background: #1e293b; border-radius: 8px; padding: 16px; margin-bottom: 16px; }
  .chart-box h3 { font-size: 13px; color: #aaa; margin-bottom: 8px; }
  #equityChart { height: 260px; }
  /* Table */
  .table-wrap { background: #1e293b; border-radius: 8px; overflow: auto; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th { padding: 10px 12px; text-align: left; color: #888; border-bottom: 1px solid #333;
       position: sticky; top: 0; background: #1e293b; white-space: nowrap; }
  td { padding: 9px 12px; border-bottom: 1px solid #1a1a2e; }
  tr:last-child td { border-bottom: none; }
  .long { color: #22c55e; }
  .short { color: #ef4444; }
  .pnl-pos { color: #22c55e; }
  .pnl-neg { color: #ef4444; }
  .badge { display: inline-block; padding: 2px 7px; border-radius: 10px; font-size: 11px; }
  .badge-filled { background: #22c55e22; color: #22c55e; }
  .badge-pending { background: #facc1522; color: #facc15; }
  .badge-canceled { background: #33333366; color: #888; }
  /* Pagination */
  .pagination { display: flex; gap: 8px; align-items: center; padding: 12px 0 0; }
  .pagination button { padding: 5px 12px; background: #1e293b; color: #e0e0e0;
                        border: 1px solid #333; border-radius: 5px; cursor: pointer; font-size: 13px; }
  .pagination button:disabled { opacity: 0.4; cursor: default; }
  .pagination .pg-info { font-size: 12px; color: #888; }
  /* Summary row */
  .summary-row td { color: #00d4aa; font-weight: 600; border-top: 1px solid #333; }
  /* Logs */
  .log-controls { display: flex; gap: 10px; align-items: center; margin-bottom: 10px; flex-wrap: wrap; }
  .log-controls select { background: #1e293b; color: #e0e0e0; border: 1px solid #333;
                          padding: 6px 10px; border-radius: 6px; font-size: 13px; flex: 1; min-width: 200px; }
  .log-controls input { background: #1e293b; color: #e0e0e0; border: 1px solid #333;
                         padding: 6px 10px; border-radius: 6px; font-size: 13px; flex: 1; min-width: 160px; }
  .log-controls button { padding: 6px 14px; background: #1e293b; color: #00d4aa;
                          border: 1px solid #00d4aa33; border-radius: 6px; cursor: pointer; font-size: 13px; }
  .log-box { background: #1e293b; border-radius: 8px; padding: 12px 16px;
             height: 480px; overflow-y: auto; font-family: 'Courier New', monospace;
             font-size: 12px; line-height: 1.6; }
  .log-line { border-bottom: 1px solid #1a1a2e; padding: 1px 0; }
  .log-line.hidden { display: none; }
  .log-line.level-warn { color: #facc15; }
  .log-line.level-error { color: #ef4444; }
  /* Position card */
  .pos-card { background: #1e293b; border-radius: 8px; padding: 16px; margin-bottom: 16px; }
  .pos-card .pos-title { font-size: 12px; color: #888; margin-bottom: 10px; letter-spacing: 0.5px; }
  .pos-card .pos-flat { color: #888; font-size: 14px; }
  .pos-fields { display: grid; grid-template-columns: repeat(auto-fill, minmax(130px, 1fr)); gap: 10px; }
  .pos-field .f-label { font-size: 11px; color: #888; }
  .pos-field .f-value { font-size: 15px; font-weight: 600; margin-top: 2px; }
  /* Misc */
  .empty-state { padding: 40px; text-align: center; color: #555; font-size: 14px; }
  .spin { animation: spin 0.8s linear infinite; display: inline-block; }
  @keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>

<div class="header">
  <div style="display:flex;align-items:center;gap:24px;">
    <h1>Crypto Backtest</h1>
    <nav style="display:flex;gap:16px;">
      <a href="/" style="color:#888;text-decoration:none;font-size:14px;">回测报告</a>
      <a href="/optimize" style="color:#888;text-decoration:none;font-size:14px;">参数优化</a>
      <a href="/live" style="color:#00d4aa;text-decoration:none;font-size:14px;border-bottom:2px solid #00d4aa;padding-bottom:2px;">实盘监控</a>
    </nav>
  </div>
</div>

<div class="filter-bar">
  <label>账户</label>
  <select id="accountSelect"><option value="">-- 加载中 --</option></select>
  <label>标的</label>
  <select id="symbolSelect"><option value="">-- 选择账户 --</option></select>
</div>

<div class="tab-bar">
  <button class="tab-btn active" data-tab="overview">概览</button>
  <button class="tab-btn" data-tab="orders">订单</button>
  <button class="tab-btn" data-tab="trades">成交</button>
  <button class="tab-btn" data-tab="logs">日志</button>
</div>

<!-- Overview Tab -->
<div id="tab-overview" class="tab-pane active">
  <div class="kpi-row" id="kpiRow">
    <div class="kpi"><div class="label">余额 USDT</div><div class="value neu-value" id="kpiBalance">—</div></div>
    <div class="kpi"><div class="label">权益 USDT</div><div class="value neu-value" id="kpiEquity">—</div></div>
  </div>
  <div class="pos-card">
    <div class="pos-title">当前仓位</div>
    <div id="posContent"><div class="pos-flat">空仓</div></div>
  </div>
  <div class="chart-box">
    <h3>权益曲线</h3>
    <div id="equityChart"></div>
  </div>
</div>

<!-- Orders Tab -->
<div id="tab-orders" class="tab-pane">
  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th>时间</th><th>方向</th><th>类型</th><th>数量(USDT)</th>
          <th>委托价</th><th>成交价</th><th>状态</th><th>手续费</th>
        </tr>
      </thead>
      <tbody id="ordersBody"></tbody>
    </table>
  </div>
  <div class="pagination" id="ordersPagination"></div>
</div>

<!-- Trades Tab -->
<div id="tab-trades" class="tab-pane">
  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th>时间</th><th>方向</th><th>成交价</th><th>数量(USDT)</th>
          <th>已实现PnL</th><th>手续费</th>
        </tr>
      </thead>
      <tbody id="tradesBody"></tbody>
    </table>
  </div>
  <div class="pagination" id="tradesPagination"></div>
</div>

<!-- Logs Tab -->
<div id="tab-logs" class="tab-pane">
  <div class="log-controls">
    <select id="logFileSelect"><option value="">-- 选择日志文件 --</option></select>
    <input type="text" id="logFilter" placeholder="关键字过滤…">
    <button id="logRefreshBtn">刷新</button>
  </div>
  <div class="log-box" id="logBox"><div class="empty-state">请选择日志文件</div></div>
</div>

<script>
const $ = id => document.getElementById(id);
let equityChart = null;
let state = { account: '', symbol: '', ordersPage: 1, tradesPage: 1 };

// ── Init ──────────────────────────────────────────────────────────────────

async function init() {
  const accounts = await fetchJSON('/api/live/accounts');
  const sel = $('accountSelect');
  sel.innerHTML = '<option value="">-- 选择账户 --</option>';
  accounts.forEach(a => {
    const opt = document.createElement('option');
    opt.value = a.account_id;
    opt.textContent = `${a.account_id} (${a.exchange})`;
    sel.appendChild(opt);
  });
  if (accounts.length === 1) {
    sel.value = accounts[0].account_id;
    await onAccountChange();
  }
}

$('accountSelect').addEventListener('change', onAccountChange);
$('symbolSelect').addEventListener('change', onSymbolChange);

async function onAccountChange() {
  state.account = $('accountSelect').value;
  $('symbolSelect').innerHTML = '<option value="">-- 加载中 --</option>';
  if (!state.account) return;
  const symbols = await fetchJSON(`/api/live/symbols?account_id=${encodeURIComponent(state.account)}`);
  const sel = $('symbolSelect');
  sel.innerHTML = '<option value="">-- 选择标的 --</option>';
  symbols.forEach(s => {
    const opt = document.createElement('option');
    opt.value = s; opt.textContent = s;
    sel.appendChild(opt);
  });
  if (symbols.length === 1) {
    sel.value = symbols[0];
    await onSymbolChange();
  }
}

async function onSymbolChange() {
  state.symbol = $('symbolSelect').value;
  state.ordersPage = 1;
  state.tradesPage = 1;
  if (!state.symbol) return;
  const active = document.querySelector('.tab-btn.active').dataset.tab;
  await loadTab(active);
}

// ── Tabs ─────────────────────────────────────────────────────────────────

document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    $('tab-' + btn.dataset.tab).classList.add('active');
    if (state.symbol) loadTab(btn.dataset.tab);
    if (btn.dataset.tab === 'logs') loadLogFiles();
  });
});

async function loadTab(tab) {
  if (tab === 'overview') await loadOverview();
  else if (tab === 'orders') await loadOrders();
  else if (tab === 'trades') await loadTrades();
  else if (tab === 'logs') await loadLogFiles();
}

// ── Overview ──────────────────────────────────────────────────────────────

async function loadOverview() {
  if (!state.account || !state.symbol) return;
  const d = await fetchJSON(`/api/live/overview?account_id=${enc(state.account)}&symbol=${enc(state.symbol)}`);
  $('kpiBalance').textContent = fmt2(d.balance);
  $('kpiEquity').textContent = fmt2(d.equity);
  renderPosition(d.position);
  renderEquityChart(d.equity_curve);
}

function renderPosition(pos) {
  const el = $('posContent');
  if (!pos) { el.innerHTML = '<div class="pos-flat">空仓</div>'; return; }
  const pnlClass = pos.unrealized_pnl >= 0 ? 'pnl-pos' : 'pnl-neg';
  el.innerHTML = `<div class="pos-fields">
    <div class="pos-field"><div class="f-label">方向</div>
      <div class="f-value ${pos.side}">${pos.side === 'long' ? '多' : '空'}</div></div>
    <div class="pos-field"><div class="f-label">数量 USDT</div>
      <div class="f-value">${fmt2(pos.quantity)}</div></div>
    <div class="pos-field"><div class="f-label">入场价</div>
      <div class="f-value">${fmt2(pos.entry_price)}</div></div>
    <div class="pos-field"><div class="f-label">杠杆</div>
      <div class="f-value">${pos.leverage}x</div></div>
    <div class="pos-field"><div class="f-label">保证金</div>
      <div class="f-value">${fmt2(pos.margin)}</div></div>
    <div class="pos-field"><div class="f-label">未实现PnL</div>
      <div class="f-value ${pnlClass}">${fmt2(pos.unrealized_pnl)}</div></div>
  </div>`;
}

function renderEquityChart(curve) {
  if (!equityChart) {
    equityChart = echarts.init($('equityChart'), 'dark');
  }
  equityChart.setOption({
    backgroundColor: 'transparent',
    grid: { left: 60, right: 20, top: 10, bottom: 40 },
    xAxis: { type: 'time', axisLabel: { color: '#888', fontSize: 11 }, axisLine: { lineStyle: { color: '#333' } } },
    yAxis: { type: 'value', axisLabel: { color: '#888', fontSize: 11 }, splitLine: { lineStyle: { color: '#1e293b' } } },
    series: [{
      type: 'line', data: curve, showSymbol: false,
      lineStyle: { color: '#00d4aa', width: 1.5 },
      areaStyle: { color: { type: 'linear', x: 0, y: 0, x2: 0, y2: 1,
        colorStops: [{ offset: 0, color: '#00d4aa33' }, { offset: 1, color: '#00d4aa05' }] } },
    }],
    tooltip: { trigger: 'axis', formatter: p => {
      const d = new Date(p[0].value[0]);
      return `${d.toLocaleString('zh-CN')}<br/><b>${fmt2(p[0].value[1])} USDT</b>`;
    }},
  });
}

// ── Orders ────────────────────────────────────────────────────────────────

async function loadOrders() {
  if (!state.account || !state.symbol) return;
  const d = await fetchJSON(
    `/api/live/orders?account_id=${enc(state.account)}&symbol=${enc(state.symbol)}&page=${state.ordersPage}`
  );
  const body = $('ordersBody');
  body.innerHTML = '';
  if (!d.items.length) { body.innerHTML = '<tr><td colspan="8" class="empty-state">暂无订单</td></tr>'; }
  d.items.forEach(o => {
    const tr = document.createElement('tr');
    const statusBadge = o.status === 'filled'
      ? `<span class="badge badge-filled">已成交</span>`
      : o.status === 'pending'
      ? `<span class="badge badge-pending">挂单中</span>`
      : `<span class="badge badge-canceled">已取消</span>`;
    tr.innerHTML = `
      <td>${fmtTs(o.ts)}</td>
      <td class="${o.side}">${o.side === 'buy' ? '买/多' : '卖/空'}</td>
      <td>${o.type}</td>
      <td>${fmt2(o.quantity)}</td>
      <td>${o.price != null ? fmt2(o.price) : '—'}</td>
      <td>${o.filled_price != null ? fmt2(o.filled_price) : '—'}</td>
      <td>${statusBadge}</td>
      <td>${o.commission != null ? fmt4(o.commission) : '—'}</td>`;
    body.appendChild(tr);
  });
  renderPagination('ordersPagination', d.total, d.page, d.page_size, p => { state.ordersPage = p; loadOrders(); });
}

// ── Trades ────────────────────────────────────────────────────────────────

async function loadTrades() {
  if (!state.account || !state.symbol) return;
  const d = await fetchJSON(
    `/api/live/trades?account_id=${enc(state.account)}&symbol=${enc(state.symbol)}&page=${state.tradesPage}`
  );
  const body = $('tradesBody');
  body.innerHTML = '';
  if (!d.items.length) { body.innerHTML = '<tr><td colspan="6" class="empty-state">暂无成交</td></tr>'; }
  d.items.forEach(t => {
    const tr = document.createElement('tr');
    const pnlClass = (t.realized_pnl || 0) >= 0 ? 'pnl-pos' : 'pnl-neg';
    tr.innerHTML = `
      <td>${fmtTs(t.ts)}</td>
      <td class="${t.side === 'buy' ? 'long' : 'short'}">${t.side === 'buy' ? '买/多' : '卖/空'}</td>
      <td>${fmt2(t.price)}</td>
      <td>${fmt2(t.notional)}</td>
      <td class="${pnlClass}">${fmt2(t.realized_pnl ?? 0)}</td>
      <td>${fmt4(t.commission)}</td>`;
    body.appendChild(tr);
  });
  // summary row
  if (d.items.length) {
    const tr = document.createElement('tr');
    tr.className = 'summary-row';
    const pClass = d.summary.total_pnl >= 0 ? 'pnl-pos' : 'pnl-neg';
    tr.innerHTML = `<td colspan="4">全部汇总</td>
      <td class="${pClass}">${fmt2(d.summary.total_pnl)}</td>
      <td>${fmt4(d.summary.total_commission)}</td>`;
    body.appendChild(tr);
  }
  renderPagination('tradesPagination', d.total, d.page, d.page_size, p => { state.tradesPage = p; loadTrades(); });
}

// ── Logs ──────────────────────────────────────────────────────────────────

async function loadLogFiles() {
  const files = await fetchJSON('/api/live/logs');
  const sel = $('logFileSelect');
  const prev = sel.value;
  sel.innerHTML = '<option value="">-- 选择日志文件 --</option>';
  files.forEach(f => {
    const opt = document.createElement('option');
    opt.value = f.filename;
    const d = new Date(f.mtime);
    opt.textContent = `${f.filename}  (${d.toLocaleString('zh-CN')})`;
    sel.appendChild(opt);
  });
  if (prev && files.find(f => f.filename === prev)) sel.value = prev;
}

$('logFileSelect').addEventListener('change', () => {
  if ($('logFileSelect').value) loadLogContent();
});

$('logRefreshBtn').addEventListener('click', () => {
  if ($('logFileSelect').value) loadLogContent();
});

$('logFilter').addEventListener('input', applyLogFilter);

async function loadLogContent() {
  const filename = $('logFileSelect').value;
  if (!filename) return;
  $('logBox').innerHTML = '<div class="empty-state"><span class="spin">⟳</span> 加载中…</div>';
  const d = await fetchJSON(`/api/live/logs/${encodeURIComponent(filename)}?tail=500`);
  const box = $('logBox');
  box.innerHTML = '';
  if (!d.lines.length) { box.innerHTML = '<div class="empty-state">日志为空</div>'; return; }
  d.lines.forEach(line => {
    const div = document.createElement('div');
    div.className = 'log-line';
    if (line.includes('[WARNING]') || line.includes('[WARN]')) div.classList.add('level-warn');
    if (line.includes('[ERROR]') || line.includes('[CRITICAL]')) div.classList.add('level-error');
    div.textContent = line;
    box.appendChild(div);
  });
  applyLogFilter();
}

function applyLogFilter() {
  const kw = $('logFilter').value.toLowerCase();
  document.querySelectorAll('.log-line').forEach(el => {
    el.classList.toggle('hidden', kw && !el.textContent.toLowerCase().includes(kw));
  });
}

// ── Helpers ───────────────────────────────────────────────────────────────

function renderPagination(containerId, total, page, pageSize, onPage) {
  const el = $(containerId);
  const totalPages = Math.ceil(total / pageSize);
  if (totalPages <= 1) { el.innerHTML = ''; return; }
  el.innerHTML = `
    <button ${page <= 1 ? 'disabled' : ''} id="${containerId}_prev">上一页</button>
    <span class="pg-info">第 ${page} / ${totalPages} 页，共 ${total} 条</span>
    <button ${page >= totalPages ? 'disabled' : ''} id="${containerId}_next">下一页</button>`;
  el.querySelector(`#${containerId}_prev`)?.addEventListener('click', () => onPage(page - 1));
  el.querySelector(`#${containerId}_next`)?.addEventListener('click', () => onPage(page + 1));
}

async function fetchJSON(url) {
  try {
    const r = await fetch(url);
    if (!r.ok) return null;
    return r.json();
  } catch { return null; }
}

const enc = encodeURIComponent;
const fmt2 = v => v == null ? '—' : Number(v).toFixed(2);
const fmt4 = v => v == null ? '—' : Number(v).toFixed(4);
function fmtTs(ms) {
  if (!ms) return '—';
  const d = new Date(ms);
  return d.toLocaleString('zh-CN', { year: 'numeric', month: '2-digit', day: '2-digit',
    hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

init();
</script>
</body>
</html>
```

- [ ] **Step 2: Commit**

```bash
git add src/backtest/web/static/live.html
git commit -m "feat: add live trading monitoring page (live.html)"
```

---

## Task 5: Add `/live` Link to Existing Navbars

**Files:**
- Modify: `src/backtest/web/static/index.html`
- Modify: `src/backtest/web/static/optimize.html`

- [ ] **Step 1: Update `index.html` navbar**

In `index.html` around line 88-89, add the `/live` link after the existing `/optimize` link:

Find:
```html
      <a href="/optimize" style="color:#888;text-decoration:none;font-size:14px;">参数优化</a>
    </nav>
```

Replace with:
```html
      <a href="/optimize" style="color:#888;text-decoration:none;font-size:14px;">参数优化</a>
      <a href="/live" style="color:#888;text-decoration:none;font-size:14px;">实盘监控</a>
    </nav>
```

- [ ] **Step 2: Update `optimize.html` navbar**

In `optimize.html` around line 87-88, add the `/live` link:

Find:
```html
      <a href="/optimize" class="active">参数优化</a>
    </nav>
```

Replace with:
```html
      <a href="/optimize" class="active">参数优化</a>
      <a href="/live">实盘监控</a>
    </nav>
```

- [ ] **Step 3: Run existing web tests to confirm no regressions**

```bash
pytest tests/test_web.py -v
```
Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add src/backtest/web/static/index.html src/backtest/web/static/optimize.html
git commit -m "feat: add 实盘监控 nav link to index and optimize pages"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Covered by |
|---|---|
| New page showing all tables | Task 4 (`live.html`) + Task 2 (API routes) |
| Historical positions P&L, per-trade P&L (like report page) | Task 4 trades tab + Task 2 `/api/live/trades` |
| Strategy logs viewer, time-descending, historical files | Task 4 logs tab + Task 2 `/api/live/logs*` |
| live_engine writes log file | Task 3 |
| Password login (WEB_PASSWORD) | Task 1 |
| Global auth protection | Task 1 (`AuthMiddleware`) |
| Account/symbol dropdown filters | Task 4 filter bar + Task 2 `/api/live/accounts`, `/api/live/symbols` |
| Equity curve chart | Task 4 overview tab (`renderEquityChart`) |
| Navbar on all pages | Task 5 |
| `--live-db` + `--log-dir` CLI params | Task 2 |
| `--log-file` CLI param | Task 3 |

**Placeholder scan:** None found. All code steps contain full implementations.

**Type consistency:** `create_app(db_path, live_db_path=..., log_dir=...)` used consistently across Task 1 (app.py), Task 2 (live_routes.py, test fixture), and Task 3 (test). `LiveEngine(..., log_file=...)` consistent across Task 3 (engine + CLI + test).
