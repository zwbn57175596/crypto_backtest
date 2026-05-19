import hashlib
import os
import re
import sqlite3
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
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
        "SELECT ts, equity FROM positions WHERE account_id=? AND symbol=? ORDER BY ts ASC LIMIT 2000",
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
    raw = _log_dir(request)
    if not raw:
        return []
    log_path = Path(raw)
    if not log_path.is_dir():
        return []
    result = []
    for f in sorted(log_path.glob("*.log"), key=lambda f: f.stat().st_mtime if f.exists() else 0, reverse=True):
        try:
            st = f.stat()
            result.append({"filename": f.name, "mtime": int(st.st_mtime * 1000), "size": st.st_size})
        except FileNotFoundError:
            pass
    return result


@router.get("/api/live/logs/{filename}")
def get_log_file(filename: str, request: Request, tail: int = Query(500, ge=1, le=10000)):
    if not re.fullmatch(r"[a-zA-Z0-9_\-\.]+\.log", filename):
        raise HTTPException(status_code=400, detail="Invalid filename")
    raw_log_dir = _log_dir(request)
    if not raw_log_dir:
        raise HTTPException(status_code=404, detail="Log directory not configured")
    log_dir_path = Path(raw_log_dir)
    if not log_dir_path.is_dir():
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
