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
