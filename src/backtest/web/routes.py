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
    batches.reverse()  # newest first for display
    return batches


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

    initial = equity_curve[0][1]
    first_ts = equity_curve[0][0]
    last_ts = equity_curve[-1][0]
    symbol = row["symbol"]
    interval = row["interval"]
    conn.close()

    # Look up kline close prices from klines.db (sibling of reports.db)
    db_path = _get_db(request)
    klines_db = str(Path(db_path).parent / "klines.db")
    try:
        kconn = sqlite3.connect(klines_db)
        kconn.row_factory = sqlite3.Row
        klines = kconn.execute(
            """SELECT timestamp, close FROM klines
               WHERE symbol = ? AND interval = ? AND timestamp >= ? AND timestamp <= ?
               ORDER BY timestamp""",
            (symbol, interval, first_ts, last_ts),
        ).fetchall()
        kconn.close()
    except Exception:
        return {"benchmark": []}

    if not klines:
        return {"benchmark": []}

    first_close = klines[0]["close"]
    benchmark = [[k["timestamp"], initial * k["close"] / first_close] for k in klines]
    return {"benchmark": benchmark}


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
