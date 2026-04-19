import json
import sqlite3
from fastapi import APIRouter, Request, HTTPException
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


@router.get("/", response_class=HTMLResponse)
def index():
    html_path = Path(__file__).parent / "static" / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
