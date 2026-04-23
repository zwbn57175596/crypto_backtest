import sqlite3

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pathlib import Path


def _migrate_db(db_path: str) -> None:
    """Ensure optimize_results has batch_id column; backfill old rows."""
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("ALTER TABLE optimize_results ADD COLUMN batch_id TEXT")
        conn.commit()
    except sqlite3.OperationalError:
        pass  # column already exists or table doesn't exist

    try:
        conn.execute("""
            UPDATE optimize_results
            SET batch_id = strftime('%Y%m%dT%H%M%S', created_at)
                           || '_' || strategy || '_' || symbol
            WHERE batch_id IS NULL
        """)
        conn.commit()
    except sqlite3.OperationalError:
        pass  # table doesn't exist yet
    conn.close()


def create_app(db_path: str) -> FastAPI:
    app = FastAPI(title="Crypto Backtest")
    app.state.db_path = db_path
    _migrate_db(db_path)
    from backtest.web.routes import router
    app.include_router(router)
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    return app
