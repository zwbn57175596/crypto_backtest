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
        pass  # column already exists or table doesn't exist

    try:
        conn.execute("ALTER TABLE optimize_results ADD COLUMN leverage REAL")
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
