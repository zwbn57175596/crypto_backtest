from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pathlib import Path


def create_app(db_path: str) -> FastAPI:
    app = FastAPI(title="Crypto Backtest")
    app.state.db_path = db_path
    from backtest.web.routes import router
    app.include_router(router)
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    return app
