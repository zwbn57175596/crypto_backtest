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
