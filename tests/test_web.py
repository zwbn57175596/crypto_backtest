import json
import sqlite3
import pytest
from fastapi.testclient import TestClient
from backtest.web.app import create_app


@pytest.fixture
def report_db(tmp_path):
    db_path = str(tmp_path / "reports.db")
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy TEXT, symbol TEXT, interval TEXT,
            created_at TEXT, report_json TEXT
        )
    """)
    report = {
        "net_return": 0.05, "max_drawdown": 0.02, "sharpe_ratio": 1.5,
        "win_rate": 0.6, "total_trades": 10,
        "equity_curve": [[1704067200000, 10000], [1704070800000, 10500]],
        "trades": [],
    }
    conn.execute(
        "INSERT INTO reports (strategy, symbol, interval, created_at, report_json) VALUES (?,?,?,?,?)",
        ("TestStrategy", "BTCUSDT", "1h", "2024-01-01T00:00:00", json.dumps(report)),
    )
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def client(report_db):
    app = create_app(report_db)
    return TestClient(app)


def test_get_reports_list(client):
    resp = client.get("/api/reports")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["strategy"] == "TestStrategy"


def test_get_report_detail(client):
    resp = client.get("/api/reports/1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["net_return"] == 0.05


def test_get_report_not_found(client):
    resp = client.get("/api/reports/999")
    assert resp.status_code == 404


def test_index_page(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "echarts" in resp.text.lower()


@pytest.fixture
def db_with_optimize(tmp_path):
    """DB with both reports and optimize_results tables."""
    db_path = str(tmp_path / "reports.db")
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy TEXT, symbol TEXT, interval TEXT,
            created_at TEXT, report_json TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE optimize_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy TEXT NOT NULL,
            symbol TEXT NOT NULL,
            interval TEXT NOT NULL,
            start_date TEXT NOT NULL,
            end_date TEXT NOT NULL,
            objective TEXT NOT NULL,
            score REAL NOT NULL,
            params_json TEXT NOT NULL,
            report_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute(
        "INSERT INTO optimize_results (strategy,symbol,interval,start_date,end_date,objective,score,params_json,report_json,created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
        ("ShadowPower", "BTCUSDT", "15m", "2024-01-01", "2024-06-30", "sharpe_ratio", 2.31,
         json.dumps({"DECISION_LEN": 40, "SHADOW_FACTOR": 2.5}),
         json.dumps({"net_return": 1.85, "max_drawdown": 0.12, "sharpe_ratio": 2.31, "win_rate": 0.55, "sortino_ratio": 3.1}),
         "2026-04-20T10:00:00+00:00"),
    )
    conn.execute(
        "INSERT INTO optimize_results (strategy,symbol,interval,start_date,end_date,objective,score,params_json,report_json,created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
        ("ShadowPower", "BTCUSDT", "15m", "2024-01-01", "2024-06-30", "sharpe_ratio", 1.85,
         json.dumps({"DECISION_LEN": 50, "SHADOW_FACTOR": 3.0}),
         json.dumps({"net_return": 1.2, "max_drawdown": 0.15, "sharpe_ratio": 1.85, "win_rate": 0.50, "sortino_ratio": 2.5}),
         "2026-04-20T10:00:00+00:00"),
    )
    conn.execute(
        "INSERT INTO optimize_results (strategy,symbol,interval,start_date,end_date,objective,score,params_json,report_json,created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
        ("MaCross", "ETHUSDT", "1h", "2024-01-01", "2024-03-01", "sharpe_ratio", 0.95,
         json.dumps({"short_period": 7, "long_period": 25}),
         json.dumps({"net_return": 0.3, "max_drawdown": 0.08, "sharpe_ratio": 0.95, "win_rate": 0.45, "sortino_ratio": 1.2}),
         "2026-04-19T10:00:00+00:00"),
    )
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def opt_client(db_with_optimize):
    app = create_app(db_with_optimize)
    return TestClient(app)


def test_get_optimize_results_all(opt_client):
    resp = opt_client.get("/api/optimize_results")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 3
    assert data[0]["score"] == 2.31
    assert data[2]["score"] == 0.95


def test_get_optimize_results_filter_strategy(opt_client):
    resp = opt_client.get("/api/optimize_results?strategy=ShadowPower")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 2
    assert all(d["strategy"] == "ShadowPower" for d in data)


def test_get_optimize_results_filter_symbol(opt_client):
    resp = opt_client.get("/api/optimize_results?symbol=ETHUSDT")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["strategy"] == "MaCross"


def test_get_optimize_strategies(opt_client):
    resp = opt_client.get("/api/optimize_results/strategies")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 2
    sp = next(d for d in data if d["strategy"] == "ShadowPower")
    assert sp["symbol"] == "BTCUSDT"
    assert sp["count"] == 2
    assert sp["best_score"] == 2.31


def test_optimize_page(opt_client):
    resp = opt_client.get("/optimize")
    assert resp.status_code == 200
    assert "echarts" in resp.text.lower()
