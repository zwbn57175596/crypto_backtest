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
