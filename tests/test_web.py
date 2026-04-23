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
        CREATE TABLE optimize_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy TEXT NOT NULL, symbol TEXT NOT NULL,
            interval TEXT NOT NULL, start_date TEXT NOT NULL,
            end_date TEXT NOT NULL, objective TEXT NOT NULL,
            score REAL NOT NULL, params_json TEXT NOT NULL,
            report_json TEXT NOT NULL, created_at TEXT NOT NULL,
            batch_id TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy TEXT, symbol TEXT, interval TEXT,
            created_at TEXT, report_json TEXT,
            optimize_result_id INTEGER
        )
    """)
    report = {
        "net_return": 0.05, "max_drawdown": 0.02, "sharpe_ratio": 1.5,
        "win_rate": 0.6, "total_trades": 10,
        "equity_curve": [[1704067200000, 10000], [1704070800000, 10500]],
        "trades": [],
    }
    conn.execute(
        "INSERT INTO reports (strategy, symbol, interval, created_at, report_json, optimize_result_id) VALUES (?,?,?,?,?,?)",
        ("TestStrategy", "BTCUSDT", "1h", "2024-01-01T00:00:00", json.dumps(report), None),
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
            created_at TEXT NOT NULL,
            batch_id TEXT
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


@pytest.fixture
def db_with_linked_report(tmp_path):
    """DB with a report linked to an optimize_results row via optimize_result_id."""
    db_path = str(tmp_path / "reports.db")
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE optimize_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy TEXT NOT NULL, symbol TEXT NOT NULL,
            interval TEXT NOT NULL, start_date TEXT NOT NULL,
            end_date TEXT NOT NULL, objective TEXT NOT NULL,
            score REAL NOT NULL, params_json TEXT NOT NULL,
            report_json TEXT NOT NULL, created_at TEXT NOT NULL,
            batch_id TEXT
        )
    """)
    conn.execute(
        "INSERT INTO optimize_results (strategy,symbol,interval,start_date,end_date,objective,score,params_json,report_json,created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
        ("TestStrategy", "BTCUSDT", "1h", "2024-01-01", "2024-12-31",
         "sharpe_ratio", 2.04,
         json.dumps({"CONSECUTIVE_THRESHOLD": 5, "POSITION_MULTIPLIER": 1.1}, sort_keys=True),
         json.dumps({}),
         "2026-04-21T00:00:00+00:00"),
    )
    conn.execute("""
        CREATE TABLE reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy TEXT, symbol TEXT, interval TEXT,
            created_at TEXT, report_json TEXT,
            optimize_result_id INTEGER
        )
    """)
    report = {
        "net_return": 0.5, "max_drawdown": 0.1, "sharpe_ratio": 2.04,
        "win_rate": 0.6, "total_trades": 20, "equity_curve": [], "trades": [],
    }
    conn.execute(
        "INSERT INTO reports (strategy,symbol,interval,created_at,report_json,optimize_result_id) VALUES (?,?,?,?,?,?)",
        ("TestStrategy_opt1", "BTCUSDT", "1h", "2026-04-21T00:00:00+00:00",
         json.dumps(report), 1),
    )
    # Unlinked report (standalone run)
    conn.execute(
        "INSERT INTO reports (strategy,symbol,interval,created_at,report_json,optimize_result_id) VALUES (?,?,?,?,?,?)",
        ("TestStrategy", "BTCUSDT", "1h", "2026-04-21T01:00:00+00:00",
         json.dumps(report), None),
    )
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def linked_client(db_with_linked_report):
    app = create_app(db_with_linked_report)
    return TestClient(app)


def test_get_report_includes_optimize_params_when_linked(linked_client):
    resp = linked_client.get("/api/reports/1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["optimize_params"] == {"CONSECUTIVE_THRESHOLD": 5, "POSITION_MULTIPLIER": 1.1}
    assert data["optimize_score"] == pytest.approx(2.04)
    assert data["optimize_objective"] == "sharpe_ratio"


def test_get_report_optimize_params_null_when_unlinked(linked_client):
    resp = linked_client.get("/api/reports/2")
    assert resp.status_code == 200
    data = resp.json()
    assert data["optimize_params"] is None
    assert data["optimize_score"] is None
    assert data["optimize_objective"] is None


@pytest.fixture
def db_with_batches(tmp_path):
    """Database with optimize_results that have batch_ids."""
    db = str(tmp_path / "reports.db")
    conn = sqlite3.connect(db)
    conn.execute("""
        CREATE TABLE optimize_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy TEXT NOT NULL, symbol TEXT NOT NULL, interval TEXT NOT NULL,
            start_date TEXT NOT NULL, end_date TEXT NOT NULL,
            objective TEXT NOT NULL, score REAL NOT NULL,
            params_json TEXT NOT NULL, report_json TEXT NOT NULL,
            created_at TEXT NOT NULL, batch_id TEXT
        )
    """)
    # Batch 1: 2 trials
    conn.execute(
        "INSERT INTO optimize_results (strategy,symbol,interval,start_date,end_date,objective,score,params_json,report_json,created_at,batch_id) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        ("MaCross","BTCUSDT","1h","2026-01-01","2026-03-31","sharpe_ratio",1.5,'{"x":1}','{"net_return":0.1}',
         "2026-04-15T14:00:00+00:00","20260415T140000_MaCross_BTCUSDT"),
    )
    conn.execute(
        "INSERT INTO optimize_results (strategy,symbol,interval,start_date,end_date,objective,score,params_json,report_json,created_at,batch_id) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        ("MaCross","BTCUSDT","1h","2026-01-01","2026-03-31","sharpe_ratio",1.2,'{"x":2}','{"net_return":0.05}',
         "2026-04-15T14:00:00+00:00","20260415T140000_MaCross_BTCUSDT"),
    )
    # Batch 2: 1 trial
    conn.execute(
        "INSERT INTO optimize_results (strategy,symbol,interval,start_date,end_date,objective,score,params_json,report_json,created_at,batch_id) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        ("MaCross","BTCUSDT","1h","2026-01-01","2026-06-30","sharpe_ratio",2.3,'{"x":3}','{"net_return":0.2}',
         "2026-04-23T12:00:00+00:00","20260423T120000_MaCross_BTCUSDT"),
    )
    conn.commit()
    conn.close()
    return db


@pytest.fixture
def batch_client(db_with_batches):
    app = create_app(db_with_batches)
    return TestClient(app)


def test_get_batches(batch_client):
    resp = batch_client.get("/api/optimize_results/batches?strategy=MaCross&symbol=BTCUSDT")
    assert resp.status_code == 200
    batches = resp.json()
    assert len(batches) == 2
    # Ordered by created_at DESC — newest batch first
    assert batches[0]["batch_id"] == "20260423T120000_MaCross_BTCUSDT"
    assert batches[0]["count"] == 1
    assert batches[0]["best_score"] == 2.3
    assert batches[0]["start_date"] == "2026-01-01"
    assert batches[0]["end_date"] == "2026-06-30"
    assert batches[0]["objective"] == "sharpe_ratio"
    assert batches[0]["batch_number"] == 2
    assert batches[1]["batch_id"] == "20260415T140000_MaCross_BTCUSDT"
    assert batches[1]["count"] == 2
    assert batches[1]["best_score"] == 1.5
    assert batches[1]["batch_number"] == 1


def test_get_batches_empty(batch_client):
    resp = batch_client.get("/api/optimize_results/batches?strategy=NoExist&symbol=BTCUSDT")
    assert resp.status_code == 200
    assert resp.json() == []


def test_get_optimize_results_filtered_by_batch_ids(batch_client):
    resp = batch_client.get(
        "/api/optimize_results?strategy=MaCross&symbol=BTCUSDT&batch_ids=20260423T120000_MaCross_BTCUSDT"
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["score"] == 2.3


def test_get_optimize_results_without_batch_ids_returns_all(batch_client):
    resp = batch_client.get("/api/optimize_results?strategy=MaCross&symbol=BTCUSDT")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 3


@pytest.fixture
def db_with_batch_linked_report(tmp_path):
    """DB with reports linked to optimize_results in a batch."""
    db = str(tmp_path / "reports.db")
    conn = sqlite3.connect(db)
    conn.execute("""
        CREATE TABLE optimize_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy TEXT NOT NULL, symbol TEXT NOT NULL, interval TEXT NOT NULL,
            start_date TEXT NOT NULL, end_date TEXT NOT NULL,
            objective TEXT NOT NULL, score REAL NOT NULL,
            params_json TEXT NOT NULL, report_json TEXT NOT NULL,
            created_at TEXT NOT NULL, batch_id TEXT
        )
    """)
    # 3 trials in same batch
    for score, params in [(2.3, '{"x":3}'), (1.8, '{"x":2}'), (1.5, '{"x":1}')]:
        conn.execute(
            """INSERT INTO optimize_results
               (strategy,symbol,interval,start_date,end_date,objective,score,params_json,report_json,created_at,batch_id)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            ("MaCross","BTCUSDT","1h","2026-01-01","2026-06-30","sharpe_ratio",score,params,
             '{"net_return":0.1}',
             "2026-04-23T12:00:00+00:00","20260423T120000_MaCross_BTCUSDT"),
        )
    report_json = json.dumps({
        "net_return": 0.1, "annual_return": 0.2, "max_drawdown": 0.05,
        "max_dd_duration": 0, "sharpe_ratio": 1.8, "sortino_ratio": 2.0,
        "win_rate": 0.6, "profit_factor": 1.5, "total_trades": 10,
        "long_trades": 5, "short_trades": 5, "avg_hold_time": 3600000,
        "total_commission": 10.0, "total_funding": 0.0,
        "equity_curve": [[1000000, 10000]], "trades": [],
    })
    conn.execute("""
        CREATE TABLE reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy TEXT, symbol TEXT, interval TEXT,
            created_at TEXT, report_json TEXT,
            optimize_result_id INTEGER
        )
    """)
    # Link to second trial (id=2, score=1.8)
    conn.execute(
        "INSERT INTO reports (strategy,symbol,interval,created_at,report_json,optimize_result_id) VALUES (?,?,?,?,?,?)",
        ("MaCross","BTCUSDT","1h","2026-04-23T12:00:00+00:00", report_json, 2),
    )
    conn.commit()
    conn.close()
    return db


@pytest.fixture
def batch_linked_client(db_with_batch_linked_report):
    app = create_app(db_with_batch_linked_report)
    return TestClient(app)


def test_get_report_includes_batch_context(batch_linked_client):
    resp = batch_linked_client.get("/api/reports/1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["optimize_batch_id"] == "20260423T120000_MaCross_BTCUSDT"
    assert data["optimize_batch_created_at"] == "2026-04-23T12:00:00+00:00"
    assert data["optimize_start_date"] == "2026-01-01"
    assert data["optimize_end_date"] == "2026-06-30"
    assert data["optimize_rank"] == 2  # second best in batch (score 1.8, behind 2.3)
    assert data["optimize_batch_total"] == 3


@pytest.fixture
def db_with_klines(tmp_path):
    """Database with reports and a separate klines.db."""
    db = str(tmp_path / "reports.db")
    conn = sqlite3.connect(db)
    report_json = json.dumps({
        "net_return": 0.1, "annual_return": 0.2, "max_drawdown": 0.05,
        "max_dd_duration": 0, "sharpe_ratio": 1.8, "sortino_ratio": 2.0,
        "win_rate": 0.6, "profit_factor": 1.5, "total_trades": 10,
        "long_trades": 5, "short_trades": 5, "avg_hold_time": 3600000,
        "total_commission": 10.0, "total_funding": 0.0,
        "equity_curve": [[1000, 10000], [2000, 10500], [3000, 10200]],
        "trades": [],
    })
    conn.execute("""
        CREATE TABLE reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy TEXT, symbol TEXT, interval TEXT,
            created_at TEXT, report_json TEXT, optimize_result_id INTEGER
        )
    """)
    conn.execute(
        "INSERT INTO reports (strategy,symbol,interval,created_at,report_json) VALUES (?,?,?,?,?)",
        ("MaCross", "BTCUSDT", "1h", "2026-04-23", report_json),
    )
    conn.commit()
    conn.close()

    # Create klines.db in same directory
    klines_db = str(tmp_path / "klines.db")
    kconn = sqlite3.connect(klines_db)
    kconn.execute("""
        CREATE TABLE klines (
            exchange TEXT, symbol TEXT, interval TEXT, timestamp INTEGER,
            open REAL, high REAL, low REAL, close REAL, volume REAL,
            PRIMARY KEY (exchange, symbol, interval, timestamp)
        )
    """)
    kconn.execute("INSERT INTO klines VALUES (?,?,?,?,?,?,?,?,?)",
        ("binance", "BTCUSDT", "1h", 1000, 100, 105, 95, 102, 1000))
    kconn.execute("INSERT INTO klines VALUES (?,?,?,?,?,?,?,?,?)",
        ("binance", "BTCUSDT", "1h", 2000, 102, 110, 100, 108, 1000))
    kconn.execute("INSERT INTO klines VALUES (?,?,?,?,?,?,?,?,?)",
        ("binance", "BTCUSDT", "1h", 3000, 108, 112, 105, 99, 1000))
    kconn.commit()
    kconn.close()
    return db


@pytest.fixture
def klines_client(db_with_klines):
    app = create_app(db_with_klines)
    return TestClient(app)


def test_get_benchmark(klines_client):
    resp = klines_client.get("/api/reports/1/benchmark")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["benchmark"]) == 3
    # First: 10000 * 102/102 = 10000
    assert data["benchmark"][0][1] == 10000.0
    # Second: 10000 * 108/102 ≈ 10588.24
    assert abs(data["benchmark"][1][1] - 10588.24) < 1
    # Third: 10000 * 99/102 ≈ 9705.88
    assert abs(data["benchmark"][2][1] - 9705.88) < 1
