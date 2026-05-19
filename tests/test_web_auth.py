import hashlib
import sqlite3

import pytest
from fastapi.testclient import TestClient

from backtest.web.app import create_app


@pytest.fixture
def bare_db(tmp_path):
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


def test_no_password_env_passes_all_requests(bare_db, monkeypatch):
    monkeypatch.delenv("WEB_PASSWORD", raising=False)
    app = create_app(bare_db)
    client = TestClient(app)
    assert client.get("/api/reports").status_code == 200


def test_password_set_api_without_cookie_returns_401(bare_db, monkeypatch):
    monkeypatch.setenv("WEB_PASSWORD", "secret")
    app = create_app(bare_db)
    client = TestClient(app, follow_redirects=False)
    resp = client.get("/api/reports")
    assert resp.status_code == 401
    assert resp.json()["detail"] == "Unauthorized"


def test_password_set_page_without_cookie_redirects_to_login(bare_db, monkeypatch):
    monkeypatch.setenv("WEB_PASSWORD", "secret")
    app = create_app(bare_db)
    client = TestClient(app, follow_redirects=False)
    resp = client.get("/")
    assert resp.status_code == 302
    assert resp.headers["location"] == "/login"


def test_static_files_skip_auth(bare_db, monkeypatch):
    monkeypatch.setenv("WEB_PASSWORD", "secret")
    app = create_app(bare_db)
    client = TestClient(app, follow_redirects=False)
    resp = client.get("/static/nonexistent.css")
    assert resp.status_code not in (401, 302)


def test_valid_cookie_passes(bare_db, monkeypatch):
    monkeypatch.setenv("WEB_PASSWORD", "secret")
    session = hashlib.sha256(b"secret").hexdigest()
    app = create_app(bare_db)
    client = TestClient(app, follow_redirects=False, cookies={"session": session})
    assert client.get("/api/reports").status_code == 200


def test_login_correct_password_sets_cookie_and_redirects(bare_db, monkeypatch):
    monkeypatch.setenv("WEB_PASSWORD", "secret")
    app = create_app(bare_db)
    client = TestClient(app, follow_redirects=False)
    resp = client.post("/login", data={"password": "secret"})
    assert resp.status_code == 302
    expected = hashlib.sha256(b"secret").hexdigest()
    assert resp.cookies.get("session") == expected


def test_login_wrong_password_returns_401(bare_db, monkeypatch):
    monkeypatch.setenv("WEB_PASSWORD", "secret")
    app = create_app(bare_db)
    client = TestClient(app, follow_redirects=False)
    resp = client.post("/login", data={"password": "wrong"})
    assert resp.status_code == 401
