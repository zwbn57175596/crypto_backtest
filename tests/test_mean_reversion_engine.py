import sqlite3

from backtest.mean_reversion import MRParams
from backtest.mean_reversion_engine import MeanReversionPortfolioEngine

INTERVAL_MS = 3_600_000


def _make_db(tmp_path, rows):
    db = tmp_path / "k.db"
    conn = sqlite3.connect(db)
    conn.execute("""
        CREATE TABLE klines (
            exchange TEXT, symbol TEXT, interval TEXT, timestamp INTEGER,
            open REAL, high REAL, low REAL, close REAL, volume REAL,
            PRIMARY KEY (exchange, symbol, interval, timestamp)
        )
    """)
    conn.executemany(
        "INSERT INTO klines VALUES (?,?,?,?,?,?,?,?,?)", rows
    )
    conn.commit()
    conn.close()
    return str(db)


def _row(ts, o, h, l, c):
    return ("okx", "AAAUSDT", "1h", ts, o, h, l, c, 1.0)


def test_engine_runs_one_profitable_short_trade(tmp_path):
    # 3 根连涨 → 触发做空；随后限价成交，价格回落触发止盈
    rows = [
        _row(0 * INTERVAL_MS, 100, 101, 100, 101),   # up  count=1
        _row(1 * INTERVAL_MS, 101, 102, 101, 102),   # up  count=2
        _row(2 * INTERVAL_MS, 102, 103, 102, 103),   # up  count=3 → submit sell limit @103*1.001=103.103
        _row(3 * INTERVAL_MS, 103, 104, 102.5, 103.5),  # high 104 >= 103.103 → sell fills; bracket set
        _row(4 * INTERVAL_MS, 103, 103.2, 101, 102),    # low 101 <= tp(103.103*0.99≈102.07) → TP
        _row(5 * INTERVAL_MS, 102, 102, 101.9, 101.95),
    ]
    db = _make_db(tmp_path, rows)
    params = MRParams(threshold=3, tp_pct=0.01, sl_pct=0.01, time_stop_bars=10,
                      risk_pct=0.01, entry_offset_pct=0.001, entry_ttl_bars=3)
    engine = MeanReversionPortfolioEngine(
        db_path=db, symbols=["AAAUSDT"], interval="1h", exchange="okx",
        params=params, balance=10000.0, leverage=10,
    )
    result = engine.run()

    order_ids = [t.order_id for t in result["trades"]]
    sides = [t.side for t in result["trades"]]
    assert "bracket-tp" in order_ids          # 止盈平仓
    assert sides[0] == "sell"                  # 首单为反手做空
    assert result["final_equity"] > result["initial_balance"]   # 单笔净盈利（含双边 maker 费）
    assert result["entries_submitted"] >= 1
    assert result["entries_filled"] >= 1
    assert 0.0 <= result["miss_rate"] <= 1.0


def test_engine_reports_miss_when_limit_never_fills(tmp_path):
    # 连涨触发做空挂单，但价格再不上探 → 挂单过期，miss_rate 上升，无成交
    rows = [
        _row(0 * INTERVAL_MS, 100, 101, 100, 101),
        _row(1 * INTERVAL_MS, 101, 102, 101, 102),
        _row(2 * INTERVAL_MS, 102, 103, 102, 103),   # count=3 → sell limit @103.103, ttl=2 bars
        _row(3 * INTERVAL_MS, 102, 102.5, 101, 101.5),  # high 102.5 < 103.103: no fill
        _row(4 * INTERVAL_MS, 101, 101.4, 100, 100.5),  # high < 103.103, ts >= expire: canceled
        _row(5 * INTERVAL_MS, 100, 100.2, 99, 99.5),
    ]
    db = _make_db(tmp_path, rows)
    params = MRParams(threshold=3, tp_pct=0.01, sl_pct=0.01, time_stop_bars=10,
                      risk_pct=0.01, entry_offset_pct=0.001, entry_ttl_bars=2)
    engine = MeanReversionPortfolioEngine(
        db_path=db, symbols=["AAAUSDT"], interval="1h", exchange="okx",
        params=params, balance=10000.0, leverage=10,
    )
    result = engine.run()
    assert result["entries_submitted"] >= 1
    assert result["entries_filled"] == 0
    assert result["miss_rate"] == 1.0
    assert result["final_equity"] == result["initial_balance"]  # 无成交无费


def test_engine_run_is_idempotent(tmp_path):
    rows = [
        _row(0 * INTERVAL_MS, 100, 101, 100, 101),
        _row(1 * INTERVAL_MS, 101, 102, 101, 102),
        _row(2 * INTERVAL_MS, 102, 103, 102, 103),
        _row(3 * INTERVAL_MS, 103, 104, 102.5, 103.5),
        _row(4 * INTERVAL_MS, 103, 103.2, 101, 102),
        _row(5 * INTERVAL_MS, 102, 102, 101.9, 101.95),
    ]
    db = _make_db(tmp_path, rows)
    params = MRParams(threshold=3, tp_pct=0.01, sl_pct=0.01, time_stop_bars=10,
                      risk_pct=0.01, entry_offset_pct=0.001, entry_ttl_bars=3)
    engine = MeanReversionPortfolioEngine(
        db_path=db, symbols=["AAAUSDT"], interval="1h", exchange="okx",
        params=params, balance=10000.0, leverage=10)
    r1 = engine.run()
    r2 = engine.run()  # same instance, twice
    assert r1["final_equity"] == r2["final_equity"]
    assert r1["entries_submitted"] == r2["entries_submitted"]
    assert r1["miss_rate"] == r2["miss_rate"]
    assert len(r1["trades"]) == len(r2["trades"])
