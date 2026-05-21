import sqlite3
import pytest
from backtest.portfolio_engine import PortfolioBacktestEngine


INTERVAL_MS = 3_600_000  # 1 hour in ms


def _make_db(tmp_path, rows):
    db = str(tmp_path / "klines.db")
    conn = sqlite3.connect(db)
    conn.execute("""
        CREATE TABLE klines (
            exchange TEXT, symbol TEXT, interval TEXT, timestamp INTEGER,
            open REAL, high REAL, low REAL, close REAL, volume REAL,
            PRIMARY KEY (exchange, symbol, interval, timestamp)
        )
    """)
    conn.executemany("INSERT INTO klines VALUES (?,?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    conn.close()
    return db


def _bullish_bars(symbol, n, start_price=100.0):
    """n consecutive bullish 1h bars."""
    return [
        ("binance", symbol, "1h", (i + 1) * INTERVAL_MS,
         start_price + i, start_price + i + 1.5, start_price + i - 0.5,
         start_price + i + 1.0, 1000.0)
        for i in range(n)
    ]


def _make_engine(db, symbols, **kwargs):
    defaults = dict(
        interval="1h", exchange="binance",
        balance=1000.0, leverage=10,
        commission_rate=0.0004, funding_rate=0.0001,
        maintenance_margin=0.005,
        consecutive_threshold=7,
        position_multiplier=1.0,
        initial_position_pct=0.1,
        profit_candle_threshold=3,
        max_position_symbols=10,
        margin_ratio_limit=0.9,
    )
    defaults.update(kwargs)
    return PortfolioBacktestEngine(db_path=db, symbols=symbols, **defaults)


# ── Basic result shape ────────────────────────────────────────────

def test_result_has_required_keys(tmp_path):
    db = _make_db(tmp_path, _bullish_bars("AAAUSDT", 8))
    engine = _make_engine(db, ["AAAUSDT"])
    result = engine.run()
    for key in ("trades", "trades_count", "equity_curve", "final_equity",
                "initial_balance", "portfolio_symbols", "per_symbol_stats"):
        assert key in result, f"missing key: {key}"


def test_equity_curve_deduplicated_one_per_ts(tmp_path):
    db = _make_db(tmp_path, _bullish_bars("AAAUSDT", 8) + _bullish_bars("BBBUSDT", 8, 200.0))
    engine = _make_engine(db, ["AAAUSDT", "BBBUSDT"])
    result = engine.run()
    tss = [ts for ts, _ in result["equity_curve"]]
    assert len(tss) == len(set(tss)), "duplicate timestamps in equity_curve"


def test_initial_balance_preserved(tmp_path):
    db = _make_db(tmp_path, _bullish_bars("AAAUSDT", 8))
    engine = _make_engine(db, ["AAAUSDT"])
    result = engine.run()
    assert result["initial_balance"] == 1000.0


# ── Strategy logic: consecutive threshold ────────────────────────

def test_no_trades_below_threshold(tmp_path):
    db = _make_db(tmp_path, _bullish_bars("AAAUSDT", 6))
    engine = _make_engine(db, ["AAAUSDT"])
    result = engine.run()
    assert result["trades_count"] == 0


def test_opens_short_after_7_bullish_bars(tmp_path):
    """7 consecutive bullish bars → SHORT submitted; filled at bar 8's open."""
    db = _make_db(tmp_path, _bullish_bars("AAAUSDT", 8))
    engine = _make_engine(db, ["AAAUSDT"])
    result = engine.run()
    open_trades = [t for t in result["trades"] if t.pnl == 0.0]
    assert len(open_trades) >= 1
    assert open_trades[0].side == "sell"  # short = sell order


# ── Per-symbol stats ─────────────────────────────────────────────

def test_per_symbol_stats_contains_all_symbols(tmp_path):
    rows = _bullish_bars("AAAUSDT", 8) + _bullish_bars("BBBUSDT", 8, 200.0)
    db = _make_db(tmp_path, rows)
    engine = _make_engine(db, ["AAAUSDT", "BBBUSDT"])
    result = engine.run()
    assert "AAAUSDT" in result["per_symbol_stats"]
    assert "BBBUSDT" in result["per_symbol_stats"]


def test_per_symbol_stats_structure(tmp_path):
    db = _make_db(tmp_path, _bullish_bars("AAAUSDT", 8))
    engine = _make_engine(db, ["AAAUSDT"])
    result = engine.run()
    stats = result["per_symbol_stats"]["AAAUSDT"]
    assert "total_trades" in stats
    assert "win_rate" in stats
    assert "net_pnl" in stats


# ── Account risk control: max_position_symbols ───────────────────

def test_max_position_symbols_limits_new_opens(tmp_path):
    """Both symbols trigger at same bar but max=1 → only 1 position opened."""
    rows = _bullish_bars("AAAUSDT", 8) + _bullish_bars("BBBUSDT", 8, 200.0)
    db = _make_db(tmp_path, rows)
    engine = _make_engine(db, ["AAAUSDT", "BBBUSDT"], max_position_symbols=1)
    result = engine.run()
    open_trades = [t for t in result["trades"] if t.pnl == 0.0]
    assert len(open_trades) <= 1


# ── Equity & balance ─────────────────────────────────────────────

def test_final_equity_positive_after_run(tmp_path):
    db = _make_db(tmp_path, _bullish_bars("AAAUSDT", 8))
    engine = _make_engine(db, ["AAAUSDT"])
    result = engine.run()
    assert result["final_equity"] > 0


def test_portfolio_symbols_in_result(tmp_path):
    rows = _bullish_bars("AAAUSDT", 8) + _bullish_bars("BBBUSDT", 8, 200.0)
    db = _make_db(tmp_path, rows)
    engine = _make_engine(db, ["AAAUSDT", "BBBUSDT"])
    result = engine.run()
    assert result["portfolio_symbols"] == ["AAAUSDT", "BBBUSDT"]
