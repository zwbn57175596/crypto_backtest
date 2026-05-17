# tests/test_live_engine.py
import json
import tempfile
from unittest.mock import MagicMock, patch

from backtest.strategy import BaseStrategy
from backtest.models import Bar


def _make_connector(balance=1000.0, position=None, mark_price=50000.0):
    connector = MagicMock()
    connector.exchange_name = "binance"
    connector.exchange_info.return_value = {
        "symbol": "BTCUSDT",
        "filters": [{"filterType": "LOT_SIZE", "stepSize": "0.001"}],
    }
    connector.fetch_balance.return_value = balance
    connector.fetch_position.return_value = position
    connector.fetch_mark_price.return_value = mark_price
    connector.fetch_orders.return_value = []
    connector.fetch_trades.return_value = []
    return connector


def _make_history_db():
    return MagicMock()


def _make_bar(ts: int = 1_700_000_000_000) -> Bar:
    return Bar("BTCUSDT", ts, 50000.0, 51000.0, 49000.0, 50500.0, 100.0, "1h")


class _NoOpStrategy(BaseStrategy):
    def on_bar(self, bar): pass


class TestProcessBar:
    def _make_engine(self, connector=None, history_db=None):
        from backtest.live_engine import LiveEngine
        from backtest.live_exchange import LiveExchange
        conn = connector or _make_connector()
        hdb = history_db or _make_history_db()
        # Use mkdtemp so the directory persists beyond the helper's scope.
        tmp = tempfile.mkdtemp()
        engine = LiveEngine(
            strategy_class=_NoOpStrategy,
            symbol="BTCUSDT",
            interval="1h",
            leverage=10,
            connector=conn,
            history_db=hdb,
            account_id="testacc",
            dry_run=True,
            state_db=f"{tmp}/state.db",
        )
        ex = LiveExchange(
            connector=conn, history_db=hdb,
            account_id="testacc", symbol="BTCUSDT",
            leverage=10, commission_rate=0.0004, dry_run=True,
        )
        ex.sync()
        strategy = _NoOpStrategy(exchange=ex, symbol="BTCUSDT")
        strategy.on_init()
        return engine, ex, strategy

    def test_process_bar_prints_status(self, capsys):
        engine, ex, strategy = self._make_engine()
        engine._process_bar(_make_bar(), strategy, ex)
        captured = capsys.readouterr()
        assert "balance=" in captured.out

    def test_process_bar_saves_state(self):
        engine, ex, strategy = self._make_engine()
        engine._process_bar(_make_bar(), strategy, ex)
        engine._history_db.record_state_snapshot.assert_called_once()

    def test_process_bar_exception_does_not_propagate(self):
        engine, ex, strategy = self._make_engine()
        strategy._push_bar = MagicMock(side_effect=RuntimeError("boom"))
        engine._process_bar(_make_bar(), strategy, ex)  # must not raise


class TestDoSync:
    def test_do_sync_fetches_orders_and_trades(self):
        from backtest.live_engine import LiveEngine
        connector = _make_connector()
        history_db = _make_history_db()
        history_db.latest_ts.return_value = 0
        with tempfile.TemporaryDirectory() as tmp:
            engine = LiveEngine(
                strategy_class=_NoOpStrategy, symbol="BTCUSDT", interval="1h",
                leverage=10, connector=connector, history_db=history_db,
                account_id="testacc", state_db=f"{tmp}/state.db",
            )
            engine._do_sync()
        connector.fetch_orders.assert_called_once_with("BTCUSDT", since_ms=None)
        connector.fetch_trades.assert_called_once_with("BTCUSDT", since_ms=None)

    def test_do_sync_uses_latest_ts_as_since(self):
        from backtest.live_engine import LiveEngine
        connector = _make_connector()
        history_db = _make_history_db()
        history_db.latest_ts.side_effect = lambda acc, ex, sym, table: (
            1_700_000_000_000 if table == "orders" else 1_699_000_000_000
        )
        with tempfile.TemporaryDirectory() as tmp:
            engine = LiveEngine(
                strategy_class=_NoOpStrategy, symbol="BTCUSDT", interval="1h",
                leverage=10, connector=connector, history_db=history_db,
                account_id="testacc", state_db=f"{tmp}/state.db",
            )
            engine._do_sync()
        connector.fetch_orders.assert_called_once_with("BTCUSDT", since_ms=1_700_000_000_000)
        connector.fetch_trades.assert_called_once_with("BTCUSDT", since_ms=1_699_000_000_000)


class TestStateDB:
    def test_save_and_load_roundtrip(self):
        from backtest.live_engine import _StateDB
        with tempfile.TemporaryDirectory() as tmp:
            db = _StateDB(f"{tmp}/state.db")
            db.save("acc1", "MyStrat", "BTCUSDT", "1h", {"count": 42})
            result = db.load("acc1", "MyStrat", "BTCUSDT", "1h")
            assert result == {"count": 42}

    def test_load_returns_none_when_missing(self):
        from backtest.live_engine import _StateDB
        with tempfile.TemporaryDirectory() as tmp:
            db = _StateDB(f"{tmp}/state.db")
            assert db.load("acc1", "MyStrat", "BTCUSDT", "1h") is None

    def test_save_overwrites_previous(self):
        from backtest.live_engine import _StateDB
        with tempfile.TemporaryDirectory() as tmp:
            db = _StateDB(f"{tmp}/state.db")
            db.save("acc1", "MyStrat", "BTCUSDT", "1h", {"v": 1})
            db.save("acc1", "MyStrat", "BTCUSDT", "1h", {"v": 2})
            assert db.load("acc1", "MyStrat", "BTCUSDT", "1h") == {"v": 2}
