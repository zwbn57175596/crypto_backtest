# tests/test_live_exchange.py
import pytest
from unittest.mock import MagicMock


def _make_connector(
    balance=1000.0,
    position=None,
    mark_price=50000.0,
    step_size="0.001",
):
    connector = MagicMock()
    connector.exchange_name = "binance"
    connector.exchange_info.return_value = {
        "symbol": "BTCUSDT",
        "filters": [{"filterType": "LOT_SIZE", "stepSize": step_size}],
    }
    connector.fetch_balance.return_value = balance
    connector.fetch_position.return_value = position
    connector.fetch_mark_price.return_value = mark_price
    return connector


def _make_history_db():
    return MagicMock()


def _make_ex(connector=None, history_db=None, **kwargs):
    from backtest.live_exchange import LiveExchange
    return LiveExchange(
        connector=connector or _make_connector(),
        history_db=history_db or _make_history_db(),
        account_id="testacc",
        symbol="BTCUSDT",
        leverage=kwargs.get("leverage", 10),
        commission_rate=kwargs.get("commission_rate", 0.0004),
        dry_run=kwargs.get("dry_run", False),
    )


class TestLiveExchangeSync:
    def test_balance_after_sync(self):
        ex = _make_ex(connector=_make_connector(balance=2500.0))
        ex.sync()
        assert ex.balance == 2500.0

    def test_no_position_when_flat(self):
        ex = _make_ex(connector=_make_connector(position=None))
        ex.sync()
        assert ex.get_position("BTCUSDT") is None

    def test_long_position_parsed(self):
        pos_data = {"side": "long", "qty": 0.1, "entry_price": 50000.0, "unrealized_pnl": 200.0}
        ex = _make_ex(connector=_make_connector(position=pos_data))
        ex.sync()
        pos = ex.get_position("BTCUSDT")
        assert pos is not None
        assert pos.side == "long"
        assert pos.entry_price == 50000.0
        assert pos.unrealized_pnl == 200.0
        assert abs(pos.quantity - 5000.0) < 0.01   # 0.1 * 50000
        assert abs(pos.margin - 500.0) < 0.01       # 5000 / 10

    def test_short_position_parsed(self):
        pos_data = {"side": "short", "qty": 0.05, "entry_price": 48000.0, "unrealized_pnl": -100.0}
        ex = _make_ex(connector=_make_connector(position=pos_data))
        ex.sync()
        pos = ex.get_position("BTCUSDT")
        assert pos is not None
        assert pos.side == "short"

    def test_equity_no_position(self):
        ex = _make_ex(connector=_make_connector(balance=1000.0, position=None))
        ex.sync()
        assert ex.equity == 1000.0

    def test_equity_with_position(self):
        pos_data = {"side": "long", "qty": 0.1, "entry_price": 50000.0, "unrealized_pnl": 200.0}
        ex = _make_ex(connector=_make_connector(balance=500.0, position=pos_data))
        ex.sync()
        # equity = balance(500) + margin(500) + unrealized(200) = 1200
        assert abs(ex.equity - 1200.0) < 0.01

    def test_get_position_wrong_symbol_returns_none(self):
        pos_data = {"side": "long", "qty": 0.1, "entry_price": 50000.0, "unrealized_pnl": 0.0}
        ex = _make_ex(connector=_make_connector(position=pos_data))
        ex.sync()
        assert ex.get_position("ETHUSDT") is None

    def test_sync_records_position_in_history(self):
        history_db = _make_history_db()
        ex = _make_ex(connector=_make_connector(balance=1000.0), history_db=history_db)
        ex.sync()
        history_db.record_position.assert_called_once()


class TestLiveExchangeSubmitOrder:
    def test_dry_run_does_not_call_api(self):
        connector = _make_connector(mark_price=50000.0)
        history_db = _make_history_db()
        ex = _make_ex(connector=connector, history_db=history_db, dry_run=True)
        ex.sync()
        order = ex.submit_order("BTCUSDT", "buy", "market", 1000.0)
        connector.submit_order.assert_not_called()
        assert order.status == "filled"

    def test_dry_run_records_order_in_history(self):
        history_db = _make_history_db()
        ex = _make_ex(connector=_make_connector(), history_db=history_db, dry_run=True)
        ex.sync()
        ex.submit_order("BTCUSDT", "buy", "market", 500.0)
        history_db.upsert_order.assert_called()

    def test_dry_run_prints_log(self, capsys):
        ex = _make_ex(dry_run=True)
        ex.sync()
        ex.submit_order("BTCUSDT", "sell", "market", 500.0)
        captured = capsys.readouterr()
        assert "dry-run" in captured.out.lower()

    def test_market_order_converts_usdt_to_contracts(self):
        connector = _make_connector(mark_price=50000.0)
        connector.submit_order.return_value = {"orderId": 1}
        ex = _make_ex(connector=connector)
        ex.sync()
        ex.submit_order("BTCUSDT", "buy", "market", 1000.0)
        call_kwargs = connector.submit_order.call_args
        # quantity arg is contracts: 1000/50000 = 0.02
        assert call_kwargs[0][3] == pytest.approx(0.020, abs=0.001)

    def test_order_id_tracked_in_pending(self):
        connector = _make_connector()
        connector.submit_order.return_value = {"orderId": 42}
        ex = _make_ex(connector=connector)
        ex.sync()
        ex.submit_order("BTCUSDT", "buy", "market", 500.0)
        assert "42" in ex._pending_order_ids

    def test_sell_side_passed_correctly(self):
        connector = _make_connector()
        connector.submit_order.return_value = {"orderId": 7}
        ex = _make_ex(connector=connector)
        ex.sync()
        ex.submit_order("BTCUSDT", "sell", "market", 500.0)
        assert connector.submit_order.call_args[0][1] == "sell"


class TestLiveExchangeWaitFills:
    def test_clears_pending_when_filled(self):
        connector = _make_connector()
        connector.submit_order.return_value = {"orderId": 99}
        connector.query_order.return_value = {
            "order_id": "99", "symbol": "BTCUSDT", "side": "buy", "type": "market",
            "quantity": 500.0, "price": None, "status": "filled",
            "filled_price": 50000.0, "filled_qty": 0.01, "commission": 0.2,
            "ts": 5000, "filled_at": 5100,
        }
        ex = _make_ex(connector=connector)
        ex.sync()
        ex.submit_order("BTCUSDT", "buy", "market", 500.0)
        assert "99" in ex._pending_order_ids
        ex.wait_fills(timeout=5.0)
        assert ex._pending_order_ids == []

    def test_fill_upserted_to_history(self):
        connector = _make_connector()
        connector.submit_order.return_value = {"orderId": 99}
        connector.query_order.return_value = {
            "order_id": "99", "symbol": "BTCUSDT", "side": "buy", "type": "market",
            "quantity": 500.0, "price": None, "status": "filled",
            "filled_price": 50000.0, "filled_qty": 0.01, "commission": 0.2,
            "ts": 5000, "filled_at": 5100,
        }
        history_db = _make_history_db()
        ex = _make_ex(connector=connector, history_db=history_db)
        ex.sync()
        ex.submit_order("BTCUSDT", "buy", "market", 500.0)
        ex.wait_fills(timeout=5.0)
        # upsert_order called: once on submit, once on fill
        assert history_db.upsert_order.call_count == 2

    def test_warns_when_order_never_fills(self, capsys):
        connector = _make_connector()
        connector.query_order.return_value = {
            "order_id": "77", "symbol": "BTCUSDT", "side": "buy", "type": "market",
            "quantity": 500.0, "price": None, "status": "new",
            "filled_price": None, "filled_qty": None, "commission": None,
            "ts": 1000, "filled_at": None,
        }
        ex = _make_ex(connector=connector)
        ex._pending_order_ids = ["77"]
        ex.wait_fills(timeout=0.01)
        captured = capsys.readouterr()
        assert "77" in captured.out or "WARN" in captured.out

    def test_no_api_call_when_no_pending(self):
        connector = _make_connector()
        ex = _make_ex(connector=connector)
        ex.wait_fills()
        connector.query_order.assert_not_called()
