from unittest.mock import MagicMock


def _make_client(
    balance_usdt="1000.00",
    position_amt="0",
    entry_price="0",
    unrealized="0",
    mark_price="50000.0",
    step_size="0.001",
):
    client = MagicMock()
    client.exchange_info.return_value = {
        "symbols": [{"symbol": "BTCUSDT", "filters": [
            {"filterType": "LOT_SIZE", "stepSize": step_size}
        ]}]
    }
    client.balance.return_value = [
        {"asset": "USDT", "availableBalance": balance_usdt, "balance": balance_usdt}
    ]
    client.get_position_risk.return_value = [{
        "positionAmt": position_amt,
        "entryPrice": entry_price,
        "unrealizedProfit": unrealized,
    }]
    client.mark_price.return_value = {"markPrice": mark_price}
    return client


class TestLiveExchangeSync:
    def test_balance_after_sync(self):
        from backtest.live_exchange import LiveExchange
        ex = LiveExchange(_make_client(balance_usdt="2500.00"), "BTCUSDT", leverage=10, commission_rate=0.0004)
        ex.sync()
        assert ex.balance == 2500.0

    def test_no_position_when_flat(self):
        from backtest.live_exchange import LiveExchange
        ex = LiveExchange(_make_client(), "BTCUSDT", leverage=10, commission_rate=0.0004)
        ex.sync()
        assert ex.get_position("BTCUSDT") is None

    def test_long_position_parsed(self):
        from backtest.live_exchange import LiveExchange
        ex = LiveExchange(
            _make_client(position_amt="0.1", entry_price="50000", unrealized="200"),
            "BTCUSDT", leverage=10, commission_rate=0.0004,
        )
        ex.sync()
        pos = ex.get_position("BTCUSDT")
        assert pos is not None
        assert pos.side == "long"
        assert pos.entry_price == 50000.0
        assert pos.unrealized_pnl == 200.0
        assert abs(pos.quantity - 5000.0) < 0.01   # 0.1 BTC * 50000 USDT/BTC
        assert abs(pos.margin - 500.0) < 0.01       # 5000 / leverage=10

    def test_short_position_parsed(self):
        from backtest.live_exchange import LiveExchange
        ex = LiveExchange(
            _make_client(position_amt="-0.05", entry_price="48000", unrealized="-100"),
            "BTCUSDT", leverage=10, commission_rate=0.0004,
        )
        ex.sync()
        pos = ex.get_position("BTCUSDT")
        assert pos is not None
        assert pos.side == "short"

    def test_equity_no_position(self):
        from backtest.live_exchange import LiveExchange
        ex = LiveExchange(_make_client(balance_usdt="1000"), "BTCUSDT", leverage=10, commission_rate=0.0004)
        ex.sync()
        assert ex.equity == 1000.0

    def test_equity_with_position(self):
        from backtest.live_exchange import LiveExchange
        # 0.1 BTC long @ 50000, unrealized=200, margin=500
        ex = LiveExchange(
            _make_client(balance_usdt="500", position_amt="0.1",
                         entry_price="50000", unrealized="200"),
            "BTCUSDT", leverage=10, commission_rate=0.0004,
        )
        ex.sync()
        # equity = balance(500) + margin(500) + unrealized(200) = 1200
        assert abs(ex.equity - 1200.0) < 0.01

    def test_get_position_wrong_symbol_returns_none(self):
        from backtest.live_exchange import LiveExchange
        ex = LiveExchange(
            _make_client(position_amt="0.1", entry_price="50000"),
            "BTCUSDT", leverage=10, commission_rate=0.0004,
        )
        ex.sync()
        assert ex.get_position("ETHUSDT") is None


class TestLiveExchangeSubmitOrder:
    def test_dry_run_does_not_call_api(self):
        from backtest.live_exchange import LiveExchange
        client = _make_client(mark_price="50000")
        ex = LiveExchange(client, "BTCUSDT", leverage=10, commission_rate=0.0004, dry_run=True)
        ex.sync()
        order = ex.submit_order("BTCUSDT", "buy", "market", 1000.0)
        client.new_order.assert_not_called()
        assert order.status == "filled"

    def test_dry_run_prints_log(self, capsys):
        from backtest.live_exchange import LiveExchange
        ex = LiveExchange(_make_client(), "BTCUSDT", leverage=10, commission_rate=0.0004, dry_run=True)
        ex.sync()
        ex.submit_order("BTCUSDT", "sell", "market", 500.0)
        captured = capsys.readouterr()
        assert "dry-run" in captured.out.lower()

    def test_market_order_converts_usdt_to_contracts(self):
        from backtest.live_exchange import LiveExchange
        client = _make_client(mark_price="50000")
        client.new_order.return_value = {"orderId": 1, "status": "NEW"}
        ex = LiveExchange(client, "BTCUSDT", leverage=10, commission_rate=0.0004)
        ex.sync()
        ex.submit_order("BTCUSDT", "buy", "market", 1000.0)
        kwargs = client.new_order.call_args[1]
        assert kwargs["symbol"] == "BTCUSDT"
        assert kwargs["side"] == "BUY"
        assert kwargs["type"] == "MARKET"
        # 1000 USDT / 50000 price = 0.020 BTC (lot_step=0.001 → 3 decimal places)
        assert kwargs["quantity"] == 0.020

    def test_order_id_tracked_in_pending(self):
        from backtest.live_exchange import LiveExchange
        client = _make_client()
        client.new_order.return_value = {"orderId": 42, "status": "NEW"}
        ex = LiveExchange(client, "BTCUSDT", leverage=10, commission_rate=0.0004)
        ex.sync()
        ex.submit_order("BTCUSDT", "buy", "market", 500.0)
        assert "42" in ex._pending_order_ids

    def test_sell_side_uppercased(self):
        from backtest.live_exchange import LiveExchange
        client = _make_client()
        client.new_order.return_value = {"orderId": 7, "status": "NEW"}
        ex = LiveExchange(client, "BTCUSDT", leverage=10, commission_rate=0.0004)
        ex.sync()
        ex.submit_order("BTCUSDT", "sell", "market", 500.0)
        assert client.new_order.call_args[1]["side"] == "SELL"


class TestLiveExchangeWaitFills:
    def test_clears_pending_when_filled(self):
        from backtest.live_exchange import LiveExchange
        client = _make_client()
        client.new_order.return_value = {"orderId": 99, "status": "NEW"}
        client.query_order.return_value = {"status": "FILLED"}
        ex = LiveExchange(client, "BTCUSDT", leverage=10, commission_rate=0.0004)
        ex.sync()
        ex.submit_order("BTCUSDT", "buy", "market", 500.0)
        assert "99" in ex._pending_order_ids
        ex.wait_fills(timeout=5.0)
        assert ex._pending_order_ids == []

    def test_warns_when_order_never_fills(self, capsys):
        from backtest.live_exchange import LiveExchange
        client = _make_client()
        client.query_order.return_value = {"status": "NEW"}
        ex = LiveExchange(client, "BTCUSDT", leverage=10, commission_rate=0.0004)
        ex._pending_order_ids = ["77"]
        ex.wait_fills(timeout=0.01)   # expires immediately
        captured = capsys.readouterr()
        assert "77" in captured.out or "WARN" in captured.out

    def test_no_api_call_when_no_pending(self):
        from backtest.live_exchange import LiveExchange
        client = _make_client()
        ex = LiveExchange(client, "BTCUSDT", leverage=10, commission_rate=0.0004)
        ex.wait_fills()
        client.query_order.assert_not_called()
