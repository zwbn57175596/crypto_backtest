# tests/test_live_connector.py
import pytest
from unittest.mock import MagicMock, patch


def _make_binance_client():
    client = MagicMock()
    client.time.return_value = {"serverTime": 1_000_000_000_000}
    return client


class TestBinanceConnectorNormalization:
    def test_normalize_order_filled(self):
        from backtest.live_connector import BinanceConnector
        raw = {
            "orderId": 12345,
            "symbol": "BTCUSDT",
            "status": "FILLED",
            "side": "BUY",
            "type": "MARKET",
            "origQty": "0.020",
            "executedQty": "0.020",
            "avgPrice": "50000.0",
            "price": "0",
            "time": 1_700_000_000_000,
            "updateTime": 1_700_000_001_000,
        }
        result = BinanceConnector._normalize_order(raw)
        assert result["order_id"] == "12345"
        assert result["symbol"] == "BTCUSDT"
        assert result["status"] == "filled"
        assert result["side"] == "buy"
        assert result["type"] == "market"
        assert result["filled_price"] == 50000.0
        assert result["ts"] == 1_700_000_000_000
        assert result["filled_at"] == 1_700_000_001_000

    def test_normalize_order_pending(self):
        from backtest.live_connector import BinanceConnector
        raw = {
            "orderId": 99,
            "symbol": "BTCUSDT",
            "status": "NEW",
            "side": "SELL",
            "type": "LIMIT",
            "origQty": "0.01",
            "executedQty": "0",
            "avgPrice": "0",
            "price": "52000",
            "time": 1_700_000_000_000,
            "updateTime": 1_700_000_000_000,
        }
        result = BinanceConnector._normalize_order(raw)
        assert result["status"] == "new"
        assert result["filled_price"] is None
        assert result["price"] == 52000.0

    def test_normalize_trade(self):
        from backtest.live_connector import BinanceConnector
        raw = {
            "id": 555,
            "orderId": 12345,
            "symbol": "BTCUSDT",
            "buyer": False,
            "price": "50000.0",
            "qty": "0.020",
            "quoteQty": "1000.0",
            "commission": "0.4",
            "realizedPnl": "50.0",
            "time": 1_700_000_001_000,
        }
        result = BinanceConnector._normalize_trade(raw)
        assert result["trade_id"] == "555"
        assert result["order_id"] == "12345"
        assert result["side"] == "sell"
        assert result["price"] == 50000.0
        assert result["qty"] == 0.020
        assert result["notional"] == 1000.0
        assert result["commission"] == 0.4
        assert result["realized_pnl"] == 50.0
        assert result["ts"] == 1_700_000_001_000

    def test_normalize_trade_buyer_side(self):
        from backtest.live_connector import BinanceConnector
        raw = {
            "id": 556, "orderId": 1, "symbol": "BTCUSDT", "buyer": True,
            "price": "50000", "qty": "0.01", "quoteQty": "500",
            "commission": "0.2", "realizedPnl": "0", "time": 1000,
        }
        result = BinanceConnector._normalize_trade(raw)
        assert result["side"] == "buy"


class TestBinanceConnectorInterface:
    def _make_connector(self):
        with patch("backtest.live_connector.UMFutures") as MockUMFutures:
            mock_client = _make_binance_client()
            mock_client.exchange_info.return_value = {
                "symbols": [{"symbol": "BTCUSDT", "filters": [
                    {"filterType": "LOT_SIZE", "stepSize": "0.001"}
                ]}]
            }
            MockUMFutures.return_value = mock_client
            from backtest.live_connector import BinanceConnector
            connector = BinanceConnector(api_key="key", secret="secret", testnet=True)
            connector._client = mock_client
            return connector, mock_client

    def test_exchange_name(self):
        connector, _ = self._make_connector()
        assert connector.exchange_name == "binance"

    def test_exchange_info_returns_symbol_dict(self):
        connector, client = self._make_connector()
        client.exchange_info.return_value = {
            "symbols": [{"symbol": "BTCUSDT", "filters": [{"filterType": "LOT_SIZE", "stepSize": "0.001"}]}]
        }
        result = connector.exchange_info("BTCUSDT")
        assert result["symbol"] == "BTCUSDT"

    def test_fetch_balance(self):
        connector, client = self._make_connector()
        client.balance.return_value = [
            {"asset": "BNB", "availableBalance": "1.0"},
            {"asset": "USDT", "availableBalance": "1234.56"},
        ]
        assert connector.fetch_balance() == 1234.56

    def test_fetch_position_flat_returns_none(self):
        connector, client = self._make_connector()
        client.get_position_risk.return_value = [{"positionAmt": "0", "entryPrice": "0", "unrealizedProfit": "0"}]
        assert connector.fetch_position("BTCUSDT") is None

    def test_fetch_position_long(self):
        connector, client = self._make_connector()
        client.get_position_risk.return_value = [
            {"positionAmt": "0.1", "entryPrice": "50000", "unrealizedProfit": "200"}
        ]
        pos = connector.fetch_position("BTCUSDT")
        assert pos is not None
        assert pos["side"] == "long"
        assert pos["qty"] == 0.1
        assert pos["entry_price"] == 50000.0

    def test_fetch_orders_passes_since_ms(self):
        connector, client = self._make_connector()
        client.get_all_orders.return_value = []
        connector.fetch_orders("BTCUSDT", since_ms=1_700_000_000_000)
        client.get_all_orders.assert_called_once_with(symbol="BTCUSDT", startTime=1_700_000_000_000)

    def test_fetch_trades_no_since_ms(self):
        connector, client = self._make_connector()
        client.get_account_trades.return_value = []
        connector.fetch_trades("BTCUSDT")
        client.get_account_trades.assert_called_once_with(symbol="BTCUSDT")
