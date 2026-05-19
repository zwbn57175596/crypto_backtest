# tests/test_okx_connector.py
import pytest
from unittest.mock import MagicMock, patch


def _make_connector():
    with patch("okx.PublicData.PublicAPI"), \
         patch("okx.MarketData.MarketAPI"), \
         patch("okx.Trade.TradeAPI"), \
         patch("okx.Account.AccountAPI"):
        from backtest.live_connector import OKXConnector
        c = OKXConnector(api_key="k", secret="s", passphrase="p", demo=True)
        c._ct_val = 0.01
        c._lot_sz = 1.0
        return c


class TestOKXSymbolMapping:
    def test_btcusdt_to_inst_id(self):
        from backtest.live_connector import OKXConnector
        assert OKXConnector._inst_id("BTCUSDT") == "BTC-USDT-SWAP"

    def test_ethusdt_to_inst_id(self):
        from backtest.live_connector import OKXConnector
        assert OKXConnector._inst_id("ETHUSDT") == "ETH-USDT-SWAP"

    def test_unsupported_symbol_raises(self):
        from backtest.live_connector import OKXConnector
        with pytest.raises(ValueError):
            OKXConnector._inst_id("BTCBTC")


class TestOKXIntervalMapping:
    def test_1h_maps_to_1H(self):
        from backtest.live_connector import OKXConnector
        assert OKXConnector._okx_bar("1h") == "1H"

    def test_1m_unchanged(self):
        from backtest.live_connector import OKXConnector
        assert OKXConnector._okx_bar("1m") == "1m"

    def test_unsupported_interval_raises(self):
        from backtest.live_connector import OKXConnector
        with pytest.raises(ValueError):
            OKXConnector._okx_bar("2d")


class TestOKXExchangeName:
    def test_exchange_name(self):
        c = _make_connector()
        assert c.exchange_name == "okx"


class TestOKXKlines:
    def test_reverses_to_oldest_first(self):
        c = _make_connector()
        # OKX returns newest first: ts=200, ts=100
        c._market.get_candlesticks.return_value = {
            "data": [
                ["200", "2.0", "2.1", "1.9", "2.0", "100"],
                ["100", "1.0", "1.1", "0.9", "1.0", "50"],
            ]
        }
        bars = c.klines("BTCUSDT", "1h", limit=2)
        assert bars[0][0] == 100  # oldest first
        assert bars[1][0] == 200

    def test_maps_starttime_to_after(self):
        c = _make_connector()
        c._market.get_candlesticks.return_value = {"data": []}
        c.klines("BTCUSDT", "1h", startTime=1000, endTime=2000, limit=10)
        call_kwargs = c._market.get_candlesticks.call_args[1]
        assert call_kwargs["after"] == "999"   # startTime - 1
        assert call_kwargs["before"] == "2001"  # endTime + 1


class TestOKXBalance:
    def test_returns_usdt_avail_eq(self):
        c = _make_connector()
        c._account.get_account_balance.return_value = {
            "data": [{"details": [
                {"ccy": "BTC", "availEq": "0.1"},
                {"ccy": "USDT", "availEq": "1234.5"},
            ]}]
        }
        assert c.fetch_balance() == 1234.5

    def test_falls_back_to_cashbal(self):
        c = _make_connector()
        c._account.get_account_balance.return_value = {
            "data": [{"details": [{"ccy": "USDT", "availEq": None, "cashBal": "999.0"}]}]
        }
        assert c.fetch_balance() == 999.0


class TestOKXPosition:
    def test_long_position(self):
        c = _make_connector()
        c._account.get_positions.return_value = {
            "data": [{"pos": "2", "avgPx": "50000", "upl": "10.5"}]
        }
        pos = c.fetch_position("BTCUSDT")
        assert pos["side"] == "long"
        assert pos["qty"] == pytest.approx(0.02)  # 2 contracts * 0.01 ctVal
        assert pos["entry_price"] == 50000.0

    def test_short_position(self):
        c = _make_connector()
        c._account.get_positions.return_value = {
            "data": [{"pos": "-3", "avgPx": "60000", "upl": "-5.0"}]
        }
        pos = c.fetch_position("BTCUSDT")
        assert pos["side"] == "short"
        assert pos["qty"] == pytest.approx(0.03)

    def test_flat_returns_none(self):
        c = _make_connector()
        c._account.get_positions.return_value = {"data": [{"pos": "0"}]}
        assert c.fetch_position("BTCUSDT") is None


class TestOKXSubmitOrder:
    def test_converts_base_to_contracts(self):
        c = _make_connector()
        c._trade.place_order.return_value = {"data": [{"ordId": "12345"}]}
        c.submit_order("BTCUSDT", "buy", "market", 0.05)  # 0.05 BTC → 5 contracts
        call_kwargs = c._trade.place_order.call_args[1]
        assert call_kwargs["sz"] == "5"  # 0.05 / 0.01 = 5

    def test_passes_client_order_id(self):
        c = _make_connector()
        c._trade.place_order.return_value = {"data": [{"ordId": "99"}]}
        c.submit_order("BTCUSDT", "buy", "market", 0.01, client_order_id="abc123")
        call_kwargs = c._trade.place_order.call_args[1]
        assert call_kwargs["clOrdId"] == "abc123"

    def test_limit_order_includes_price(self):
        c = _make_connector()
        c._trade.place_order.return_value = {"data": [{"ordId": "77"}]}
        c.submit_order("BTCUSDT", "buy", "limit", 0.01, price=50000.0)
        call_kwargs = c._trade.place_order.call_args[1]
        assert call_kwargs["ordType"] == "limit"
        assert call_kwargs["px"] == "50000.0"

    def test_minimum_one_lot(self):
        c = _make_connector()
        c._trade.place_order.return_value = {"data": [{"ordId": "1"}]}
        c.submit_order("BTCUSDT", "buy", "market", 0.001)  # < 1 contract → floor to 1
        call_kwargs = c._trade.place_order.call_args[1]
        assert call_kwargs["sz"] == "1"


class TestOKXNormalizeOrder:
    def test_filled_status(self):
        c = _make_connector()
        o = {
            "ordId": "111", "clOrdId": "abc", "instId": "BTC-USDT-SWAP",
            "side": "buy", "ordType": "market", "sz": "2", "fillSz": "2",
            "avgPx": "50000", "px": None, "fee": "-1.0",
            "state": "filled", "cTime": "1000", "uTime": "2000",
        }
        result = c._normalize_order(o)
        assert result["status"] == "filled"
        assert result["quantity"] == pytest.approx(0.02)
        assert result["filled_qty"] == pytest.approx(0.02)
        assert result["commission"] == -1.0
        assert result["filled_at"] == 2000
