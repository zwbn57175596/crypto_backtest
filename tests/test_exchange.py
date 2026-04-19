import pytest
from backtest.exchange import SimExchange
from backtest.models import Bar


@pytest.fixture
def exchange():
    return SimExchange(
        balance=10000.0,
        leverage=10,
        commission_rate=0.0004,
        funding_rate=0.0001,
        maintenance_margin=0.005,
    )


@pytest.fixture
def bar1():
    return Bar(
        symbol="BTCUSDT", timestamp=1704067200000,
        open=42000.0, high=42500.0, low=41800.0, close=42300.0,
        volume=1500.0, interval="1h",
    )


@pytest.fixture
def bar2():
    return Bar(
        symbol="BTCUSDT", timestamp=1704070800000,
        open=42300.0, high=42800.0, low=42100.0, close=42600.0,
        volume=1200.0, interval="1h",
    )


class TestMarketOrder:
    def test_buy_market_opens_long(self, exchange, bar1):
        order = exchange.submit_order("BTCUSDT", "buy", "market", 1000.0)
        exchange.on_new_bar(bar1)
        assert order.status == "filled"
        assert order.filled_price == 42000.0
        pos = exchange.get_position("BTCUSDT")
        assert pos is not None
        assert pos.side == "long"
        assert pos.quantity == 1000.0

    def test_sell_market_opens_short(self, exchange, bar1):
        order = exchange.submit_order("BTCUSDT", "sell", "market", 1000.0)
        exchange.on_new_bar(bar1)
        assert order.status == "filled"
        pos = exchange.get_position("BTCUSDT")
        assert pos.side == "short"

    def test_close_long_position(self, exchange, bar1, bar2):
        exchange.submit_order("BTCUSDT", "buy", "market", 1000.0)
        exchange.on_new_bar(bar1)
        exchange.submit_order("BTCUSDT", "sell", "market", 1000.0)
        exchange.on_new_bar(bar2)
        assert exchange.get_position("BTCUSDT") is None


class TestLimitOrder:
    def test_limit_buy_fills_when_low_reaches(self, exchange, bar1):
        order = exchange.submit_order("BTCUSDT", "buy", "limit", 1000.0, price=41900.0)
        exchange.on_new_bar(bar1)
        assert order.status == "filled"
        assert order.filled_price == 41900.0

    def test_limit_buy_not_filled_when_low_above(self, exchange, bar1):
        order = exchange.submit_order("BTCUSDT", "buy", "limit", 1000.0, price=41700.0)
        exchange.on_new_bar(bar1)
        assert order.status == "pending"

    def test_limit_sell_fills_when_high_reaches(self, exchange, bar1):
        order = exchange.submit_order("BTCUSDT", "sell", "limit", 1000.0, price=42400.0)
        exchange.on_new_bar(bar1)
        assert order.status == "filled"
        assert order.filled_price == 42400.0


class TestCommission:
    def test_commission_deducted(self, exchange, bar1):
        exchange.submit_order("BTCUSDT", "buy", "market", 1000.0)
        exchange.on_new_bar(bar1)
        assert exchange.balance < 10000.0
        trades = exchange.get_trades()
        assert trades[0].commission == pytest.approx(0.4)


class TestUnrealizedPnl:
    def test_long_unrealized_pnl(self, exchange, bar1, bar2):
        exchange.submit_order("BTCUSDT", "buy", "market", 1000.0)
        exchange.on_new_bar(bar1)
        exchange.on_new_bar(bar2)
        pos = exchange.get_position("BTCUSDT")
        expected = 1000.0 * (42600.0 - 42000.0) / 42000.0
        assert pos.unrealized_pnl == pytest.approx(expected, rel=1e-4)

    def test_short_unrealized_pnl(self, exchange, bar1, bar2):
        exchange.submit_order("BTCUSDT", "sell", "market", 1000.0)
        exchange.on_new_bar(bar1)
        exchange.on_new_bar(bar2)
        pos = exchange.get_position("BTCUSDT")
        expected = 1000.0 * (42000.0 - 42600.0) / 42000.0
        assert pos.unrealized_pnl == pytest.approx(expected, rel=1e-4)


class TestFundingRate:
    def test_funding_settled_at_8h_boundary(self, exchange):
        bar_pre = Bar(
            symbol="BTCUSDT", timestamp=1704092400000,
            open=42000.0, high=42100.0, low=41900.0, close=42000.0,
            volume=500.0, interval="1h",
        )
        bar_funding = Bar(
            symbol="BTCUSDT", timestamp=1704096000000,
            open=42000.0, high=42100.0, low=41900.0, close=42050.0,
            volume=500.0, interval="1h",
        )
        exchange.submit_order("BTCUSDT", "buy", "market", 1000.0)
        exchange.on_new_bar(bar_pre)
        balance_before = exchange.balance
        exchange.on_new_bar(bar_funding)
        assert exchange.balance < balance_before


class TestLiquidation:
    def test_long_liquidated_on_large_drop(self):
        exch = SimExchange(
            balance=100.0, leverage=100,
            commission_rate=0.0004, funding_rate=0.0001,
            maintenance_margin=0.005,
        )
        exch.submit_order("BTCUSDT", "buy", "market", 5000.0)
        bar_open = Bar(
            symbol="BTCUSDT", timestamp=1704067200000,
            open=42000.0, high=42000.0, low=41999.0, close=42000.0,
            volume=100.0, interval="1h",
        )
        exch.on_new_bar(bar_open)
        bar_crash = Bar(
            symbol="BTCUSDT", timestamp=1704070800000,
            open=42000.0, high=42000.0, low=40000.0, close=40500.0,
            volume=100.0, interval="1h",
        )
        exch.on_new_bar(bar_crash)
        assert exch.get_position("BTCUSDT") is None
