from backtest.models import Bar, Order, Position, Trade


def test_bar_creation(sample_bar):
    assert sample_bar.symbol == "BTCUSDT"
    assert sample_bar.close == 42300.0
    assert sample_bar.interval == "1h"


def test_order_defaults():
    order = Order(
        id="o1",
        symbol="BTCUSDT",
        side="buy",
        type="market",
        quantity=1000.0,
    )
    assert order.status == "pending"
    assert order.price is None
    assert order.filled_price == 0.0
    assert order.commission == 0.0


def test_position_unrealized_pnl():
    pos = Position(
        symbol="BTCUSDT",
        side="long",
        quantity=1000.0,
        entry_price=42000.0,
        leverage=10,
    )
    assert pos.unrealized_pnl == 0.0
    assert pos.margin == 0.0


def test_trade_creation():
    trade = Trade(
        id="t1",
        order_id="o1",
        symbol="BTCUSDT",
        side="buy",
        price=42000.0,
        quantity=1000.0,
        pnl=0.0,
        commission=0.4,
        timestamp=1704067200000,
    )
    assert trade.commission == 0.4
