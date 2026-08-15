from backtest.models import Bar, Order
from backtest.exchange import SimExchange


def _bar(symbol, ts, o, h, l, c, interval="1h"):
    return Bar(symbol=symbol, timestamp=ts, open=o, high=h, low=l, close=c,
               volume=1.0, interval=interval)


def test_order_has_expire_ts_default_zero():
    order = Order(id="x", symbol="BTCUSDT", side="buy", type="limit", quantity=100.0)
    assert order.expire_ts == 0
