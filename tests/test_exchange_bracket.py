from backtest.models import Bar, Order
from backtest.exchange import SimExchange


def _bar(symbol, ts, o, h, l, c, interval="1h"):
    return Bar(symbol=symbol, timestamp=ts, open=o, high=h, low=l, close=c,
               volume=1.0, interval=interval)


def test_order_has_expire_ts_default_zero():
    order = Order(id="x", symbol="BTCUSDT", side="buy", type="limit", quantity=100.0)
    assert order.expire_ts == 0


def test_limit_order_uses_maker_rate():
    ex = SimExchange(balance=1000, leverage=10, commission_rate=0.0004,
                     funding_rate=0.0, maintenance_margin=0.005,
                     maker_commission_rate=0.0002, taker_commission_rate=0.0005)
    ex.submit_order("BTCUSDT", "buy", "limit", 100.0, price=100.0)
    ex.on_new_bar(_bar("BTCUSDT", 1000, 101, 101, 99, 100))
    trades = ex.get_trades()
    assert trades[0].commission == 100.0 * 0.0002


def test_market_order_uses_taker_rate():
    ex = SimExchange(balance=1000, leverage=10, commission_rate=0.0004,
                     funding_rate=0.0, maintenance_margin=0.005,
                     maker_commission_rate=0.0002, taker_commission_rate=0.0005)
    ex.submit_order("BTCUSDT", "buy", "market", 100.0)
    ex.on_new_bar(_bar("BTCUSDT", 1000, 100, 101, 99, 100))
    assert ex.get_trades()[0].commission == 100.0 * 0.0005


def test_rates_fall_back_to_commission_rate():
    ex = SimExchange(balance=1000, leverage=10, commission_rate=0.0004,
                     funding_rate=0.0, maintenance_margin=0.005)
    assert ex.maker_commission_rate == 0.0004
    assert ex.taker_commission_rate == 0.0004
