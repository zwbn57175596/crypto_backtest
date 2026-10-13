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


def test_limit_order_expires_when_unfilled():
    ex = SimExchange(1000, 10, 0.0004, 0.0, 0.005)
    ex.submit_order("BTCUSDT", "buy", "limit", 100.0, price=90.0, expire_ts=2000)
    ex.on_new_bar(_bar("BTCUSDT", 1000, 100, 101, 95, 100))  # low 95 > 90: no fill, not expired
    assert len(ex._pending_orders) == 1
    ex.on_new_bar(_bar("BTCUSDT", 2000, 100, 101, 95, 100))  # ts >= expire, still no fill: canceled
    assert len(ex._pending_orders) == 0
    assert ex.get_position("BTCUSDT") is None


def test_limit_order_fills_on_expiry_bar_before_cancel():
    ex = SimExchange(1000, 10, 0.0004, 0.0, 0.005)
    ex.submit_order("BTCUSDT", "buy", "limit", 100.0, price=90.0, expire_ts=2000)
    ex.on_new_bar(_bar("BTCUSDT", 2000, 100, 101, 89, 100))  # low 89 <= 90: fills even on expiry bar
    assert ex.get_position("BTCUSDT") is not None
    assert len(ex._pending_orders) == 0


def test_expire_ts_zero_never_expires():
    ex = SimExchange(1000, 10, 0.0004, 0.0, 0.005)
    ex.submit_order("BTCUSDT", "buy", "limit", 100.0, price=90.0)  # expire_ts default 0
    ex.on_new_bar(_bar("BTCUSDT", 9_999_999, 100, 101, 95, 100))
    assert len(ex._pending_orders) == 1  # still alive


def _open_long_at_100(ex):
    ex.submit_order("BTCUSDT", "buy", "market", 100.0)
    ex.on_new_bar(_bar("BTCUSDT", 1000, 100, 100, 100, 100))
    assert ex.get_position("BTCUSDT").entry_price == 100.0


def test_bracket_take_profit_long_uses_maker():
    ex = SimExchange(1000, 10, 0.0, 0.0, 0.005,
                     maker_commission_rate=0.0002, taker_commission_rate=0.0005)
    _open_long_at_100(ex)
    ex.set_bracket("BTCUSDT", tp_price=105.0, sl_price=95.0, expire_ts=0)
    ex.on_new_bar(_bar("BTCUSDT", 2000, 100, 106, 99, 104))  # high 106 >= 105 → TP
    assert ex.get_position("BTCUSDT") is None
    t = ex.get_trades()[-1]
    assert t.order_id == "bracket-tp"
    assert t.price == 105.0
    assert abs(t.pnl - 5.0) < 1e-9              # 100*(105-100)/100
    assert abs(t.commission - 100.0 * 0.0002) < 1e-12


def test_bracket_stop_loss_long_uses_taker_and_sl_wins_ties():
    ex = SimExchange(1000, 10, 0.0, 0.0, 0.005,
                     maker_commission_rate=0.0002, taker_commission_rate=0.0005)
    _open_long_at_100(ex)
    ex.set_bracket("BTCUSDT", tp_price=105.0, sl_price=95.0, expire_ts=0)
    ex.on_new_bar(_bar("BTCUSDT", 2000, 100, 106, 94, 96))  # both hit → SL first
    t = ex.get_trades()[-1]
    assert t.order_id == "bracket-sl"
    assert t.price == 95.0
    assert abs(t.pnl - (-5.0)) < 1e-9
    assert abs(t.commission - 100.0 * 0.0005) < 1e-12


def test_bracket_time_stop_closes_at_close_price_taker():
    ex = SimExchange(1000, 10, 0.0, 0.0, 0.005,
                     maker_commission_rate=0.0002, taker_commission_rate=0.0005)
    _open_long_at_100(ex)
    ex.set_bracket("BTCUSDT", tp_price=105.0, sl_price=95.0, expire_ts=2000)
    ex.on_new_bar(_bar("BTCUSDT", 2000, 100, 102, 99, 101))  # neither TP/SL, ts >= expire
    t = ex.get_trades()[-1]
    assert t.order_id == "bracket-time"
    assert t.price == 101.0
    assert abs(t.pnl - 1.0) < 1e-9
    assert abs(t.commission - 100.0 * 0.0005) < 1e-12


def test_bracket_short_take_profit():
    ex = SimExchange(1000, 10, 0.0, 0.0, 0.005,
                     maker_commission_rate=0.0002, taker_commission_rate=0.0005)
    ex.submit_order("BTCUSDT", "sell", "market", 100.0)
    ex.on_new_bar(_bar("BTCUSDT", 1000, 100, 100, 100, 100))
    assert ex.get_position("BTCUSDT").side == "short"
    ex.set_bracket("BTCUSDT", tp_price=95.0, sl_price=105.0, expire_ts=0)  # short: tp below, sl above
    ex.on_new_bar(_bar("BTCUSDT", 2000, 100, 101, 94, 96))  # low 94 <= 95 → TP
    t = ex.get_trades()[-1]
    assert t.order_id == "bracket-tp"
    assert t.price == 95.0
    assert abs(t.pnl - 5.0) < 1e-9  # short: 100*(100-95)/100


def test_liquidation_pops_stale_bracket():
    ex = SimExchange(1000, 10, 0.0, 0.0, 0.005)  # isolated, maintenance 0.005
    ex.submit_order("BTCUSDT", "buy", "market", 1000.0)
    ex.on_new_bar(_bar("BTCUSDT", 1000, 100, 100, 100, 100))  # long @100, margin 100
    # SL far away (won't trigger), so liquidation is what removes the position
    ex.set_bracket("BTCUSDT", tp_price=200.0, sl_price=50.0, expire_ts=0)
    assert "BTCUSDT" in ex._brackets
    # ~11% adverse move: sl(50)/tp(200) untouched, but 10x isolated liquidates
    ex.on_new_bar(_bar("BTCUSDT", 2000, 100, 100, 89, 89))
    assert ex.get_position("BTCUSDT") is None          # liquidated
    assert "BTCUSDT" not in ex._brackets                # <-- the fix: no stale bracket
