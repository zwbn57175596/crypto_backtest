from backtest.mean_reversion import (
    MRParams, bar_direction, update_streak, fade_side,
    entry_price, bracket_prices, position_notional,
)


def test_mrparams_defaults_present():
    p = MRParams()
    assert p.threshold >= 2
    assert 0 < p.tp_pct < 0.1 and 0 < p.sl_pct < 0.1
    assert p.time_stop_bars >= 1
    assert 0 < p.risk_pct < 0.1
    assert p.entry_offset_pct >= 0
    assert p.entry_ttl_bars >= 1


def test_bar_direction():
    assert bar_direction(100, 101) == 1
    assert bar_direction(101, 100) == -1
    assert bar_direction(100, 100) == 0


def test_update_streak_extends_and_resets():
    assert update_streak(0, 0, 1) == (1, 1)      # first up
    assert update_streak(1, 1, 1) == (2, 1)      # extend up
    assert update_streak(3, 1, -1) == (1, -1)    # flip to down
    assert update_streak(3, 1, 0) == (3, 1)      # doji: unchanged


def test_fade_side():
    assert fade_side(1) == "sell"   # up-streak → short
    assert fade_side(-1) == "buy"   # down-streak → long


def test_entry_price_offsets_pullback_side():
    assert entry_price(100.0, 1, 0.001) == 100.0 * 1.001   # sell above
    assert entry_price(100.0, -1, 0.001) == 100.0 * 0.999  # buy below


def test_bracket_prices_long_and_short():
    tp, sl = bracket_prices(100.0, "long", 0.01, 0.005)
    assert tp == 101.0 and sl == 99.5
    tp, sl = bracket_prices(100.0, "short", 0.01, 0.005)
    assert tp == 99.0 and sl == 100.5


def test_position_notional():
    assert position_notional(10000.0, 0.005, 0.01) == 5000.0
    assert position_notional(10000.0, 0.005, 0.0) == 0.0
