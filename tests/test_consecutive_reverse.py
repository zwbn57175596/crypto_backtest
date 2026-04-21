"""Tests for ConsecutiveReverseStrategy."""
import sys
import os
from pathlib import Path

import pytest
from backtest.models import Bar
from backtest.exchange import SimExchange

# Add strategies to path so we can import ConsecutiveReverseStrategy
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


def make_bar(open_: float, close: float, ts: int = 1000) -> Bar:
    """Helper to create a Bar with given open/close."""
    return Bar(
        symbol="BTCUSDT", interval="1h", timestamp=ts,
        open=open_, high=max(open_, close) + 10,
        low=min(open_, close) - 10, close=close, volume=100.0,
    )


def create_strategy(params: dict | None = None, leverage: int = 50):
    """Create strategy with a real SimExchange."""
    from strategies.consecutive_reverse import ConsecutiveReverseStrategy

    exchange = SimExchange(
        balance=1000.0, leverage=leverage,
        commission_rate=0.0005, funding_rate=0.0001,
        maintenance_margin=0.005,
    )
    strategy = ConsecutiveReverseStrategy(exchange=exchange, symbol="BTCUSDT")
    if params:
        for k, v in params.items():
            setattr(strategy, k, v)
    strategy.on_init()
    return strategy, exchange


class TestDirectionDetection:
    def test_bullish_candle(self):
        strategy, _ = create_strategy()
        bar = make_bar(open_=100, close=105)
        assert strategy._get_direction(bar) == 1

    def test_bearish_candle(self):
        strategy, _ = create_strategy()
        bar = make_bar(open_=105, close=100)
        assert strategy._get_direction(bar) == -1

    def test_doji_candle(self):
        strategy, _ = create_strategy()
        bar = make_bar(open_=100, close=100)
        assert strategy._get_direction(bar) == 0


class TestStreakTracking:
    def test_consecutive_up_streak(self):
        strategy, _ = create_strategy()
        for i in range(4):
            strategy._update_streak(1)
        assert strategy._consecutive_count == 4
        assert strategy._streak_direction == 1

    def test_streak_resets_on_direction_change(self):
        strategy, _ = create_strategy()
        strategy._update_streak(1)
        strategy._update_streak(1)
        strategy._update_streak(1)
        strategy._update_streak(-1)
        assert strategy._consecutive_count == 1
        assert strategy._streak_direction == -1


class TestPositionSizing:
    def test_below_threshold_returns_zero(self):
        strategy, _ = create_strategy()
        strategy._consecutive_count = 4
        assert strategy._calc_quantity() == 0

    def test_at_threshold_returns_base_size(self):
        strategy, _ = create_strategy()
        strategy._consecutive_count = 5
        # base = 1000 * 0.01 = 10, multiplier = 1.1^0 = 1, qty = 10 * 50 = 500
        assert strategy._calc_quantity() == pytest.approx(500.0)

    def test_above_threshold_applies_multiplier(self):
        strategy, _ = create_strategy()
        strategy._consecutive_count = 7
        # base = 10, n = 3, multiplier = 1.1^2 = 1.21, qty = 10 * 1.21 * 50 = 605
        assert strategy._calc_quantity() == pytest.approx(605.0)


class TestOpenPosition:
    def test_opens_short_after_consecutive_up(self):
        """5 consecutive up candles should trigger a short position."""
        # Use smaller leverage to avoid liquidation from small price moves
        strategy, exchange = create_strategy(leverage=10)
        for i in range(5):
            bar = make_bar(open_=100 + i, close=105 + i, ts=1000 + i * 3600000)
            exchange.on_new_bar(bar)
            strategy._push_bar(bar)

        # Sell order pending, fills on 6th bar at open price, then price stays flat
        bar6 = make_bar(open_=105, close=105, ts=1000 + 5 * 3600000)  # fill at 105
        exchange.on_new_bar(bar6)

        pos = exchange.get_position("BTCUSDT")
        assert pos is not None
        assert pos.side == "short"

    def test_opens_long_after_consecutive_down(self):
        """5 consecutive down candles should trigger a long position."""
        # Use smaller leverage to avoid liquidation from small price moves
        strategy, exchange = create_strategy(leverage=10)
        for i in range(5):
            bar = make_bar(open_=200 - i, close=195 - i, ts=1000 + i * 3600000)
            exchange.on_new_bar(bar)
            strategy._push_bar(bar)

        # Buy order pending, fills on 6th bar at open price, then price stays flat
        bar6 = make_bar(open_=195, close=195, ts=1000 + 5 * 3600000)  # fill at 195
        exchange.on_new_bar(bar6)

        pos = exchange.get_position("BTCUSDT")
        assert pos is not None
        assert pos.side == "long"

    def test_no_open_below_threshold(self):
        """4 consecutive candles should NOT trigger opening."""
        strategy, exchange = create_strategy()
        for i in range(4):
            bar = make_bar(open_=100 + i, close=105 + i, ts=1000 + i * 3600000)
            exchange.on_new_bar(bar)
            strategy._push_bar(bar)

        bar5 = make_bar(open_=110, close=115, ts=1000 + 4 * 3600000)
        exchange.on_new_bar(bar5)

        pos = exchange.get_position("BTCUSDT")
        assert pos is None


class TestClosePosition:
    def test_close_on_profit_candle(self):
        """With threshold=1, one profit candle should trigger close."""
        strategy, exchange = create_strategy({"PROFIT_CANDLE_THRESHOLD": 1})
        # Open short: 5 up candles
        for i in range(5):
            bar = make_bar(open_=100, close=105, ts=1000 + i * 3600000)
            exchange.on_new_bar(bar)
            strategy._push_bar(bar)
        # Fill sell order
        bar_fill = make_bar(open_=106, close=108, ts=1000 + 5 * 3600000)
        exchange.on_new_bar(bar_fill)
        strategy._push_bar(bar_fill)

        assert exchange.get_position("BTCUSDT") is not None
        assert exchange.get_position("BTCUSDT").side == "short"

        # Profit candle for short = down candle
        bar_profit = make_bar(open_=108, close=103, ts=1000 + 6 * 3600000)
        exchange.on_new_bar(bar_profit)
        strategy._push_bar(bar_profit)

        # Close order fills on next bar
        bar_next = make_bar(open_=103, close=104, ts=1000 + 7 * 3600000)
        exchange.on_new_bar(bar_next)

        trades = exchange.get_trades()
        close_trades = [t for t in trades if t.pnl != 0]
        assert len(close_trades) >= 1

    def test_profit_candle_threshold_3(self):
        """With threshold=3, need 3 consecutive profit candles."""
        strategy, exchange = create_strategy({"PROFIT_CANDLE_THRESHOLD": 3})
        # Open short: 5 up candles
        for i in range(5):
            bar = make_bar(open_=100, close=105, ts=1000 + i * 3600000)
            exchange.on_new_bar(bar)
            strategy._push_bar(bar)
        # Fill
        bar_fill = make_bar(open_=106, close=108, ts=1000 + 5 * 3600000)
        exchange.on_new_bar(bar_fill)
        strategy._push_bar(bar_fill)

        assert exchange.get_position("BTCUSDT").side == "short"

        # 2 profit candles (down) - not enough
        for i in range(2):
            bar = make_bar(open_=108 - i, close=106 - i, ts=1000 + (6 + i) * 3600000)
            exchange.on_new_bar(bar)
            strategy._push_bar(bar)

        assert strategy._profit_candle_count == 2

    def test_close_on_loss_candle(self):
        """Loss candle should close immediately."""
        strategy, exchange = create_strategy()
        # Open short: 5 up candles
        for i in range(5):
            bar = make_bar(open_=100, close=105, ts=1000 + i * 3600000)
            exchange.on_new_bar(bar)
            strategy._push_bar(bar)
        # Fill sell order
        bar_fill = make_bar(open_=106, close=108, ts=1000 + 5 * 3600000)
        exchange.on_new_bar(bar_fill)
        strategy._push_bar(bar_fill)

        assert exchange.get_position("BTCUSDT").side == "short"

        # Loss candle for short = up candle (continues trend)
        bar_loss = make_bar(open_=108, close=112, ts=1000 + 6 * 3600000)
        exchange.on_new_bar(bar_loss)
        strategy._push_bar(bar_loss)

        # Close order fills on next bar
        bar_next = make_bar(open_=112, close=110, ts=1000 + 7 * 3600000)
        exchange.on_new_bar(bar_next)

        trades = exchange.get_trades()
        assert len(trades) >= 2  # open + close


class TestOptimizerCompatibility:
    def test_class_attributes_are_overridable(self):
        """Optimizer creates subclass with overridden class attributes."""
        from strategies.consecutive_reverse import ConsecutiveReverseStrategy  # noqa: F401

        ParamStrategy = type(
            "ParamStrategy",
            (ConsecutiveReverseStrategy,),
            {"CONSECUTIVE_THRESHOLD": 3, "POSITION_MULTIPLIER": 1.5},
        )
        exchange = SimExchange(
            balance=1000.0, leverage=50,
            commission_rate=0.0005, funding_rate=0.0001,
            maintenance_margin=0.005,
        )
        strategy = ParamStrategy(exchange=exchange, symbol="BTCUSDT")
        strategy.on_init()

        assert strategy.CONSECUTIVE_THRESHOLD == 3
        assert strategy.POSITION_MULTIPLIER == 1.5

    def test_full_backtest_sequence(self):
        """Run a short sequence and verify trades produced."""
        strategy, exchange = create_strategy()

        bars = []
        for i in range(5):
            bars.append(make_bar(open_=100, close=105, ts=1000 + i * 3600000))
        bars.append(make_bar(open_=105, close=100, ts=1000 + 5 * 3600000))
        bars.append(make_bar(open_=100, close=106, ts=1000 + 6 * 3600000))

        for bar in bars:
            exchange.on_new_bar(bar)
            strategy._push_bar(bar)

        trades = exchange.get_trades()
        assert len(trades) > 0

    def test_stress_1000_bars(self):
        """1000 bars should complete without error."""
        strategy, exchange = create_strategy()

        for i in range(1000):
            if i % 7 < 4:
                bar = make_bar(open_=100, close=105, ts=1000 + i * 3600000)
            elif i % 7 < 6:
                bar = make_bar(open_=105, close=100, ts=1000 + i * 3600000)
            else:
                bar = make_bar(open_=100, close=100, ts=1000 + i * 3600000)
            exchange.on_new_bar(bar)
            strategy._push_bar(bar)

        assert exchange.equity > 0
