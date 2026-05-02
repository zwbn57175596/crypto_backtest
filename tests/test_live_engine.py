from strategies.consecutive_reverse import ConsecutiveReverseStrategy
from unittest.mock import MagicMock
from backtest.models import Position


def _make_exchange():
    ex = MagicMock()
    ex.balance = 1000.0
    ex.equity = 1000.0
    ex.get_position.return_value = None
    return ex


class TestStrategyStateHooks:
    def test_base_strategy_save_state_returns_empty(self):
        from backtest.strategy import BaseStrategy
        strategy = BaseStrategy(exchange=_make_exchange(), symbol="BTCUSDT")
        assert strategy.save_state() == {}

    def test_base_strategy_load_state_does_nothing(self):
        from backtest.strategy import BaseStrategy
        strategy = BaseStrategy(exchange=_make_exchange(), symbol="BTCUSDT")
        strategy.load_state({"anything": 42})  # must not raise

    def test_consecutive_reverse_save_state(self):
        strategy = ConsecutiveReverseStrategy(exchange=_make_exchange(), symbol="BTCUSDT")
        strategy.on_init()
        strategy._consecutive_count = 5
        strategy._streak_direction = -1
        strategy._profit_candle_count = 2
        state = strategy.save_state()
        assert state == {
            "consecutive_count": 5,
            "streak_direction": -1,
            "profit_candle_count": 2,
        }

    def test_consecutive_reverse_load_state(self):
        strategy = ConsecutiveReverseStrategy(exchange=_make_exchange(), symbol="BTCUSDT")
        strategy.on_init()
        strategy.load_state({"consecutive_count": 7, "streak_direction": 1, "profit_candle_count": 3})
        assert strategy._consecutive_count == 7
        assert strategy._streak_direction == 1
        assert strategy._profit_candle_count == 3

    def test_consecutive_reverse_load_state_missing_keys(self):
        strategy = ConsecutiveReverseStrategy(exchange=_make_exchange(), symbol="BTCUSDT")
        strategy.on_init()
        strategy.load_state({})  # empty state → all defaults
        assert strategy._consecutive_count == 0
        assert strategy._streak_direction == 0
        assert strategy._profit_candle_count == 0
