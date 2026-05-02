from strategies.consecutive_reverse import ConsecutiveReverseStrategy
from unittest.mock import MagicMock


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


import json
import os


def _make_live_client(balance="1000", position_amt="0"):
    from unittest.mock import MagicMock
    client = MagicMock()
    client.exchange_info.return_value = {"symbols": [{"symbol": "BTCUSDT", "filters": [
        {"filterType": "LOT_SIZE", "stepSize": "0.001"}
    ]}]}
    client.balance.return_value = [{"asset": "USDT", "availableBalance": balance, "balance": balance}]
    client.get_position_risk.return_value = [{
        "positionAmt": position_amt, "entryPrice": "0", "unrealizedProfit": "0"
    }]
    client.mark_price.return_value = {"markPrice": "50000"}
    return client


class TestLiveEngineStatePersistenceIntegration:
    def test_save_state_writes_json_file(self, tmp_path):
        from backtest.live_exchange import LiveExchange
        from backtest.live_engine import LiveEngine
        from strategies.consecutive_reverse import ConsecutiveReverseStrategy

        client = _make_live_client()
        ex = LiveExchange(client, "BTCUSDT", leverage=10, commission_rate=0.0004)
        ex.sync()

        strategy = ConsecutiveReverseStrategy(exchange=ex, symbol="BTCUSDT")
        strategy.on_init()
        strategy._consecutive_count = 4
        strategy._streak_direction = 1
        strategy._profit_candle_count = 1

        engine = LiveEngine(
            ConsecutiveReverseStrategy, "BTCUSDT", "1h",
            leverage=10, state_dir=str(tmp_path)
        )
        engine._save_state(strategy)

        state_file = tmp_path / "BTCUSDT_1h.json"
        assert state_file.exists()
        saved = json.loads(state_file.read_text())
        assert saved["consecutive_count"] == 4
        assert saved["streak_direction"] == 1
        assert saved["profit_candle_count"] == 1

    def test_load_state_restores_strategy_on_restart(self, tmp_path):
        from backtest.live_exchange import LiveExchange
        from backtest.live_engine import LiveEngine
        from strategies.consecutive_reverse import ConsecutiveReverseStrategy

        state_file = tmp_path / "BTCUSDT_4h.json"
        state_file.write_text(json.dumps({
            "consecutive_count": 6,
            "streak_direction": -1,
            "profit_candle_count": 0,
        }))

        client = _make_live_client()
        ex = LiveExchange(client, "BTCUSDT", leverage=10, commission_rate=0.0004)
        ex.sync()

        strategy = ConsecutiveReverseStrategy(exchange=ex, symbol="BTCUSDT")
        strategy.on_init()

        engine = LiveEngine(
            ConsecutiveReverseStrategy, "BTCUSDT", "4h",
            leverage=10, state_dir=str(tmp_path)
        )
        if os.path.exists(engine._state_file):
            with open(engine._state_file) as f:
                strategy.load_state(json.load(f))

        assert strategy._consecutive_count == 6
        assert strategy._streak_direction == -1
