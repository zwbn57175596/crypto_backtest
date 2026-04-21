# Consecutive Reverse Strategy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a single-symbol `BaseStrategy` subclass that implements the consecutive-candle reverse strategy with optimizable parameters for the framework's grid/optuna optimizer.

**Architecture:** Single file strategy inheriting `BaseStrategy`. Internal state tracks consecutive candle streaks. Opens reverse positions when threshold is met, closes on profit/loss candles. All parameters are class attributes searchable by the optimizer.

**Tech Stack:** Python 3.11+, backtest framework (BaseStrategy, Bar, SimExchange)

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `strategies/consecutive_reverse.py` | Create | Strategy implementation with all trading logic |
| `tests/test_consecutive_reverse.py` | Create | Unit tests for strategy logic |

---

### Task 1: Core Strategy - Direction Detection and Streak Tracking

**Files:**
- Create: `strategies/consecutive_reverse.py`
- Create: `tests/test_consecutive_reverse.py`

- [ ] **Step 1: Write failing tests for direction detection and streak tracking**

```python
"""Tests for ConsecutiveReverseStrategy."""
import pytest
from unittest.mock import MagicMock
from backtest.models import Bar, Position
from backtest.exchange import SimExchange


def make_bar(open_: float, close: float, ts: int = 1000) -> Bar:
    """Helper to create a Bar with given open/close."""
    return Bar(
        symbol="BTCUSDT", interval="1h", timestamp=ts,
        open=open_, high=max(open_, close) + 10,
        low=min(open_, close) - 10, close=close, volume=100.0,
    )


def create_strategy(params: dict | None = None):
    """Create strategy with a real SimExchange."""
    from strategies.consecutive_reverse import ConsecutiveReverseStrategy

    exchange = SimExchange(
        balance=1000.0, leverage=50,
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
        from strategies.consecutive_reverse import ConsecutiveReverseStrategy
        strategy, _ = create_strategy()
        bar = make_bar(open_=100, close=105)
        assert strategy._get_direction(bar) == 1

    def test_bearish_candle(self):
        from strategies.consecutive_reverse import ConsecutiveReverseStrategy
        strategy, _ = create_strategy()
        bar = make_bar(open_=105, close=100)
        assert strategy._get_direction(bar) == -1

    def test_doji_candle(self):
        from strategies.consecutive_reverse import ConsecutiveReverseStrategy
        strategy, _ = create_strategy()
        bar = make_bar(open_=100, close=100)
        assert strategy._get_direction(bar) == 0


class TestStreakTracking:
    def test_consecutive_up_streak(self):
        strategy, _ = create_strategy()
        for i in range(4):
            bar = make_bar(open_=100, close=105, ts=1000 + i * 3600000)
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_consecutive_reverse.py -v`
Expected: FAIL with ModuleNotFoundError (strategies.consecutive_reverse not found)

- [ ] **Step 3: Implement direction detection and streak tracking**

```python
"""
Consecutive Reverse Strategy - 连续K线反转策略

基于连续同向K线后反向开仓的均值回归策略。
当连续N根同向K线出现时，认为趋势过度延伸，反向建仓。

运行方式:
    python -m backtest run --strategy strategies/consecutive_reverse.py \
        --symbol BTCUSDT --interval 1h \
        --start 2024-01-01 --end 2024-12-31 \
        --balance 1000 --leverage 50

参数优化:
    python -m backtest optimize --strategy strategies/consecutive_reverse.py \
        --symbol BTCUSDT --interval 1h \
        --start 2024-01-01 --end 2024-12-31 \
        --balance 1000 --leverage 50 \
        --params "CONSECUTIVE_THRESHOLD=3:8:1,POSITION_MULTIPLIER=1.0:1.5:0.1,INITIAL_POSITION_PCT=0.005:0.03:0.005,PROFIT_CANDLE_THRESHOLD=1:5:1" \
        --method grid --objective sharpe_ratio
"""

from backtest.strategy import BaseStrategy
from backtest.models import Bar


class ConsecutiveReverseStrategy(BaseStrategy):
    """连续K线反转策略"""

    # ==================== 可优化参数 ====================
    CONSECUTIVE_THRESHOLD = 5       # 连续K线触发阈值
    POSITION_MULTIPLIER = 1.1       # 仓位递增倍数
    INITIAL_POSITION_PCT = 0.01     # 初始仓位比例（占余额）
    PROFIT_CANDLE_THRESHOLD = 1     # 盈利K线平仓阈值
    LEVERAGE = 50                   # 杠杆倍数

    def on_init(self):
        self._consecutive_count = 0
        self._streak_direction = 0  # +1 up, -1 down, 0 none
        self._profit_candle_count = 0

    def on_bar(self, bar: Bar):
        direction = self._get_direction(bar)
        if direction == 0:
            return  # skip doji

        self._update_streak(direction)

    def _get_direction(self, bar: Bar) -> int:
        """判断K线方向: +1 阳线, -1 阴线, 0 十字星"""
        if bar.close > bar.open:
            return 1
        elif bar.close < bar.open:
            return -1
        return 0

    def _update_streak(self, direction: int):
        """更新连续计数"""
        if direction == self._streak_direction:
            self._consecutive_count += 1
        else:
            self._consecutive_count = 1
            self._streak_direction = direction
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_consecutive_reverse.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add strategies/consecutive_reverse.py tests/test_consecutive_reverse.py
git commit -m "feat: add consecutive reverse strategy skeleton with direction and streak tracking"
```

---

### Task 2: Position Sizing and Opening Logic

**Files:**
- Modify: `strategies/consecutive_reverse.py`
- Modify: `tests/test_consecutive_reverse.py`

- [ ] **Step 1: Write failing tests for position sizing and opening**

Add to `tests/test_consecutive_reverse.py`:

```python
class TestPositionSizing:
    def test_below_threshold_returns_zero(self):
        strategy, _ = create_strategy()
        strategy._consecutive_count = 4  # below default threshold of 5
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
        strategy, exchange = create_strategy()
        # Feed 5 up bars (first bar sets price, orders fill on next bar open)
        for i in range(5):
            bar = make_bar(open_=100 + i, close=105 + i, ts=1000 + i * 3600000)
            exchange.on_new_bar(bar)
            strategy._push_bar(bar)

        # After 5 up candles, strategy should have submitted a sell order
        # which fills on the 6th bar
        bar6 = make_bar(open_=110, close=115, ts=1000 + 5 * 3600000)
        exchange.on_new_bar(bar6)

        pos = exchange.get_position("BTCUSDT")
        assert pos is not None
        assert pos.side == "short"

    def test_opens_long_after_consecutive_down(self):
        """5 consecutive down candles should trigger a long position."""
        strategy, exchange = create_strategy()
        for i in range(5):
            bar = make_bar(open_=200 - i, close=195 - i, ts=1000 + i * 3600000)
            exchange.on_new_bar(bar)
            strategy._push_bar(bar)

        bar6 = make_bar(open_=190, close=185, ts=1000 + 5 * 3600000)
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_consecutive_reverse.py::TestPositionSizing -v`
Expected: FAIL with AttributeError (_calc_quantity not defined)

Run: `pytest tests/test_consecutive_reverse.py::TestOpenPosition -v`
Expected: FAIL (no position opened)

- [ ] **Step 3: Implement position sizing and open logic**

Update `strategies/consecutive_reverse.py` — add `_calc_quantity` and `_try_open`, update `on_bar`:

```python
    def on_bar(self, bar: Bar):
        direction = self._get_direction(bar)
        if direction == 0:
            return  # skip doji

        self._update_streak(direction)

        pos = self.position
        if pos is None:
            self._try_open(direction)

    def _calc_quantity(self) -> float:
        """计算开仓名义价值（USDT）"""
        if self._consecutive_count < self.CONSECUTIVE_THRESHOLD:
            return 0
        base = self.balance * self.INITIAL_POSITION_PCT
        n = self._consecutive_count - self.CONSECUTIVE_THRESHOLD + 1
        multiplier = self.POSITION_MULTIPLIER ** (n - 1)
        return base * multiplier * self.LEVERAGE

    def _try_open(self, direction: int):
        """尝试反向开仓"""
        quantity = self._calc_quantity()
        if quantity <= 0:
            return
        # 反向开仓：连涨做空，连跌做多
        if direction == 1:
            self.sell(quantity)
        else:
            self.buy(quantity)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_consecutive_reverse.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add strategies/consecutive_reverse.py tests/test_consecutive_reverse.py
git commit -m "feat: add position sizing and opening logic to consecutive reverse strategy"
```

---

### Task 3: Close Position Logic (Profit and Loss Candles)

**Files:**
- Modify: `strategies/consecutive_reverse.py`
- Modify: `tests/test_consecutive_reverse.py`

- [ ] **Step 1: Write failing tests for close logic**

Add to `tests/test_consecutive_reverse.py`:

```python
class TestClosePosition:
    def test_close_on_loss_candle(self):
        """Position should close when candle opposes position direction."""
        strategy, exchange = create_strategy()
        # Open short: 5 up candles
        for i in range(5):
            bar = make_bar(open_=100, close=105, ts=1000 + i * 3600000)
            exchange.on_new_bar(bar)
            strategy._push_bar(bar)
        # Fill the sell order
        bar_fill = make_bar(open_=106, close=108, ts=1000 + 5 * 3600000)
        exchange.on_new_bar(bar_fill)
        strategy._push_bar(bar_fill)

        # Now we should have a short position
        assert exchange.get_position("BTCUSDT") is not None
        assert exchange.get_position("BTCUSDT").side == "short"

        # Loss candle for short = another up candle (triggers close)
        bar_loss = make_bar(open_=108, close=112, ts=1000 + 6 * 3600000)
        exchange.on_new_bar(bar_loss)
        strategy._push_bar(bar_loss)

        # Close order submitted, fills on next bar
        bar_next = make_bar(open_=112, close=110, ts=1000 + 7 * 3600000)
        exchange.on_new_bar(bar_next)

        # Position should be closed (or reversed)
        pos = exchange.get_position("BTCUSDT")
        # After close, if streak still >= threshold, may re-open
        # But streak reset to 1 (direction changed), so no re-open
        # Actually the loss candle is still up (direction=1), streak continues
        # streak = 7 up candles total, so it re-opens short
        # Let's just verify the close order was submitted
        trades = exchange.get_trades()
        assert len(trades) >= 2  # at least open + close

    def test_close_on_profit_candle_threshold_1(self):
        """With threshold=1, one profit candle should close position."""
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

        assert exchange.get_position("BTCUSDT").side == "short"

        # Profit candle for short = down candle
        bar_profit = make_bar(open_=108, close=103, ts=1000 + 6 * 3600000)
        exchange.on_new_bar(bar_profit)
        strategy._push_bar(bar_profit)

        # Close order fills on next bar
        bar_next = make_bar(open_=103, close=104, ts=1000 + 7 * 3600000)
        exchange.on_new_bar(bar_next)

        # After profit close + streak reset (down candle breaks streak)
        # streak = 1 down, below threshold, so no re-open
        trades = exchange.get_trades()
        close_trades = [t for t in trades if t.pnl != 0]
        assert len(close_trades) >= 1

    def test_profit_candle_threshold_3_requires_consecutive(self):
        """With threshold=3, need 3 consecutive profit candles to close."""
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

        # Still open after 2 profit candles
        # (close order not yet submitted, or if submitted won't have filled)
        # Check profit counter
        assert strategy._profit_candle_count == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_consecutive_reverse.py::TestClosePosition -v`
Expected: FAIL (close logic not implemented in on_bar)

- [ ] **Step 3: Implement close logic in on_bar**

Update `on_bar` in `strategies/consecutive_reverse.py`:

```python
    def on_bar(self, bar: Bar):
        direction = self._get_direction(bar)
        if direction == 0:
            return  # skip doji

        self._update_streak(direction)

        pos = self.position
        if pos is None:
            self._try_open(direction)
        elif self._is_profit_candle(pos, direction):
            self._profit_candle_count += 1
            if self._profit_candle_count >= self.PROFIT_CANDLE_THRESHOLD:
                self.close()
                self._profit_candle_count = 0
                self._try_open(direction)
        else:
            # Loss candle - close immediately
            self.close()
            self._profit_candle_count = 0
            self._try_open(direction)

    def _is_profit_candle(self, pos, direction: int) -> bool:
        """判断当前K线是否为盈利K线（与持仓方向一致）"""
        if pos.side == "long" and direction == 1:
            return True
        if pos.side == "short" and direction == -1:
            return True
        return False
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_consecutive_reverse.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add strategies/consecutive_reverse.py tests/test_consecutive_reverse.py
git commit -m "feat: add close position logic with profit/loss candle handling"
```

---

### Task 4: Integration Test with Optimizer Compatibility

**Files:**
- Modify: `tests/test_consecutive_reverse.py`

- [ ] **Step 1: Write integration tests**

Add to `tests/test_consecutive_reverse.py`:

```python
class TestOptimizerCompatibility:
    def test_class_attributes_are_overridable(self):
        """Optimizer creates subclass with overridden class attributes."""
        from strategies.consecutive_reverse import ConsecutiveReverseStrategy

        # This is how the optimizer parameterizes strategies
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
        """Run a short sequence of bars and verify strategy produces trades."""
        strategy, exchange = create_strategy()

        # Simulate: 5 up bars, then 1 down bar, then 1 up bar
        bars = []
        for i in range(5):
            bars.append(make_bar(open_=100, close=105, ts=1000 + i * 3600000))
        bars.append(make_bar(open_=105, close=100, ts=1000 + 5 * 3600000))
        bars.append(make_bar(open_=100, close=106, ts=1000 + 6 * 3600000))

        for bar in bars:
            exchange.on_new_bar(bar)
            strategy._push_bar(bar)

        trades = exchange.get_trades()
        assert len(trades) > 0  # should have at least opened a position

    def test_strategy_does_not_crash_on_many_bars(self):
        """Stress test: 1000 bars with alternating patterns."""
        strategy, exchange = create_strategy()

        for i in range(1000):
            # Create semi-random pattern
            if i % 7 < 4:
                bar = make_bar(open_=100, close=105, ts=1000 + i * 3600000)
            elif i % 7 < 6:
                bar = make_bar(open_=105, close=100, ts=1000 + i * 3600000)
            else:
                bar = make_bar(open_=100, close=100, ts=1000 + i * 3600000)
            exchange.on_new_bar(bar)
            strategy._push_bar(bar)

        # Should complete without error
        assert exchange.equity > 0
```

- [ ] **Step 2: Run all tests**

Run: `pytest tests/test_consecutive_reverse.py -v`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_consecutive_reverse.py
git commit -m "test: add optimizer compatibility and integration tests"
```

---

### Task 5: Final Cleanup and Verify CLI Works

**Files:**
- Modify: `strategies/consecutive_reverse.py` (add unused add/reduce as private methods)

- [ ] **Step 1: Add preserved add/reduce methods (inactive)**

Add to `strategies/consecutive_reverse.py` after `_try_open`:

```python
    # ==================== 预留方法（未激活）====================

    def _add_position(self, direction: int):
        """加仓（预留，当前未在 on_bar 中调用）"""
        quantity = self._calc_quantity()
        if quantity <= 0:
            return
        pos = self.position
        if pos is None:
            return
        # 计算需要追加的量
        add_qty = quantity - pos.quantity
        if add_qty <= 0:
            return
        if pos.side == "long":
            self.buy(add_qty)
        else:
            self.sell(add_qty)

    def _reduce_position(self, target_quantity: float):
        """减仓到目标量（预留，当前未在 on_bar 中调用）"""
        pos = self.position
        if pos is None:
            return
        reduce_qty = pos.quantity - target_quantity
        if reduce_qty <= 0:
            return
        # 部分平仓
        if pos.side == "long":
            self.sell(reduce_qty)
        else:
            self.buy(reduce_qty)
```

- [ ] **Step 2: Run full test suite**

Run: `pytest tests/test_consecutive_reverse.py -v`
Expected: All tests PASS

- [ ] **Step 3: Run existing tests to verify no regression**

Run: `pytest tests/ -v`
Expected: All existing tests still PASS

- [ ] **Step 4: Verify CLI discovery works**

Run: `python -c "from strategies.consecutive_reverse import ConsecutiveReverseStrategy; print('OK:', ConsecutiveReverseStrategy.CONSECUTIVE_THRESHOLD)"`
Expected: `OK: 5`

- [ ] **Step 5: Commit**

```bash
git add strategies/consecutive_reverse.py
git commit -m "feat: complete consecutive reverse strategy with preserved add/reduce methods"
```

---

## Final File: `strategies/consecutive_reverse.py` (Complete)

For reference, the complete file after all tasks:

```python
"""
Consecutive Reverse Strategy - 连续K线反转策略

基于连续同向K线后反向开仓的均值回归策略。
当连续N根同向K线出现时，认为趋势过度延伸，反向建仓。

运行方式:
    python -m backtest run --strategy strategies/consecutive_reverse.py \
        --symbol BTCUSDT --interval 1h \
        --start 2024-01-01 --end 2024-12-31 \
        --balance 1000 --leverage 50

参数优化:
    python -m backtest optimize --strategy strategies/consecutive_reverse.py \
        --symbol BTCUSDT --interval 1h \
        --start 2024-01-01 --end 2024-12-31 \
        --balance 1000 --leverage 50 \
        --params "CONSECUTIVE_THRESHOLD=3:8:1,POSITION_MULTIPLIER=1.0:1.5:0.1,INITIAL_POSITION_PCT=0.005:0.03:0.005,PROFIT_CANDLE_THRESHOLD=1:5:1" \
        --method grid --objective sharpe_ratio
"""

from backtest.strategy import BaseStrategy
from backtest.models import Bar


class ConsecutiveReverseStrategy(BaseStrategy):
    """连续K线反转策略"""

    # ==================== 可优化参数 ====================
    CONSECUTIVE_THRESHOLD = 5       # 连续K线触发阈值
    POSITION_MULTIPLIER = 1.1       # 仓位递增倍数
    INITIAL_POSITION_PCT = 0.01     # 初始仓位比例（占余额）
    PROFIT_CANDLE_THRESHOLD = 1     # 盈利K线平仓阈值
    LEVERAGE = 50                   # 杠杆倍数

    def on_init(self):
        self._consecutive_count = 0
        self._streak_direction = 0  # +1 up, -1 down, 0 none
        self._profit_candle_count = 0

    def on_bar(self, bar: Bar):
        direction = self._get_direction(bar)
        if direction == 0:
            return  # skip doji

        self._update_streak(direction)

        pos = self.position
        if pos is None:
            self._try_open(direction)
        elif self._is_profit_candle(pos, direction):
            self._profit_candle_count += 1
            if self._profit_candle_count >= self.PROFIT_CANDLE_THRESHOLD:
                self.close()
                self._profit_candle_count = 0
                self._try_open(direction)
        else:
            # Loss candle - close immediately
            self.close()
            self._profit_candle_count = 0
            self._try_open(direction)

    def _get_direction(self, bar: Bar) -> int:
        """判断K线方向: +1 阳线, -1 阴线, 0 十字星"""
        if bar.close > bar.open:
            return 1
        elif bar.close < bar.open:
            return -1
        return 0

    def _update_streak(self, direction: int):
        """更新连续计数"""
        if direction == self._streak_direction:
            self._consecutive_count += 1
        else:
            self._consecutive_count = 1
            self._streak_direction = direction

    def _calc_quantity(self) -> float:
        """计算开仓名义价值（USDT）"""
        if self._consecutive_count < self.CONSECUTIVE_THRESHOLD:
            return 0
        base = self.balance * self.INITIAL_POSITION_PCT
        n = self._consecutive_count - self.CONSECUTIVE_THRESHOLD + 1
        multiplier = self.POSITION_MULTIPLIER ** (n - 1)
        return base * multiplier * self.LEVERAGE

    def _try_open(self, direction: int):
        """尝试反向开仓"""
        quantity = self._calc_quantity()
        if quantity <= 0:
            return
        # 反向开仓：连涨做空，连跌做多
        if direction == 1:
            self.sell(quantity)
        else:
            self.buy(quantity)

    def _is_profit_candle(self, pos, direction: int) -> bool:
        """判断当前K线是否为盈利K线（与持仓方向一致）"""
        if pos.side == "long" and direction == 1:
            return True
        if pos.side == "short" and direction == -1:
            return True
        return False

    # ==================== 预留方法（未激活）====================

    def _add_position(self, direction: int):
        """加仓（预留，当前未在 on_bar 中调用）"""
        quantity = self._calc_quantity()
        if quantity <= 0:
            return
        pos = self.position
        if pos is None:
            return
        add_qty = quantity - pos.quantity
        if add_qty <= 0:
            return
        if pos.side == "long":
            self.buy(add_qty)
        else:
            self.sell(add_qty)

    def _reduce_position(self, target_quantity: float):
        """减仓到目标量（预留，当前未在 on_bar 中调用）"""
        pos = self.position
        if pos is None:
            return
        reduce_qty = pos.quantity - target_quantity
        if reduce_qty <= 0:
            return
        if pos.side == "long":
            self.sell(reduce_qty)
        else:
            self.buy(reduce_qty)
```
