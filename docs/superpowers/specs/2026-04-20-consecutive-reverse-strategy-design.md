# Consecutive Reverse Strategy - Design Spec

## Overview

Adapt `strategies/portfolio_backtest.py` (standalone multi-symbol script) into a single-symbol `BaseStrategy` subclass compatible with the backtesting framework's optimizer for parameter search.

## Strategy Logic

**Core idea:** After N consecutive same-direction candles, open a position in the **reverse** direction (mean reversion bet). Close on profit candles or loss candles.

### Signal Rules

1. Track consecutive same-direction candles (up=+1, down=-1, doji=0 skipped)
2. When `consecutive_count >= CONSECUTIVE_THRESHOLD`: open reverse position
3. Close when:
   - **Profit candle** (candle matches position direction): close after `PROFIT_CANDLE_THRESHOLD` consecutive profit candles
   - **Loss candle** (candle opposes position direction): close immediately
4. After closing, if `consecutive_count >= CONSECUTIVE_THRESHOLD`: re-open immediately

### Position Sizing

```
base_size = balance * INITIAL_POSITION_PCT
n = consecutive_count - CONSECUTIVE_THRESHOLD + 1
multiplier = POSITION_MULTIPLIER ^ (n - 1)
quantity = base_size * multiplier * LEVERAGE   # notional value in USDT
```

## Optimizable Parameters

All defined as class attributes (standard pattern for this framework's optimizer):

| Parameter | Type | Default | Search Range (example) |
|-----------|------|---------|----------------------|
| `CONSECUTIVE_THRESHOLD` | int | 5 | 3:8:1 |
| `POSITION_MULTIPLIER` | float | 1.1 | 1.0:1.5:0.1 |
| `INITIAL_POSITION_PCT` | float | 0.01 | 0.005:0.03:0.005 |
| `PROFIT_CANDLE_THRESHOLD` | int | 1 | 1:5:1 |
| `LEVERAGE` | int | 50 | 10:50:10 |

## Implementation Structure

```python
class ConsecutiveReverseStrategy(BaseStrategy):
    # === Optimizable parameters (class attributes) ===
    CONSECUTIVE_THRESHOLD = 5
    POSITION_MULTIPLIER = 1.1
    INITIAL_POSITION_PCT = 0.01
    PROFIT_CANDLE_THRESHOLD = 1
    LEVERAGE = 50

    # === Internal state ===
    def on_init(self):
        self._consecutive_count = 0
        self._streak_direction = 0  # +1 up, -1 down
        self._profit_candle_count = 0

    def on_bar(self, bar: Bar):
        direction = self._get_direction(bar)
        if direction == 0:
            return  # skip doji

        self._update_streak(direction)

        pos = self.position
        if pos is None:
            self._try_open(bar, direction)
        elif self._is_profit_candle(pos, direction):
            self._profit_candle_count += 1
            if self._profit_candle_count >= self.PROFIT_CANDLE_THRESHOLD:
                self.close()
                self._profit_candle_count = 0
                self._try_open(bar, direction)
        else:
            # Loss candle - close immediately
            self.close()
            self._profit_candle_count = 0
            self._try_open(bar, direction)
```

## Key Methods

- `_get_direction(bar)`: Returns +1 (close > open), -1 (close < open), 0 (doji)
- `_update_streak(direction)`: Update consecutive count and streak direction
- `_calc_quantity()`: Compute notional USDT value based on position sizing formula
- `_try_open(bar, direction)`: Open reverse position if threshold met
- `_is_profit_candle(pos, direction)`: Check if candle aligns with position side

## Framework Mapping

| Original | Framework |
|----------|-----------|
| Open short | `self.sell(quantity)` |
| Open long | `self.buy(quantity)` |
| Close position | `self.close()` |
| Commission | Handled by SimExchange: `quantity * commission_rate` |
| Funding fee | Handled by SimExchange: fixed rate every 8h |
| Liquidation | Handled by SimExchange: maintenance margin check |
| Equity curve | Handled by SimExchange: recorded per bar |

## Differences from Original

1. **Execution price**: Framework fills market orders at next bar's open (more realistic). Original uses current bar's close.
2. **Funding fee**: Framework uses constant `funding_rate` param. Original reads historical CSV rates.
3. **Liquidation**: Framework has proper maintenance margin liquidation. Original only checks equity <= 0.
4. **Add/reduce position**: Methods preserved in code (as private methods) but not called from `on_bar`. Available for future activation.

## Usage

### Single backtest
```bash
python -m backtest run \
  --strategy strategies/consecutive_reverse.py \
  --symbol BTCUSDT --interval 1h \
  --start 2024-01-01 --end 2024-12-31 \
  --balance 1000 --leverage 50
```

### Parameter optimization
```bash
python -m backtest optimize \
  --strategy strategies/consecutive_reverse.py \
  --symbol BTCUSDT --interval 1h \
  --start 2024-01-01 --end 2024-12-31 \
  --balance 1000 --leverage 50 \
  --params "CONSECUTIVE_THRESHOLD=3:8:1,POSITION_MULTIPLIER=1.0:1.5:0.1,INITIAL_POSITION_PCT=0.005:0.03:0.005,PROFIT_CANDLE_THRESHOLD=1:5:1" \
  --method grid --objective sharpe_ratio --top 10
```

### Bayesian optimization (larger search space)
```bash
python -m backtest optimize \
  --strategy strategies/consecutive_reverse.py \
  --symbol BTCUSDT --interval 1h \
  --start 2024-01-01 --end 2024-12-31 \
  --balance 1000 --leverage 50 \
  --params "CONSECUTIVE_THRESHOLD=3:8:1,POSITION_MULTIPLIER=1.0:2.0:0.05,INITIAL_POSITION_PCT=0.005:0.05:0.005,PROFIT_CANDLE_THRESHOLD=1:8:1,LEVERAGE=10:50:5" \
  --method optuna --n-trials 200 --objective sharpe_ratio
```

## File Output

- `strategies/consecutive_reverse.py` — Single file, self-contained strategy

## Not In Scope

- Multi-symbol portfolio support (future phase)
- CSV data loading (uses framework SQLite)
- Excel report generation (uses framework web UI)
- Custom funding fee from CSV (uses framework constant rate)
