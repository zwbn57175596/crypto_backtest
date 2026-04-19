from backtest.strategy import BaseStrategy
from backtest.models import Bar


class MaCrossStrategy(BaseStrategy):
    short_period = 7
    long_period = 25
    trade_quantity = 1000.0

    def on_bar(self, bar: Bar) -> None:
        bars = self.history(self.long_period)
        if len(bars) < self.long_period:
            return

        short_ma = sum(b.close for b in bars[-self.short_period:]) / self.short_period
        long_ma = sum(b.close for b in bars) / self.long_period

        prev_bars = self.history(self.long_period + 1)
        if len(prev_bars) < self.long_period + 1:
            return
        prev_short = sum(b.close for b in prev_bars[-self.short_period - 1:-1]) / self.short_period
        prev_long = sum(b.close for b in prev_bars[:-1]) / self.long_period

        pos = self.position

        if prev_short <= prev_long and short_ma > long_ma:
            if pos is None:
                self.buy(self.trade_quantity)
            elif pos.side == "short":
                self.close()
                self.buy(self.trade_quantity)

        elif prev_short >= prev_long and short_ma < long_ma:
            if pos is None:
                self.sell(self.trade_quantity)
            elif pos.side == "long":
                self.close()
                self.sell(self.trade_quantity)
