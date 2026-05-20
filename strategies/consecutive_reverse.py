"""
Consecutive Reverse Strategy - 连续K线反转策略（平仓-重开版本）

基于连续同向K线后反向开仓的均值回归策略。
当连续N根同向K线出现时，认为趋势过度延伸，反向建仓。

核心逻辑：
  - 盈利K线：当K线方向与持仓方向一致时，计数加1。达到阈值后平仓，重新开仓
  - 亏损K线：当K线方向与持仓方向相反时，立即平仓，然后尝试重新开仓
  - 与 ConsecutiveReverseMartingaleStrategy 的区别：
    此版本在亏损时采用"平仓-重开"策略，不进行加仓

与 CUDA 对应关系：
  - CPU:  ConsecutiveReverseStrategy
  - CUDA: consecutive_reverse_close_reopen_kernel

运行方式:
    python -m backtest run --strategy strategies/consecutive_reverse.py \
        --symbol BTCUSDT --interval 1h \
        --start 2020-01-01 --end 2026-03-30 \
        --balance 1000 --leverage 50

参数优化 (CPU):
    python -m backtest optimize --strategy strategies/consecutive_reverse.py \
        --symbol BTCUSDT --interval 1h \
        --start 2020-01-01 --end 2026-03-30 \
        --balance 1000 --leverage 50 \
        --params "CONSECUTIVE_THRESHOLD=3:8:1,POSITION_MULTIPLIER=0.1:3.0:0.1,INITIAL_POSITION_PCT=0.1:1.5:0.05,PROFIT_CANDLE_THRESHOLD=1:5:1" \
        --method grid --objective sharpe_ratio

参数优化 (GPU 加速):
    python -m backtest optimize --strategy strategies/consecutive_reverse.py \
        --symbol BTCUSDT --interval 1h \
        --start 2020-01-01 --end 2026-03-30 \
        --balance 1000 --leverage 50 \
        --params "CONSECUTIVE_THRESHOLD=3:8:1,POSITION_MULTIPLIER=0.1:3.0:0.1,INITIAL_POSITION_PCT=0.1:1.5:0.05,PROFIT_CANDLE_THRESHOLD=1:5:1" \
        --method cuda-grid --objective sharpe_ratio
"""

import datetime

from backtest.strategy import BaseStrategy
from backtest.models import Bar, Position


class ConsecutiveReverseStrategy(BaseStrategy):
    """连续K线反转策略"""

    # ==================== 可优化参数 ====================
    CONSECUTIVE_THRESHOLD = 7       # 连续K线触发阈值
    POSITION_MULTIPLIER = 2.9       # 仓位递增倍数
    INITIAL_POSITION_PCT = 1.19     # 初始仓位比例（占余额）
    PROFIT_CANDLE_THRESHOLD = 3     # 盈利K线平仓阈值

    def on_init(self) -> None:
        self._consecutive_count = 0
        self._streak_direction = 0  # +1 up, -1 down, 0 none
        self._profit_candle_count = 0
        self._current_bar: Bar | None = None

    def on_bar(self, bar: Bar) -> None:
        direction = self._get_direction(bar)
        if direction == 0:
            return  # skip doji

        self._current_bar = bar
        self._update_streak(direction)
        self._log_snapshot(bar)

        pos = self.position
        if pos is None:
            self._profit_candle_count = 0  # reset stale count from external close (e.g. liquidation)
            self._try_open(direction)
        elif self._is_profit_candle(pos, direction):
            self._profit_candle_count += 1
            if self._profit_candle_count >= self.PROFIT_CANDLE_THRESHOLD:
                self._log_close(bar, pos, reason=f"盈利K线={self._profit_candle_count}")
                self.close()
                self._profit_candle_count = 0
                self._try_open(direction)
        else:
            # Loss candle - close immediately
            self._log_close(bar, pos, reason="亏损K线")
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

    def _update_streak(self, direction: int) -> None:
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
        return base * multiplier

    def _try_open(self, direction: int) -> None:
        """尝试反向开仓"""
        quantity = self._calc_quantity()
        if quantity <= 0:
            return
        bar = self._current_bar
        t = self._fmt_time(bar.timestamp) if bar else "??:??"
        dir_arrow = "↑" if self._streak_direction == 1 else "↓"
        # 反向开仓：连涨做空，连跌做多
        if direction == 1:
            print(f"[开仓] {t} SHORT qty≈{quantity:.2f} 连续={self._consecutive_count}{dir_arrow} price≈{bar.close:.2f}")
            self.sell(quantity)
        else:
            print(f"[开仓] {t} LONG  qty≈{quantity:.2f} 连续={self._consecutive_count}{dir_arrow} price≈{bar.close:.2f}")
            self.buy(quantity)

    def _is_profit_candle(self, pos: Position, direction: int) -> bool:
        """判断当前K线是否为盈利K线（与持仓方向一致）"""
        if pos.side == "long" and direction == 1:
            return True
        if pos.side == "short" and direction == -1:
            return True
        return False

    # ==================== 日志辅助 ====================

    def _fmt_time(self, ts_ms: int) -> str:
        return datetime.datetime.fromtimestamp(ts_ms / 1000).strftime("%H:%M:%S")

    def _log_snapshot(self, bar: Bar) -> None:
        t = self._fmt_time(bar.timestamp)
        dir_arrow = "↑" if self._streak_direction == 1 else ("↓" if self._streak_direction == -1 else "-")
        print(f"[账户][余额] {t} balance≈{self.balance:.2f} equity≈{self.equity:.2f} 连续={self._consecutive_count}{dir_arrow} price={bar.close:.2f}")
        pos = self.position
        if pos is not None:
            side_str = "LONG " if pos.side == "long" else "SHORT"
            pnl = pos.unrealized_pnl
            pnl_str = f"+{pnl:.2f}" if pnl >= 0 else f"{pnl:.2f}"
            print(f"[持仓快照] {t} side={side_str} qty≈{pos.quantity:.2f} entry≈{pos.entry_price:.2f} uPnL={pnl_str} margin≈{pos.margin:.2f} profit_count={self._profit_candle_count}")

    def _log_close(self, bar: Bar, pos: Position, reason: str) -> None:
        t = self._fmt_time(bar.timestamp)
        pnl = pos.unrealized_pnl
        pnl_str = f"+{pnl:.2f}" if pnl >= 0 else f"{pnl:.2f}"
        side_str = "LONG " if pos.side == "long" else "SHORT"
        print(f"[平仓] {t} side={side_str} reason={reason} uPnL={pnl_str} qty≈{pos.quantity:.2f} entry≈{pos.entry_price:.2f} close={bar.close:.2f}")

    def save_state(self) -> dict:
        return {
            "consecutive_count": self._consecutive_count,
            "streak_direction": self._streak_direction,
            "profit_candle_count": self._profit_candle_count,
        }

    def load_state(self, state: dict) -> None:
        self._consecutive_count = state.get("consecutive_count", 0)
        self._streak_direction = state.get("streak_direction", 0)
        self._profit_candle_count = state.get("profit_candle_count", 0)

    # ==================== 预留方法（未激活）====================

    def _add_position(self, direction: int) -> None:
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

    def _reduce_position(self, target_quantity: float) -> None:
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
