"""
Consecutive Reverse Strategy - 连续K线反转策略

基于连续同向K线后反向开仓的均值回归策略。
当连续N根同向K线出现时，认为趋势过度延伸，反向建仓。

运行方式:
    python -m backtest run --strategy strategies/consecutive_reverse.py `
        --symbol BTCUSDT --interval 1h `
        --start 2024-01-01 --end 2024-12-31 `
        --balance 1000 --leverage 5 `
        --initial_position_pct 0.15 `
        --consecutive_threshold 8 `
        --position_multiplier 1.5


参数优化:
    python -m backtest optimize --strategy strategies/consecutive_reverse.py \
        --symbol BTCUSDT --interval 1h \
        --start 2020-01-01 --end 2025-12-31 \
        --balance 1000 --leverage 50 \
        --params "CONSECUTIVE_THRESHOLD=2:8:1,POSITION_MULTIPLIER=1.0:1.5:0.1,INITIAL_POSITION_PCT=0.005:0.03:0.005,PROFIT_CANDLE_THRESHOLD=1:5:1" \
        --method cuda-grid --objective sharpe_ratio

自动 CUDA 寻优:
    python -m backtest optimize --strategy strategies/consecutive_reverse.py \
        --symbol BTCUSDT --interval 1h \
        --start 2020-01-01 --end 2025-12-31 \
        --balance 1000 \
        --method cuda-auto --objective sharpe_ratio
"""

from backtest.strategy import BaseStrategy
from backtest.models import Bar, Position


class ConsecutiveReverseStrategy(BaseStrategy):
    """连续K线反转策略"""

    # ==================== 可优化参数 ====================
    CONSECUTIVE_THRESHOLD = 5       # 连续K线触发阈值
    POSITION_MULTIPLIER = 1.1       # 仓位递增倍数
    INITIAL_POSITION_PCT = 0.01     # 初始仓位比例（占余额）
    PROFIT_CANDLE_THRESHOLD = 1     # 盈利K线平仓阈值
    LEVERAGE = 50                   # 杠杆倍数

    # ==================== 内置搜索空间 ====================
    OPTIMIZE_SPACE = {
        "CONSECUTIVE_THRESHOLD": (2, 8, 1),
        "POSITION_MULTIPLIER": (1.0, 1.5, 0.1),
        "INITIAL_POSITION_PCT": (0.005, 0.03, 0.005),
        "PROFIT_CANDLE_THRESHOLD": (1, 10, 1),
        "LEVERAGE": (1, 50, 3),
    }
    AUTO_OPTIMIZE_CONFIG = {
        "refine_top_k": 8,
        "refine_space": {
            "CONSECUTIVE_THRESHOLD": {"radius": 1, "step": 1},
            "POSITION_MULTIPLIER": {"radius": 0.1, "step": 0.05},
            "INITIAL_POSITION_PCT": {"radius": 0.005, "step": 0.0025, "min": 0.0025, "max": 0.04},
            "PROFIT_CANDLE_THRESHOLD": {"radius": 1, "step": 1},
            "LEVERAGE": {"radius": 3, "step": 1},
        },
    }

    def on_init(self) -> None:
        self._consecutive_count = 0
        self._streak_direction = 0  # +1 up, -1 down, 0 none
        self._profit_candle_count = 0

    def on_bar(self, bar: Bar) -> None:
        direction = self._get_direction(bar)
        if direction == 0:
            return  # skip doji

        self._update_streak(direction)

        pos = self.position
        if pos is None:
            self._profit_candle_count = 0  # reset stale count from external close (e.g. liquidation)
            self._try_open(direction)
        elif self._is_profit_candle(pos, direction):
            self._profit_candle_count += 1
            if self._profit_candle_count >= self.PROFIT_CANDLE_THRESHOLD:
                self.close()
                self._profit_candle_count = 0
                self._try_open(direction)
        else:
            # Loss candle - streak continues, add to contrarian position instead of close+reopen
            self._profit_candle_count = 0
            self._add_position(direction)

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
        return base * multiplier * self.LEVERAGE

    def _try_open(self, direction: int) -> None:
        """尝试反向开仓"""
        quantity = self._calc_quantity()
        if quantity <= 0:
            return
        # 反向开仓：连涨做空，连跌做多
        if direction == 1:
            self.sell(quantity)
        else:
            self.buy(quantity)

    def _is_profit_candle(self, pos: Position, direction: int) -> bool:
        """判断当前K线是否为盈利K线（与持仓方向一致）"""
        if pos.side == "long" and direction == 1:
            return True
        if pos.side == "short" and direction == -1:
            return True
        return False

    # ==================== 辅助方法 ====================

    def _add_position(self, direction: int) -> None:
        """加仓至当前连续K线对应的目标仓位，只补差值，节省手续费"""
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
