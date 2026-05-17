"""
Shadow Power Build 策略 - 基于项目回测框架的实现

使用 15min 作为 DataFeed 周期，策略内部聚合 4H K线用于信号判断，
每根 15min bar 做止损检查。

运行方式:
    python -m backtest run --strategy strategies/shadow_power_backtest.py \
        --symbol BTCUSDT --interval 15m \
        --start 2024-01-01 --end 2024-12-31 \
        --balance 1000 --leverage 49
"""

from dataclasses import dataclass, asdict
from backtest.strategy import BaseStrategy
from backtest.models import Bar


# ==================== 4H Bar 聚合结构 ====================

@dataclass
class Bar4H:
    """聚合后的 4H K线"""
    open: float
    high: float
    low: float
    close: float
    volume: float
    timestamp: int  # open timestamp (ms)
    close_time: int  # close timestamp (ms)


# ==================== 策略实现 ====================

class ShadowPowerStrategy(BaseStrategy):
    """
    Shadow Power Build 策略

    信号逻辑（4H级别）:
      1. 当前K线是 DECISION_LEN 根内最高点 + 上影线 → 做空
      2. 当前K线是 DECISION_LEN 根内最低点 + 下影线 → 做多
      3. 前一根是最高点 + 当前阴线 + 成交量最大 → 做空
      4. 前一根是最低点 + 当前阳线 + 成交量最大 → 做多

    止损逻辑（15min级别）:
      - FP止损: 浮动亏损 >= 保证金 × STOPLOSS_FACTOR
      - TB止损: 总权益跌破历史峰值 × TB_LOST_LIMIT
    """

    # ==================== 策略参数 ====================

    # 形态判断
    TOLERANCE_RATE = 0.00618       # min/max 容忍率 (0.618%)
    SHADOW_FACTOR = 2.5            # 影线倍数
    MAINPART_RATE = 0.09           # 实体/振幅最小比
    VOLATILE_RATE = 0.00382        # 振幅/价格最小比

    # 判断长度
    DECISION_LEN = 49              # 信号判断回看长度
    VOLUME_DECISION_LEN = 49       # 成交量判断长度
    SL_LEN = 49                    # 止损列表长度

    # 下单力度
    ORDER_AMOUNT_PERCENT = 0.1     # 余额 × 杠杆 × 该比例 = 名义价值

    # 止损
    RATE_OF_STOPLOSS = 0.5
    STOPLOSS_FACTOR = 1 / RATE_OF_STOPLOSS  # = 2.0
    TB_LOST_LIMIT = 0.3            # TB 止损阈值

    # 追仓
    WEEKLY_CHECK_SEC = 60 * 60 * 24 * 5  # 5天
    BUILD_TO_LVL = 0.08

    # 连续盈亏调整
    MAX_NEGATIVE_POWER = 10
    MAX_POSITIVE_POWER = 6
    BASE_FOR_POWER = 0.3
    NEGATIVE_CONFIG = 0
    ROUNDED_E = 2.7183

    # 4H 聚合: 15min × 16 = 4H
    BARS_PER_4H = 16

    def on_init(self) -> None:
        # 15min bar 缓存，用于聚合 4H
        self._15m_buffer: list[Bar] = []
        # 已聚合的 4H K线列表
        self._bars_4h: list[Bar4H] = []
        # 上一根处理过的 4H close_time（防重复）
        self._last_4h_close_time: int = 0

        # 止损/盈亏追踪（策略内部维护）
        self._tb_list: list[float] = []
        self._fp_list: list[float] = []
        self._sl_tp_list: list[int] = []  # 1=盈利平仓, -1=亏损平仓

        # 追仓时间追踪
        self._first_bar_ts: int = 0
        self._last_build_check_ts: int = 0

    def on_bar(self, bar: Bar) -> None:
        # 记录第一根 bar 的时间
        if self._first_bar_ts == 0:
            self._first_bar_ts = bar.timestamp

        # 1) 每根 15min bar 做止损检查
        self._check_stop_loss(bar.close)

        # 2) 聚合 4H K线
        self._15m_buffer.append(bar)
        if not self._is_4h_boundary(bar):
            return

        # 聚合出一根 4H bar
        bar_4h = self._aggregate_4h()
        if bar_4h is None:
            return

        # 防重复处理
        if bar_4h.close_time == self._last_4h_close_time:
            return
        self._last_4h_close_time = bar_4h.close_time

        self._bars_4h.append(bar_4h)

        # 需要足够的 4H 历史
        if len(self._bars_4h) < self.DECISION_LEN + 2:
            return

        # 3) 追仓检查
        self._check_position_building(bar_4h)

        # 4) 信号判断
        self._check_signals(bar_4h)

    # ==================== 4H 聚合 ====================

    def _is_4h_boundary(self, bar: Bar) -> bool:
        """判断当前 15min bar 是否为 4H 边界（4H K线收盘）"""
        # 4H 对齐: UTC 0:00, 4:00, 8:00, 12:00, 16:00, 20:00
        # 15min bar 的 timestamp 是 open time (ms)
        # 该 bar 的 close time = timestamp + 15*60*1000
        close_time_sec = (bar.timestamp + 15 * 60 * 1000) // 1000
        return close_time_sec % (4 * 3600) == 0

    def _aggregate_4h(self) -> Bar4H | None:
        """从 15min buffer 聚合出一根 4H bar，然后清空 buffer"""
        buf = self._15m_buffer
        if not buf:
            return None

        bar_4h = Bar4H(
            open=buf[0].open,
            high=max(b.high for b in buf),
            low=min(b.low for b in buf),
            close=buf[-1].close,
            volume=sum(b.volume for b in buf),
            timestamp=buf[0].timestamp,
            close_time=buf[-1].timestamp + 15 * 60 * 1000,
        )
        self._15m_buffer = []
        return bar_4h

    # ==================== 止损检查 (每15min) ====================

    def _check_stop_loss(self, current_price: float) -> None:
        pos = self.position
        if pos is None:
            return

        # 计算浮动盈亏（框架 PnL 是 qty*(price_diff)/entry_price，这里用原策略公式）
        # 原策略: (current - entry) * amount (对 long)
        # 框架 Position.quantity 是 USDT notional
        # 框架 unrealized_pnl = qty * (close - entry) / entry (for long)
        # 但我们需要用策略自己的止损逻辑
        if pos.side == "long":
            floating_profit = pos.quantity * (current_price - pos.entry_price) / pos.entry_price
        else:
            floating_profit = pos.quantity * (pos.entry_price - current_price) / pos.entry_price

        margin = pos.margin
        total_balance = self.balance + floating_profit

        # 更新列表
        self._fp_list.append(floating_profit)
        self._tb_list.append(total_balance)
        if len(self._tb_list) > self.SL_LEN:
            self._tb_list.pop(0)
            self._fp_list.pop(0)

        # TB止损: 总权益跌破历史峰值 × TB_LOST_LIMIT
        if self._tb_list:
            max_tb = max(self._tb_list)
            if max_tb > 0 and total_balance / max_tb <= self.TB_LOST_LIMIT:
                if floating_profit > 0:
                    self._sl_tp_list.append(1)
                else:
                    self._sl_tp_list.append(-1)
                self.close()
                return

        # FP止损: 浮动亏损 >= margin × STOPLOSS_FACTOR
        if floating_profit != 0 and floating_profit * (-self.STOPLOSS_FACTOR) >= margin:
            self._sl_tp_list.append(-1)
            self.close()

    # ==================== 信号判断 (每4H) ====================

    def _check_signals(self, current_4h: Bar4H) -> None:
        bars = self._bars_4h
        n = len(bars)
        i = n - 1  # current index

        # 回看窗口
        window = bars[i - self.DECISION_LEN: i]  # 不含当前
        vol_window = window[-self.VOLUME_DECISION_LEN:]

        # 当前盈亏状态
        fp_before = self._fp_list[-1] if self._fp_list else 0
        cft = self._count_fp_trend()
        if cft == -3:
            cft = cft - 3
        elif cft >= 1:
            cft = cft + 2

        current_price = current_4h.close
        signal_direction: str | None = None
        amount = 0.0

        # 信号1: 当前4H是最高点 + 上影线 → 做空
        if self._is_max(window, current_4h) and self._is_up_shadow(current_4h):
            self._record_close_pnl("long", fp_before)
            signal_direction = "short"
            amount = self._calc_order_amount(current_price, cft)

        # 信号2: 当前4H是最低点 + 下影线 → 做多
        elif self._is_min(window, current_4h) and self._is_down_shadow(current_4h):
            self._record_close_pnl("short", fp_before)
            signal_direction = "long"
            amount = self._calc_order_amount(current_price, cft)

        # 信号3: 前一根是最高点 + 当前阴线 + 成交量最大 → 做空
        if signal_direction is None and i >= 2:
            prev_4h = bars[i - 1]
            if self._is_max(window, prev_4h):
                if current_4h.close < current_4h.open and self._is_volume_max(vol_window, current_4h):
                    self._record_close_pnl("long", fp_before)
                    signal_direction = "short"
                    amount = self._calc_order_amount(current_price, cft)

        # 信号4: 前一根是最低点 + 当前阳线 + 成交量最大 → 做多
        if signal_direction is None and i >= 2:
            prev_4h = bars[i - 1]
            if self._is_min(window, prev_4h):
                if current_4h.close > current_4h.open and self._is_volume_max(vol_window, current_4h):
                    self._record_close_pnl("short", fp_before)
                    signal_direction = "long"
                    amount = self._calc_order_amount(current_price, cft)

        # 执行信号
        if signal_direction and amount > 0:
            pos = self.position
            # 先平仓（反手）
            if pos is not None:
                self.close()
            # 开仓
            if signal_direction == "long":
                self.buy(amount)
            else:
                self.sell(amount)

    def _record_close_pnl(self, existing_side: str, fp_before: float) -> None:
        """如果当前持仓方向与 existing_side 相同，记录盈亏"""
        pos = self.position
        if pos is None or pos.side != existing_side:
            return
        if fp_before > 0:
            self._sl_tp_list.append(1)
        elif fp_before < 0:
            self._sl_tp_list.append(-1)

    # ==================== 追仓 (每5天) ====================

    def _check_position_building(self, current_4h: Bar4H) -> None:
        pos = self.position
        if pos is None:
            return

        elapsed_sec = (current_4h.timestamp - self._first_bar_ts) // 1000
        if elapsed_sec <= 0:
            return

        # 检查是否到达 5 天间隔
        check_interval = self.WEEKLY_CHECK_SEC
        last_check = self._last_build_check_ts
        current_sec = current_4h.timestamp // 1000

        if last_check == 0:
            # 首次：从回测开始计算
            periods_elapsed = elapsed_sec // check_interval
            if periods_elapsed == 0:
                return
            next_check_at = self._first_bar_ts // 1000 + periods_elapsed * check_interval
            if current_sec < next_check_at:
                return
        else:
            if current_sec - last_check < check_interval:
                return

        self._last_build_check_ts = current_sec

        current_price = current_4h.close
        # 计算当前浮动盈亏
        if pos.side == "long":
            fp = pos.quantity * (current_price - pos.entry_price) / pos.entry_price
        else:
            fp = pos.quantity * (pos.entry_price - current_price) / pos.entry_price

        total_balance = self.balance + fp
        leverage = pos.leverage

        # 目标仓位量 (USDT notional)
        target_qty = total_balance * leverage * self.BUILD_TO_LVL

        additional = target_qty - pos.quantity
        if additional > 0:
            # 追加仓位：同方向加仓
            if pos.side == "long":
                self.buy(additional)
            else:
                self.sell(additional)

    # ==================== 形态判断 ====================

    def _is_up_shadow(self, bar: Bar4H) -> bool:
        """上影线形态（看跌）: 阴线 + 上影远大于下影 + 实体比例 + 振幅"""
        if bar.open <= bar.close:
            return False

        total_len = bar.high - bar.low
        if total_len == 0:
            return False
        upper_shadow = bar.high - bar.open
        lower_shadow = bar.close - bar.low
        body = bar.open - bar.close

        if upper_shadow < lower_shadow * self.SHADOW_FACTOR:
            return False
        if body < total_len * self.MAINPART_RATE:
            return False
        if total_len < bar.open * self.VOLATILE_RATE:
            return False
        return True

    def _is_down_shadow(self, bar: Bar4H) -> bool:
        """下影线形态（看涨）: 阳线 + 下影远大于上影 + 实体比例 + 振幅"""
        if bar.open >= bar.close:
            return False

        total_len = bar.high - bar.low
        if total_len == 0:
            return False
        lower_shadow = bar.open - bar.low
        upper_shadow = bar.high - bar.close
        body = bar.close - bar.open

        if lower_shadow < upper_shadow * self.SHADOW_FACTOR:
            return False
        if body < total_len * self.MAINPART_RATE:
            return False
        if total_len < bar.open * self.VOLATILE_RATE:
            return False
        return True

    def _is_max(self, window: list[Bar4H], target: Bar4H) -> bool:
        """target.high 是否为 window 内最高（允许容忍度）"""
        if not window:
            return False
        max_high = max(b.high for b in window)
        return target.high >= max_high * (1 - self.TOLERANCE_RATE)

    def _is_min(self, window: list[Bar4H], target: Bar4H) -> bool:
        """target.low 是否为 window 内最低（允许容忍度）"""
        if not window:
            return False
        min_low = min(b.low for b in window)
        return target.low <= min_low * (1 + self.TOLERANCE_RATE)

    def _is_volume_max(self, window: list[Bar4H], target: Bar4H) -> bool:
        """target.volume 是否为 window 内最大（允许容忍度）"""
        if not window:
            return False
        max_vol = max(b.volume for b in window)
        return target.volume >= max_vol * (1 - self.TOLERANCE_RATE)

    # ==================== 仓位计算 ====================

    def _count_fp_trend(self) -> int:
        """连续盈亏计数"""
        if not self._sl_tp_list:
            return 0
        last_value = self._sl_tp_list[-1]
        count = 0
        for v in reversed(self._sl_tp_list):
            if v == last_value:
                count += 1
            else:
                break
        return count if last_value == 1 else -count

    def _calc_order_amount(self, current_price: float, cft: int) -> float:
        """
        计算开仓 USDT notional（框架中 quantity = USDT notional value）
        base = balance × leverage × ORDER_AMOUNT_PERCENT
        """
        leverage = self._exchange.leverage
        base = self.balance * leverage * self.ORDER_AMOUNT_PERCENT

        if cft == 0:
            factor = 1.0
        elif cft > 0:
            factor = 1.0 / (1 + self.BASE_FOR_POWER) ** (
                (cft - 1) * self.ROUNDED_E / self.MAX_POSITIVE_POWER
            )
        else:
            if abs(cft) <= self.MAX_NEGATIVE_POWER + self.NEGATIVE_CONFIG:
                factor = (1 + self.BASE_FOR_POWER) ** (
                    (cft + self.NEGATIVE_CONFIG)
                    * self.ROUNDED_E
                    / (-self.MAX_NEGATIVE_POWER)
                )
            else:
                factor = (1 + self.BASE_FOR_POWER) ** self.ROUNDED_E

        return round(base * factor, 2)

    # ==================== 状态持久化 ====================

    def save_state(self) -> dict:
        return {
            "15m_buffer": [asdict(b) for b in self._15m_buffer],
            "bars_4h": [asdict(b) for b in self._bars_4h],
            "last_4h_close_time": self._last_4h_close_time,
            "tb_list": self._tb_list,
            "fp_list": self._fp_list,
            "sl_tp_list": self._sl_tp_list,
            "first_bar_ts": self._first_bar_ts,
            "last_build_check_ts": self._last_build_check_ts,
        }

    def load_state(self, state: dict) -> None:
        self._15m_buffer = [Bar(**d) for d in state.get("15m_buffer", [])]
        self._bars_4h = [Bar4H(**d) for d in state.get("bars_4h", [])]
        self._last_4h_close_time = state.get("last_4h_close_time", 0)
        self._tb_list = state.get("tb_list", [])
        self._fp_list = state.get("fp_list", [])
        self._sl_tp_list = state.get("sl_tp_list", [])
        self._first_bar_ts = state.get("first_bar_ts", 0)
        self._last_build_check_ts = state.get("last_build_check_ts", 0)
