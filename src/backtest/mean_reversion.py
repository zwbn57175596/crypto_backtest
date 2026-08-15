from dataclasses import dataclass


@dataclass
class MRParams:
    """全币共享的一套均值回归参数（抗过拟合）。"""
    threshold: int = 4            # 连续同色 K 线数达到后反手
    tp_pct: float = 0.006         # 止盈距离（比例）
    sl_pct: float = 0.006         # 止损距离（比例）
    time_stop_bars: int = 12      # 最长持仓 bar 数
    risk_pct: float = 0.005       # 每笔风险占净值比例
    entry_offset_pct: float = 0.001  # maker 限价相对 close 的偏移
    entry_ttl_bars: int = 2       # 限价单有效期（bar 数）


def bar_direction(open_: float, close: float) -> int:
    if close > open_:
        return 1
    if close < open_:
        return -1
    return 0


def update_streak(count: int, streak_dir: int, direction: int) -> tuple[int, int]:
    if direction == 0:
        return count, streak_dir
    if direction == streak_dir:
        return count + 1, streak_dir
    return 1, direction


def fade_side(streak_dir: int) -> str:
    # 连涨 → 做空（sell）；连跌 → 做多（buy）
    return "sell" if streak_dir == 1 else "buy"


def entry_price(close: float, streak_dir: int, offset_pct: float) -> float:
    # 在反弹侧挂 maker 限价
    if streak_dir == 1:                 # 做空：挂在 close 上方
        return close * (1 + offset_pct)
    return close * (1 - offset_pct)      # 做多：挂在 close 下方


def bracket_prices(
    entry: float, position_side: str, tp_pct: float, sl_pct: float
) -> tuple[float, float]:
    if position_side == "long":
        return entry + entry * tp_pct, entry - entry * sl_pct
    return entry - entry * tp_pct, entry + entry * sl_pct


def position_notional(equity: float, risk_pct: float, sl_pct: float) -> float:
    if sl_pct <= 0:
        return 0.0
    return equity * risk_pct / sl_pct
