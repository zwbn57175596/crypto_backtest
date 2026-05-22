#!/usr/bin/env python3
"""Binance USDT-M 永续（fapi）下单与查询；经典合约账户，非 Portfolio Margin。"""

from __future__ import annotations

import math
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

try:
    from binance.um_futures import UMFutures
except ImportError:
    UMFutures = None

from bet_config import (
    ORDER_VERIFY_MAX_WAIT_SEC,
    ORDER_VERIFY_POLL_SEC,
    RECONCILE_MAX_ATTEMPTS,
    RECONCILE_ORDER_SETTLE_SEC,
)

T = TypeVar("T")


def _retry_api(fn: Callable[[], T], retries: int = 4, base_delay: float = 0.45) -> T:
    """短时网络/429 等可恢复错误：指数退避重试。"""
    last: Optional[BaseException] = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            last = e
            if attempt < retries - 1:
                time.sleep(base_delay * (2**attempt))
    assert last is not None
    raise last


class BinanceUMAdapter:
    def __init__(self, client: Any):
        self.client = client
        self._filters: Dict[str, Dict[str, float]] = {}

    def ensure_filters(self, symbol: str) -> Dict[str, float]:
        if symbol in self._filters:
            return self._filters[symbol]
        info = _retry_api(lambda: self.client.exchange_info())
        for s in info.get("symbols", []):
            if s.get("symbol") != symbol:
                continue
            step = 0.0
            min_qty = 0.0
            min_notional = 0.0
            for f in s.get("filters", []):
                ft = f.get("filterType")
                if ft == "LOT_SIZE":
                    step = float(f.get("stepSize", "0"))
                    min_qty = float(f.get("minQty", "0"))
                elif ft == "MIN_NOTIONAL":
                    min_notional = float(f.get("notional", f.get("minNotional", "0")))
            precision = int(s.get("quantityPrecision", 8))
            out = {
                "step": step,
                "min_qty": min_qty,
                "min_notional": min_notional,
                "precision": precision,
            }
            self._filters[symbol] = out
            return out
        raise ValueError(f"symbol not in exchange_info: {symbol}")

    def round_qty(self, symbol: str, qty: float, round_up: bool = False) -> float:
        f = self.ensure_filters(symbol)
        step = f["step"]
        prec = f["precision"]
        if qty <= 0:
            return 0.0
        if step > 0:
            n = qty / step
            n = math.ceil(n - 1e-12) if round_up else math.floor(n + 1e-12)
            q = n * step
        else:
            q = qty
        return float(f"{q:.{prec}f}")

    def get_usdt_balances(self) -> Tuple[float, float]:
        """(钱包余额, 可用余额) USDT。bet_live 中档位 sizing 使用钱包余额；下单实际受可用约束。"""
        balances = _retry_api(lambda: self.client.balance())
        for b in balances:
            if b.get("asset") != "USDT":
                continue
            w = float(
                b.get("balance")
                or b.get("walletBalance")
                or b.get("wallet_balance")
                or 0
            )
            raw_a = b.get("availableBalance")
            if raw_a is None:
                raw_a = b.get("available_balance")
            if raw_a is not None:
                a = float(raw_a)
            else:
                a = w
            return w, max(0.0, a)
        return 0.0, 0.0

    def get_usdt_balance(self) -> float:
        """钱包余额（展示用）。"""
        return self.get_usdt_balances()[0]

    def get_usdt_available_balance(self) -> float:
        return self.get_usdt_balances()[1]

    def get_position_amt(self, symbol: str) -> Tuple[float, float]:
        risks = _retry_api(lambda: self.client.get_position_risk(symbol=symbol))
        for r in risks:
            if r.get("symbol") != symbol:
                continue
            amt = float(r.get("positionAmt", 0) or 0)
            entry = float(r.get("entryPrice", 0) or 0)
            return amt, entry
        return 0.0, 0.0

    def get_all_position_amts(self) -> Dict[str, float]:
        """一次拉取全合约持仓 positionAmt（无 symbol 的 positionRisk）。未返回列表时为空 dict。"""
        raw = _retry_api(lambda: self.client.get_position_risk())
        out: Dict[str, float] = {}
        if not isinstance(raw, list):
            return out
        for r in raw:
            sym = str(r.get("symbol", "") or "")
            if not sym:
                continue
            out[sym] = float(r.get("positionAmt", 0) or 0)
        return out

    def set_leverage(self, symbol: str, leverage: int):
        self.client.change_leverage(symbol=symbol, leverage=int(leverage))

    def mark_price(self, symbol: str) -> float:
        p = _retry_api(lambda: self.client.mark_price(symbol=symbol))
        return float(p.get("markPrice", 0) or 0)


def fetch_um_margin_usage_ratio(adapter: BinanceUMAdapter) -> Tuple[Optional[float], str]:
    """
    估算 USDT-M 全仓维度的「初始保证金 / 保证金余额」，用于账户级禁止「新槽位」开仓。
    优先使用 GET /fapi/v2/account 的 totalInitialMargin / totalMarginBalance；缺失时回退 USDT 钱包 (w-a)/w。
    """
    try:
        raw = _retry_api(lambda: adapter.client.account())
    except Exception as e:
        return None, f"account() failed: {e!s}"

    if not isinstance(raw, dict):
        return None, "account() response not a dict"

    def _f(v: Any, default: float = 0.0) -> float:
        try:
            return float(v if v is not None else default)
        except (TypeError, ValueError):
            return default

    tim = _f(raw.get("totalInitialMargin"))
    if tim <= 0:
        tim = _f(raw.get("totalPositionInitialMargin")) + _f(
            raw.get("totalOpenOrderInitialMargin")
        )
    tmb = _f(raw.get("totalMarginBalance"))
    if tmb > 0:
        r = min(1.0, max(0.0, tim / tmb))
        return r, f"initial≈{tim:.4f} marginBalance≈{tmb:.4f}"

    w, a = adapter.get_usdt_balances()
    if w > 0:
        r = min(1.0, max(0.0, (w - a) / w))
        return r, f"fallback(w-a)/wallet w={w:.4f} a={a:.4f}"
    return None, "no margin balance basis"


def list_um_nonzero_positions(adapter: BinanceUMAdapter) -> List[Dict[str, Any]]:
    """一次拉取 positionRisk，返回非零 U 本位永续仓位（按 symbol 排序，供日志汇总）。"""
    raw = _retry_api(lambda: adapter.client.get_position_risk())
    rows: List[Dict[str, Any]] = []
    if not isinstance(raw, list):
        return rows
    for r in raw:
        amt = float(r.get("positionAmt", 0) or 0)
        if abs(amt) < 1e-12:
            continue
        sym = str(r.get("symbol", "") or "")
        fp = float(
            r.get("unRealizedProfit")
            or r.get("unrealizedProfit")
            or r.get("unRealizedPnl")
            or 0
        )
        notion = r.get("notional")
        if notion is None:
            notion = r.get("notionalValue")
        rows.append(
            {
                "symbol": sym,
                "position_amt": amt,
                "entry_price": float(r.get("entryPrice", 0) or 0),
                "unrealized_profit": fp,
                "leverage": int(float(r.get("leverage", 0) or 0)),
                "notional": float(notion or 0.0),
            }
        )
    rows.sort(key=lambda x: str(x["symbol"]))
    return rows


def margin_to_signed_qty(margin_usdt: float, leverage: int, price: float, direction: int) -> float:
    if margin_usdt <= 0 or price <= 0:
        return 0.0
    q = margin_usdt * leverage / price
    return q if direction == 1 else -q


def snap_target_signed_qty_to_lot(
    f: Dict[str, float], precision: int, signed_ideal_qty: float
) -> float:
    """
    将 margin 公式得到的连续张数，映射为 LOT_SIZE 网格上最近、且满足 min_qty 的可持仓数量（保留符号）。
    调仓对比、下单 delta 均基于该离散目标，避免「容差外 + 名义过小」无法成交的夹缝。
    """
    if abs(signed_ideal_qty) < 1e-15:
        return 0.0
    step = float(f["step"])
    min_q = float(f["min_qty"])
    sgn = 1.0 if signed_ideal_qty > 0 else -1.0
    a = abs(float(signed_ideal_qty))
    if step > 0:
        n = int(round(a / step))
        a = n * step
    a = float(f"{a:.{precision}f}")
    if a < 1e-12:
        return 0.0
    if a + 1e-12 < min_q:
        if abs(float(signed_ideal_qty)) + 1e-12 < min_q:
            return 0.0
        k = math.ceil((min_q - 1e-12) / step) if step > 0 else 1
        a = k * step if step > 0 else min_q
        a = float(f"{a:.{precision}f}")
    return sgn * a


class ReconcileExhaustedError(RuntimeError):
    """经多轮下单与 API 持仓确认后仍无法对齐目标。"""


def _tol_qty(f: Dict[str, float]) -> float:
    return max(float(f["step"]) * 2.0, float(f["min_qty"]) * 0.5, 1e-10)


def _is_flat(pos_amt: float, f: Dict[str, float]) -> bool:
    return abs(pos_amt) <= max(float(f["min_qty"]) * 0.5, 1e-10)


def _is_near_target(pos_amt: float, target_qty: float, f: Dict[str, float]) -> bool:
    return abs(pos_amt - target_qty) <= _tol_qty(f)


def _wait_position_until(
    adapter: BinanceUMAdapter,
    symbol: str,
    ok: Callable[[float], bool],
    max_wait: float = ORDER_VERIFY_MAX_WAIT_SEC,
    poll: float = ORDER_VERIFY_POLL_SEC,
) -> Tuple[float, bool]:
    deadline = time.time() + max_wait
    amt = adapter.get_position_amt(symbol)[0]
    if ok(amt):
        return amt, True
    while time.time() < deadline:
        time.sleep(poll)
        amt = adapter.get_position_amt(symbol)[0]
        if ok(amt):
            return amt, True
    return amt, ok(amt)


def _after_order_position_probe(
    adapter: BinanceUMAdapter,
    symbol: str,
    ok: Callable[[float], bool],
    defer_per_order_verify: bool,
) -> Tuple[float, bool]:
    """defer 为 True：仅短暂等待，不读 positionRisk（省 API）；下一轮 reconcile 循环起始再拉或在 bet_live 末 bulk 复核。"""
    if defer_per_order_verify:
        time.sleep(RECONCILE_ORDER_SETTLE_SEC)
        return (0.0, False)
    return _wait_position_until(adapter, symbol, ok)


def _place_market_um(
    adapter: BinanceUMAdapter,
    symbol: str,
    side: str,
    qty_str: str,
    *,
    reduce_only: bool,
) -> None:
    kwargs: Dict[str, Any] = dict(
        symbol=symbol, side=side, type="MARKET", quantity=qty_str
    )
    if reduce_only:
        kwargs["reduceOnly"] = "true"
    _retry_api(lambda: adapter.client.new_order(**kwargs))


def _reduce_only_for_delta(
    pos_amt: float, target_qty: float, side: str
) -> bool:
    """减仓/回补空头侧等使用 reduceOnly，避免单边模式下来回开过仓。"""
    if side == "BUY" and pos_amt < -1e-12:
        return True
    if side == "SELL" and pos_amt > 1e-12:
        return True
    if (pos_amt > 0 and target_qty > 0) or (pos_amt < 0 and target_qty < 0):
        if abs(target_qty) < abs(pos_amt) - 1e-10:
            return True
    return False


def _reconcile_flatten_to_flat(
    adapter: BinanceUMAdapter,
    symbol: str,
    mark: float,
    min_delta_notional: float,
    dry_run: bool,
    f: Dict[str, float],
    prec: int,
    defer_per_order_verify: bool,
    initial_pos_amt: Optional[float] = None,
) -> Optional[str]:
    """目标空仓：轮询确认，未平净则按剩余仓位重试，不依赖单次成交假设。"""

    def fmt_qty(q: float) -> str:
        s = f"{q:.{prec}f}"
        if "." in s:
            s = s.rstrip("0").rstrip(".")
        return s if s else "0"

    remaining_hint: Optional[float] = initial_pos_amt
    if remaining_hint is not None:
        pos_amt = float(remaining_hint)
        remaining_hint = None
    else:
        pos_amt, _ = adapter.get_position_amt(symbol)
    if _is_flat(pos_amt, f):
        return None

    if dry_run:
        notional = abs(pos_amt) * mark
        side = "SELL" if pos_amt > 0 else "BUY"
        q = adapter.round_qty(symbol, abs(pos_amt), round_up=True)
        return (
            f"[DRY-RUN] {side} {fmt_qty(q)} {symbol} "
            f"(平至目标空仓 notional≈{notional:.2f})"
        )

    parts: list[str] = []
    for attempt in range(RECONCILE_MAX_ATTEMPTS):
        if attempt > 0:
            pos_amt, _ = adapter.get_position_amt(symbol)
        mark_live = adapter.mark_price(symbol)
        if mark_live > 0:
            mark = mark_live
        if _is_flat(pos_amt, f):
            if parts:
                return "（平至空仓）" + " | ".join(parts)
            return None

        notional = abs(pos_amt) * mark
        if notional < min_delta_notional and attempt == 0:
            parts.append(
                f"残余名义≈{notional:.2f} 低于 min_delta，仍尝试平净"
            )
        qty_abs = adapter.round_qty(symbol, abs(pos_amt), round_up=True)
        if qty_abs < f["min_qty"]:
            raise ReconcileExhaustedError(
                f"{symbol} 平仓失败: 剩余持仓 {pos_amt} 无法凑最小下单量 min_qty={f['min_qty']}"
            )
        side = "SELL" if pos_amt > 0 else "BUY"
        qty_str = fmt_qty(qty_abs)
        _place_market_um(adapter, symbol, side, qty_str, reduce_only=True)
        parts.append(f"第{attempt + 1}次平仓 {side} {qty_str} 下单已发")

        last_amt, ok = _after_order_position_probe(
            adapter,
            symbol,
            lambda a: _is_flat(a, f),
            defer_per_order_verify,
        )
        if defer_per_order_verify:
            parts.append("下单后等待（单标的不再读 positionRisk，结束整批 bulk 复核）")
        else:
            parts.append(f"确认 positionAmt={last_amt:.8f}")
        if ok:
            return "（平至空仓）" + " | ".join(parts)

    pos_amt, _ = adapter.get_position_amt(symbol)
    raise ReconcileExhaustedError(
        f"{symbol} 平仓经{RECONCILE_MAX_ATTEMPTS}次重试后仍有仓 positionAmt={pos_amt}"
    )


def _reconcile_toward_target(
    adapter: BinanceUMAdapter,
    symbol: str,
    leverage: int,
    desired: Tuple[int, float, float],
    mark: float,
    min_delta_notional: float,
    dry_run: bool,
    f: Dict[str, float],
    prec: int,
    defer_per_order_verify: bool,
    initial_pos_amt: Optional[float] = None,
) -> Optional[str]:
    d, margin, _ = desired

    def fmt_qty(q: float) -> str:
        s = f"{q:.{prec}f}"
        if "." in s:
            s = s.rstrip("0").rstrip(".")
        return s if s else "0"

    mark_used = mark
    target_qty = snap_target_signed_qty_to_lot(
        f, prec, margin_to_signed_qty(margin, leverage, mark_used, d)
    )

    remaining_hint: Optional[float] = initial_pos_amt
    if remaining_hint is not None:
        pos_amt = float(remaining_hint)
        remaining_hint = None
    else:
        pos_amt, _ = adapter.get_position_amt(symbol)
    if _is_near_target(pos_amt, target_qty, f):
        return None

    if dry_run:
        delta = target_qty - pos_amt
        notional = abs(delta) * mark_used
        qty_abs = adapter.round_qty(symbol, abs(delta), round_up=False)
        if qty_abs < f["min_qty"]:
            qty_abs = adapter.round_qty(symbol, abs(delta), round_up=True)
        side = "BUY" if delta > 0 else "SELL"
        return (
            f"[DRY-RUN] {side} {fmt_qty(qty_abs)} {symbol} "
            f"(delta_notional≈{notional:.2f})"
        )

    parts: list[str] = []
    for attempt in range(RECONCILE_MAX_ATTEMPTS):
        if attempt > 0:
            pos_amt, _ = adapter.get_position_amt(symbol)
        mark_live = adapter.mark_price(symbol)
        if mark_live > 0:
            mark_used = mark_live
        target_qty = snap_target_signed_qty_to_lot(
            f, prec, margin_to_signed_qty(margin, leverage, mark_used, d)
        )

        if _is_near_target(pos_amt, target_qty, f):
            if parts:
                return " | ".join(parts)
            return None

        delta = target_qty - pos_amt
        notional = abs(delta) * mark_used
        if notional < min_delta_notional:
            if _is_near_target(pos_amt, target_qty, f):
                if parts:
                    return " | ".join(parts)
                return None
            parts.append(
                f"第{attempt + 1}轮: delta 名义 {notional:.2f} < min_delta 且未达容差"
            )
            raise ReconcileExhaustedError(
                f"{symbol} 调仓未达标且 delta 过小 positionAmt={pos_amt} 目标≈{target_qty}"
            )

        qty_abs = adapter.round_qty(symbol, abs(delta), round_up=False)
        if qty_abs < f["min_qty"]:
            qty_abs = adapter.round_qty(symbol, abs(delta), round_up=True)
        if qty_abs < f["min_qty"]:
            raise ReconcileExhaustedError(
                f"{symbol} 调仓: delta 无法凑 min_qty positionAmt={pos_amt} 目标≈{target_qty}"
            )
        if f["min_notional"] > 0 and qty_abs * mark_used < f["min_notional"]:
            raise ReconcileExhaustedError(
                f"{symbol} 调仓: 不满足 MIN_NOTIONAL positionAmt={pos_amt}"
            )

        side = "BUY" if delta > 0 else "SELL"
        qty_str = fmt_qty(qty_abs)
        ro = _reduce_only_for_delta(pos_amt, target_qty, side)
        _place_market_um(adapter, symbol, side, qty_str, reduce_only=ro)
        parts.append(
            f"第{attempt + 1}次调仓 {side} {qty_str} reduceOnly={ro} "
            f"(目标张数≈{target_qty:.8f})"
        )

        last_amt, ok = _after_order_position_probe(
            adapter,
            symbol,
            lambda a, tq=target_qty: _is_near_target(a, tq, f),
            defer_per_order_verify,
        )
        if defer_per_order_verify:
            parts.append("下单后等待（单标的不再读 positionRisk，结束整批 bulk 复核）")
        else:
            parts.append(f"确认 positionAmt={last_amt:.8f}")
        if ok:
            return " | ".join(parts)

    pos_amt, _ = adapter.get_position_amt(symbol)
    mark_live = adapter.mark_price(symbol)
    if mark_live > 0:
        mark_used = mark_live
    target_qty = snap_target_signed_qty_to_lot(
        f, prec, margin_to_signed_qty(margin, leverage, mark_used, d)
    )
    raise ReconcileExhaustedError(
        f"{symbol} 调仓经{RECONCILE_MAX_ATTEMPTS}次后仍偏离 "
        f"positionAmt={pos_amt} 目标≈{target_qty}"
    )


def bulk_verify_positions_vs_engine(
    adapter: BinanceUMAdapter,
    engine: Any,
    symbols: List[str],
    leverage: int,
) -> List[str]:
    """
    一次拉取全量 positionRisk，对 symbols 逐一与策略目标比对（容许步长容差）。
    返回差异说明列表；空列表表示均未发现差异。
    """
    amap = adapter.get_all_position_amts()
    issues: List[str] = []
    for sym in symbols:
        if sym not in engine.states:
            continue
        f = adapter.ensure_filters(sym)
        mark = adapter.mark_price(sym)
        if mark <= 0:
            issues.append(f"{sym} mark 无效，跳过复核")
            continue
        leg = engine.get_desired_position(sym)
        pos_amt = float(amap.get(sym, 0.0))
        if leg is None:
            if not _is_flat(pos_amt, f):
                issues.append(
                    f"{sym} 策略空仓 所内仍有 positionAmt={pos_amt:.8f}"
                )
        else:
            target_qty = snap_target_signed_qty_to_lot(
                f,
                int(f["precision"]),
                margin_to_signed_qty(leg.size, leverage, mark, leg.direction),
            )
            if not _is_near_target(pos_amt, target_qty, f):
                issues.append(
                    f"{sym} 偏差 positionAmt={pos_amt:.8f} "
                    f"目标张数≈{target_qty:.8f}"
                )
    return issues


def reconcile_pending_sort_key(
    symbol: str,
    desired: Optional[Tuple[int, float, float]],
    mark: float,
    leverage: int,
    initial_pos_amt: Optional[float],
    adapter: BinanceUMAdapter,
) -> Tuple[int, str]:
    """
    同一根 K 线调仓批次内的执行顺序：先卸仓释放保证金，再开仓/同向加仓。
    返回 (phase, symbol)，phase 越小越先执行；symbol 仅作稳定排序。
    0 = 策略目标空仓（走平仓链路）
    1 = 减仓、反手（相对批次开始时快照仓位）
    2 = 由空开仓或同向加仓
    """
    if desired is None:
        return (0, symbol)

    if mark <= 0:
        return (2, symbol)

    f = adapter.ensure_filters(symbol)
    prec = int(f["precision"])
    tol = _tol_qty(f)
    init = 0.0 if initial_pos_amt is None else float(initial_pos_amt)

    d, margin, _ = desired
    target_qty = snap_target_signed_qty_to_lot(
        f, prec, margin_to_signed_qty(margin, leverage, mark, d)
    )

    if _is_flat(init, f):
        return (2, symbol)
    if init * target_qty < -1e-12:
        return (1, symbol)
    if abs(target_qty) + tol < abs(init):
        return (1, symbol)
    if abs(target_qty) > abs(init) + tol:
        return (2, symbol)
    return (1, symbol)


def reconcile_market_order(
    adapter: BinanceUMAdapter,
    symbol: str,
    leverage: int,
    desired: Optional[Tuple[int, float, float]],
    mark: float,
    min_delta_notional: float,
    dry_run: bool,
    *,
    defer_per_order_verify: bool = False,
    initial_pos_amt: Optional[float] = None,
) -> Optional[str]:
    """
    desired: (direction 1/-1, margin_usdt, entry_ignored) 或 None 表示策略目标为空仓。
    defer_per_order_verify=True：下单后仅等待，不读 positionRisk；由 bet_live 在每轮末 bulk 复核。
    initial_pos_amt：调仓批次前 get_all_position_amts 快照中的该标的仓位（省首轮 per-symbol positionRisk）。
    """
    if mark <= 0:
        return "mark_price 无效，跳过下单"

    f = adapter.ensure_filters(symbol)
    prec = int(f["precision"])

    if desired is None:
        msg = _reconcile_flatten_to_flat(
            adapter,
            symbol,
            mark,
            min_delta_notional,
            dry_run,
            f,
            prec,
            defer_per_order_verify,
            initial_pos_amt=initial_pos_amt,
        )
        if msg:
            return f"已市价 [平仓链路] {symbol} {msg}"
        return None

    msg = _reconcile_toward_target(
        adapter,
        symbol,
        leverage,
        desired,
        mark,
        min_delta_notional,
        dry_run,
        f,
        prec,
        defer_per_order_verify,
        initial_pos_amt=initial_pos_amt,
    )
    if msg:
        return f"已市价 [调仓链路] {symbol} {msg}"
    return None


def fetch_last_closed_kline(client: Any, symbol: str, interval: str) -> Optional[list]:
    """取最近一根已完全收盘的 K 线（open 时间、OHLC、closeTime）。"""

    def _inner() -> Optional[list]:
        raw = client.klines(symbol=symbol, interval=interval, limit=5)
        if not raw:
            return None
        now_ms = int(time.time() * 1000)
        for k in reversed(raw):
            close_time = int(k[6])
            if close_time <= now_ms - 500:
                return k
        return None

    return _retry_api(_inner)
