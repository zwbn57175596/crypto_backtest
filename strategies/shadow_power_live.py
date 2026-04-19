#!/usr/bin/env python3
"""
Shadow Power Build 策略 - binance 实盘运行脚本

本目录为实盘用（binance_live_real），与 backtest 平行。
- 使用前请在测试网目录 programs/binance_live_test/ 充分验证。
- 本目录需配置实盘 API Key，且 BINANCE_TESTNET=false（或 api_key_config.txt 中设为 false）。

依赖：binance-futures-connector（已安装则无需改）
策略逻辑与回测共用同一脚本：本目录下 shadow_power_backtest.py（与 backtest/ 下内容一致，便于单独部署）。
"""

import os
import sys
import time
from datetime import datetime, timezone, timedelta

# 东八区，用于日志显示
UTC8 = timezone(timedelta(hours=8))

import pandas as pd

# 策略逻辑：本目录下 shadow_power_backtest.py（同目录引用，部署时只需复制本文件夹）
from shadow_power_backtest import (
    BLUE,
    CYAN,
    GREEN,
    MAGENTA,
    RED,
    RESET,
    YELLOW,
    BacktestEngine,
    Position,
    StrategyParams,
)

from shadow_live_config import check_config, get_config
from alert_manager import AlertManager

try:
    from binance.um_futures import UMFutures
except ImportError:
    UMFutures = None

# 交易对与 K 线周期（改这里即可同时调整「下单判断周期」与「止损判断周期」）
SYMBOL = "BTCUSDT"
INTERVAL_4H = "4h"   # 开仓/反手判断周期：拉 K 线、信号、调度均基于此（如 "4h" 实盘 / "3m" "1m" 测试）
INTERVAL_15M = "15m"  # 止损判断周期：每根该周期 K 线收盘后 5s 内做一次止损检查

# 周期字符串 -> 秒数（用于漏K线/跳K线修正及调度）
INTERVAL_SECONDS = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
}
# 由上面两个周期推导，保证调度与 K 线一致
INTERVAL_4H_SEC = INTERVAL_SECONDS.get(INTERVAL_4H, 14400)
INTERVAL_15M_SEC = INTERVAL_SECONDS.get(INTERVAL_15M, 900)

RETRY_SLEEP_SEC = 5

# ---------- 开平仓 vs 止损 时间点设计 ----------
# 1) 开仓/反手：按 INTERVAL_4H 周期，每根 K 线收盘后 OPEN_CLOSE_AFTER_4H_CLOSE_SEC 秒内启动。
# 2) 止损：按 INTERVAL_15M 周期，每根 K 线收盘后 5s 内启动。
# 3) 两周期重合时（如 4h 与 15m 整点重叠）开平仓延后 30s，避免与止损单冲突。
OPEN_CLOSE_AFTER_4H_CLOSE_SEC = 30  # 开仓周期 K 线收盘后该秒数启动开平仓判断
STOPLOSS_AFTER_15M_CLOSE_SEC = 5    # 止损周期 K 线收盘后该秒数启动止损判断
# 开平仓轮入口若检测到有挂单：先等待该秒数再重检一次，避免因 12:00:05 止损单或本轮重复触发导致的挂单尚未成交就整轮跳过、错过开仓
OPEN_ORDER_WAIT_AND_RECHECK_SEC = 20
# 开仓/反手下单后等待该秒数再查询持仓与权益，用于「操作后」推送
POST_OPEN_QUERY_DELAY_SEC = 15
# 兼容旧逻辑的别名
SLEEP_AFTER_BAR_CLOSE_SEC = OPEN_CLOSE_AFTER_4H_CLOSE_SEC
STOPLOSS_CHECK_INTERVAL_SEC = INTERVAL_15M_SEC  # 仅用于 _sleep_until_next_bar_with_stop_checks（当前已由统一调度替代）


def _interval_seconds(interval_str):
    """解析 interval 字符串为秒数，未知则按 1m。"""
    return INTERVAL_SECONDS.get(interval_str, 60)


def _fetch_klines_with_retry(client, symbol, interval, limit, end_time):
    """拉取 K 线，失败则每 5s 重试直到成功（避免漏K线）。"""
    while True:
        try:
            df = _fetch_klines(client, symbol, interval, limit=limit, end_time=end_time)
            return df
        except Exception as e:
            ts = datetime.now(UTC8).strftime("%H:%M:%S")
            print(
                f"  {YELLOW}[{ts}] 拉取K线失败，{RETRY_SLEEP_SEC}s 后重试: {e}{RESET}"
            )
            time.sleep(RETRY_SLEEP_SEC)


def _fetch_single_bar_for_time(client, symbol, interval, close_time_ms, interval_sec):
    """拉取指定收盘时间点的一根K线（用于补漏）。返回该根对应的 Series 或 None。"""
    start_ms = close_time_ms - interval_sec * 1000
    end_ms = close_time_ms + 5000
    df = _fetch_klines(
        client, symbol, interval, limit=5, start_time=start_ms, end_time=end_ms
    )
    if df.empty:
        return None
    df = df.copy()
    df["_diff"] = (df["close_time"] - close_time_ms).abs()
    df = df.sort_values("_diff").drop(columns=["_diff"])
    row = df.iloc[0]
    if abs(row["close_time"] - close_time_ms) > interval_sec * 1000 * 0.5:
        return None
    return row


def _log_kline(current_record, alert_mgr=None):
    """打印当前用于判断的最后一根K线信息（东八区），含起始与结束时间。若提供 alert_mgr 且为详细模式则推送到 Telegram。"""
    open_ts = current_record["open_time"] / 1000.0
    close_ts = current_record["close_time"] / 1000.0
    start_dt = datetime.fromtimestamp(open_ts, tz=timezone.utc).astimezone(UTC8)
    end_dt = datetime.fromtimestamp(close_ts, tz=timezone.utc).astimezone(UTC8)
    start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
    end_str = end_dt.strftime("%Y-%m-%d %H:%M:%S")
    o, h, l, c = (
        current_record["open"],
        current_record["high"],
        current_record["low"],
        current_record["close"],
    )
    vol = current_record.get("volume", 0)
    msg = (
        f"起始(东八区)={start_str} 结束(东八区)={end_str}  "
        f"O={o:.2f}  H={h:.2f}  L={l:.2f}  C={c:.2f}  Vol={vol:.2f}"
    )
    print(f"  [K线] {msg}")
    if alert_mgr:
        try:
            alert_mgr.send_log("kline", msg)
        except Exception:
            pass


def _klines_to_df(klines_list):
    """将 binance klines API 返回的列表转为与回测一致的 DataFrame。"""
    if not klines_list:
        return pd.DataFrame()
    # [ openTime, open, high, low, close, volume, closeTime, ... ]
    rows = []
    for k in klines_list:
        open_time = int(k[0])
        close_time = int(k[6])
        rows.append(
            {
                "open_time": open_time,
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
                "close_time": close_time,
            }
        )
    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["open_time"], unit="ms")
    return df


def _fetch_klines(client, symbol, interval, limit=500, end_time=None, start_time=None):
    """请求 K 线并返回 DataFrame。"""
    kwargs = {"symbol": symbol, "interval": interval, "limit": limit}
    if end_time is not None:
        kwargs["endTime"] = int(end_time)
    if start_time is not None:
        kwargs["startTime"] = int(start_time)
    raw = client.klines(**kwargs)
    return _klines_to_df(raw)


def _fetch_funding(client, symbol, limit=500, end_time=None):
    """获取资金费率历史，返回与回测格式兼容的 DataFrame。"""
    kwargs = {"symbol": symbol, "limit": limit}
    if end_time is not None:
        kwargs["endTime"] = int(end_time)
    raw = client.funding_rate(**kwargs)
    if not raw:
        return None
    rows = []
    for r in raw:
        t = int(r["fundingTime"])
        rows.append(
            {
                "timestamp": t,
                "fundingRate": float(r["fundingRate"]),
                "datetime": pd.to_datetime(t, unit="ms"),
            }
        )
    df = pd.DataFrame(rows)
    return df


def _get_margin_balance(client):
    """从 account 接口获取总保证金（多资产模式下用 totalMarginBalance，否则用 USDT）。"""
    acc = client.account()
    total = acc.get("totalMarginBalance")
    if total is not None:
        try:
            return float(total)
        except (TypeError, ValueError):
            pass
    for a in acc.get("assets", []):
        if a.get("asset") == "USDT":
            return float(a.get("totalWalletBalance", 0) or 0)
    return 0.0


def _get_position_risk(client, symbol):
    """获取指定 symbol 的持仓信息。返回 (position_amt, entry_price) 或 (0, 0)。"""
    risks = client.get_position_risk(symbol=symbol)
    for r in risks:
        amt = float(r.get("positionAmt", 0) or 0)
        if amt != 0:
            return amt, float(r.get("entryPrice", 0) or 0)
    return 0.0, 0.0


def _get_position_margin(client, symbol, leverage):
    """
    获取指定 symbol 的实际 margin 占用（基于当前价格，而非开仓价）。
    优先使用 API 返回的 isolatedMargin（逐仓），否则用 notional / leverage 计算。
    返回 margin 值（USDT），无持仓返回 0.0。
    """
    risks = client.get_position_risk(symbol=symbol)
    for r in risks:
        amt = float(r.get("positionAmt", 0) or 0)
        if amt != 0:
            isolated_margin = r.get("isolatedMargin")
            if isolated_margin is not None:
                try:
                    margin_val = float(isolated_margin)
                    if margin_val > 0:
                        return margin_val
                except (TypeError, ValueError):
                    pass
            notional = r.get("notional")
            if notional is not None:
                try:
                    notional_val = float(notional)
                    if notional_val > 0 and leverage > 0:
                        return notional_val / leverage
                except (TypeError, ValueError):
                    pass
            # 兜底：用当前价格计算（需要先获取当前价）
            current_price = _get_current_price(client, symbol)
            if current_price > 0 and leverage > 0:
                return (current_price * abs(amt)) / leverage
            return 0.0
    return 0.0


def _get_current_price(client, symbol):
    """从 24h ticker 获取最新价。"""
    tickers = client.ticker_24hr_price_change(symbol=symbol)
    if isinstance(tickers, list):
        for t in tickers:
            if t.get("symbol") == symbol:
                return float(t.get("lastPrice", 0) or 0)
    return float(tickers.get("lastPrice", 0) or 0)


def _round_quantity(qty, step=0.001):
    """按合约数量精度舍入（BTC 常用 3 位）。"""
    return round(qty, 3)


def _next_15m_bar_close_sec(now_sec=None):
    """下一个 15min K 线收盘时间戳（UTC 秒）。binance 15m 对齐 0/15/30/45 分。"""
    if now_sec is None:
        now_sec = time.time()
    return (int(now_sec) // INTERVAL_15M_SEC) * INTERVAL_15M_SEC + INTERVAL_15M_SEC


def _next_4h_bar_close_sec(now_sec, last_4h_close_sec=None):
    """返回「下一次开平仓应对应的 K 线收盘」时间戳（UTC 秒）。调用方会加上 OPEN_CLOSE_AFTER_4H_CLOSE_SEC 得到 target_open。"""
    if last_4h_close_sec is not None:
        return last_4h_close_sec + INTERVAL_4H_SEC
    # 当前所在 K 线的起始 = 刚收盘那根 K 线的收盘时间
    bar_just_closed = (int(now_sec) // INTERVAL_4H_SEC) * INTERVAL_4H_SEC
    # 若尚未过「收盘+30s」窗口，则下一次开平仓就是 bar_just_closed + 30s；否则是下一根 K 线收盘 + 30s
    if now_sec < bar_just_closed + OPEN_CLOSE_AFTER_4H_CLOSE_SEC:
        return bar_just_closed
    return bar_just_closed + INTERVAL_4H_SEC


def _run_stop_check_once(client, engine, params, dry_run, alert_mgr=None):
    """
    执行一次止损检查：同步仓位与现价，check_stop_loss，未触发打 [止损检查] 日志，触发则发平仓单。
    不 sleep，调用方负责调度。
    """
    now = datetime.now(timezone.utc)
    engine.balance = _get_margin_balance(client)
    pos_amt, entry_price = _get_position_risk(client, SYMBOL)
    if pos_amt != 0:
        direction = "long" if pos_amt > 0 else "short"
        engine.position = Position(direction, entry_price, abs(pos_amt), now)
        engine.position.calculate_margin(params.MARGIN_LEVEL)
    else:
        engine.position = None
    price_for_sl = _get_current_price(client, SYMBOL)
    pos_amt_before = pos_amt
    ts8 = datetime.now(UTC8).strftime("%H:%M:%S")
    if engine.position is None:
        print(f"  [止损检查] [{ts8}] 无持仓")
        return
    triggered = engine.check_stop_loss(price_for_sl, now)
    if triggered:
        pos_now, _ = _get_position_risk(client, SYMBOL)
        if pos_now == 0:
            print(f"  {YELLOW}[防护] 仓位已为 0，跳过重复平仓{RESET}")
            engine.position = None
            engine.balance = _get_margin_balance(client)
            return
        if _has_open_orders(client, SYMBOL):
            print(f"  {YELLOW}[防护] 当前已有挂单，跳过止损平仓，下轮再检{RESET}")
            return
        sl_profit = engine.position.calculate_profit(price_for_sl)
        sl_direction = engine.position.direction
        tb_at_trigger = engine.balance + sl_profit
        margin_at_trigger = _get_position_margin(client, SYMBOL, params.MARGIN_LEVEL)
        # 触发止损：先打日志并推送（含 TB/FP/占用margin），再发告警
        trigger_msg = (
            f"现价={price_for_sl:.2f} 当前TB={tb_at_trigger:.2f} FP={sl_profit:.2f} "
            f"占用margin={margin_at_trigger:.2f} 触发止损"
        )
        print(f"  [止损检查] [{ts8}] {trigger_msg}")
        if alert_mgr:
            try:
                alert_mgr.send_log("stop_check", trigger_msg)
            except Exception:
                pass
        side = "SELL" if pos_amt_before > 0 else "BUY"
        qty = _round_quantity(abs(pos_amt_before))
        if not dry_run:
            try:
                client.new_order(
                    symbol=SYMBOL, side=side, type="MARKET", quantity=qty, reduceOnly="true"
                )
                print(f"{GREEN}已市价平仓(止损) {side} {qty} {SYMBOL}{RESET}")
            except Exception as e:
                print(f"{RED}平仓失败: {e}{RESET}")
        else:
            print(f"{MAGENTA}[DRY-RUN] 将平仓(止损) {side} {qty} {SYMBOL}{RESET}")
        if alert_mgr:
            try:
                alert_mgr.alert_stop_loss(
                    sl_direction, price_for_sl, sl_profit, "止损触发",
                    total_balance=tb_at_trigger, margin=margin_at_trigger,
                )
            except Exception:
                pass
        engine.balance = _get_margin_balance(client)
        engine.position = None
        return
    fp = engine.position.calculate_profit(price_for_sl)
    total_bal = engine.balance + fp
    max_tb = max(engine.tb_list) if engine.tb_list else total_bal
    tb_threshold = max_tb * params.TB_LOST_LIMIT
    if engine.position.direction == "long":
        fp_stop_price = (
            engine.position.entry_price
            - engine.position.margin / (params.STOPLOSS_FACTOR * engine.position.amount)
        )
        tb_stop_price = (
            engine.position.entry_price
            + (tb_threshold - engine.balance) / engine.position.amount
        )
    else:
        fp_stop_price = (
            engine.position.entry_price
            + engine.position.margin / (params.STOPLOSS_FACTOR * engine.position.amount)
        )
        tb_stop_price = (
            engine.position.entry_price
            - (tb_threshold - engine.balance) / engine.position.amount
        )
    current_margin = _get_position_margin(client, SYMBOL, params.MARGIN_LEVEL)
    stop_check_msg = (
        f"现价={price_for_sl:.2f} FP止损价={fp_stop_price:.2f} "
        f"当前TB={total_bal:.2f} FP={fp:.2f} 占用margin={current_margin:.2f} "
        f"TB止损价={tb_stop_price:.2f} 未触发止损。"
    )
    print(f"  [止损检查] [{ts8}] {stop_check_msg}")
    if alert_mgr:
        try:
            alert_mgr.send_log("stop_check", stop_check_msg)
        except Exception:
            pass


def _handle_pos_command(client, engine, params, alert_mgr):
    """
    处理 /pos 或 /position 指令：查询当前持仓与权益，print 并推送到 Telegram（始终推送，不依赖 ALERT_VERBOSE_MODE）。
    """
    if not alert_mgr or not alert_mgr.telegram_bot_token or not alert_mgr.telegram_chat_id:
        return
    try:
        balance = _get_margin_balance(client)
        price = _get_current_price(client, SYMBOL)
        pos_amt, entry_price = _get_position_risk(client, SYMBOL)
        if pos_amt == 0:
            msg = f"当前无持仓。现价={price:.2f}，当前TB={balance:.2f}。"
            print(f"  [查询/pos] {msg}")
            alert_mgr.send_alert("position_query", msg, force=True)
            return
        now = datetime.now(timezone.utc)
        direction = "long" if pos_amt > 0 else "short"
        engine.position = Position(direction, entry_price, abs(pos_amt), now)
        engine.position.calculate_margin(params.MARGIN_LEVEL)
        engine.balance = balance
        fp = engine.position.calculate_profit(price)
        total_bal = balance + fp
        max_tb = max(engine.tb_list) if engine.tb_list else total_bal
        tb_threshold = max_tb * params.TB_LOST_LIMIT
        if engine.position.direction == "long":
            fp_stop_price = (
                engine.position.entry_price
                - engine.position.margin / (params.STOPLOSS_FACTOR * engine.position.amount)
            )
            tb_stop_price = (
                engine.position.entry_price
                + (tb_threshold - engine.balance) / engine.position.amount
            )
        else:
            fp_stop_price = (
                engine.position.entry_price
                + engine.position.margin / (params.STOPLOSS_FACTOR * engine.position.amount)
            )
            tb_stop_price = (
                engine.position.entry_price
                - (tb_threshold - engine.balance) / engine.position.amount
            )
        current_margin = _get_position_margin(client, SYMBOL, params.MARGIN_LEVEL)
        msg = (
            f"现价={price:.2f} FP止损价={fp_stop_price:.2f} "
            f"当前TB={total_bal:.2f} FP={fp:.2f} 占用margin={current_margin:.2f} "
            f"TB止损价={tb_stop_price:.2f}"
        )
        print(f"  [查询/pos] {msg}")
        alert_mgr.send_alert("position_query", msg, force=True)
    except Exception as e:
        err_msg = f"查询持仓失败: {e}"
        print(f"  {RED}[查询/pos] {err_msg}{RESET}")
        try:
            alert_mgr.send_alert("position_query", err_msg, force=True)
        except Exception:
            pass


def _sleep_until_next_bar(interval_sec, after_close_sec, last_closed_bar_close_ms=None):
    """
    睡眠到「下一根 K 线收盘 + after_close_sec 秒」再执行下一轮，使判断时点稳定在收盘后约 after_close_sec 秒内。
    若 last_closed_bar_close_ms 为 None（尚无有效 K 线），则固定 sleep 60s。
    若已过目标时间（本轮耗时过长），则至少睡 MIN_SLEEP_WHEN_LATE_SEC 秒，避免空转刷日志、打爆 API。
    """
    MIN_SLEEP_WHEN_LATE_SEC = 10
    if last_closed_bar_close_ms is None:
        time.sleep(60)
        return
    next_close_sec = last_closed_bar_close_ms / 1000.0 + interval_sec
    target = next_close_sec + after_close_sec
    duration = target - time.time()
    if duration > 0:
        ts8 = datetime.now(UTC8).strftime("%H:%M:%S")
        print(
            f"  {CYAN}[调度] [{ts8}] 等待至下一根K线收盘后 {after_close_sec}s（约 {duration:.0f}s 后执行）{RESET}"
        )
        time.sleep(duration)
    else:
        ts8 = datetime.now(UTC8).strftime("%H:%M:%S")
        print(
            f"  {YELLOW}[调度] [{ts8}] 已过目标时间，休息 {MIN_SLEEP_WHEN_LATE_SEC}s 后继续{RESET}"
        )
        time.sleep(MIN_SLEEP_WHEN_LATE_SEC)


def _sleep_until_next_bar_with_stop_checks(
    client,
    interval_sec,
    after_close_sec,
    last_closed_bar_close_ms,
    engine,
    params,
    dry_run,
):
    """
    在等待「下一根 K 线收盘 + after_close_sec」期间，每隔 STOPLOSS_CHECK_INTERVAL_SEC 唤醒一次，
    仅同步仓位与现价、执行止损判断并打印 [止损检查] 日志；若触发止损则发平仓单并返回。
    若无 last_closed_bar_close_ms 则固定 sleep 60s 后返回。
    """
    if last_closed_bar_close_ms is None:
        time.sleep(60)
        return
    next_close_sec = last_closed_bar_close_ms / 1000.0 + interval_sec
    target = next_close_sec + after_close_sec
    ts8_first = datetime.now(UTC8).strftime("%H:%M:%S")
    duration_first = target - time.time()
    if duration_first > 0:
        print(
            f"  {CYAN}[调度] [{ts8_first}] 等待至下一根K线收盘后 {after_close_sec}s"
            f"（约 {duration_first:.0f}s），期间每 {STOPLOSS_CHECK_INTERVAL_SEC // 60}min 做止损检查{RESET}"
        )
    while time.time() < target:
        sleep_sec = min(STOPLOSS_CHECK_INTERVAL_SEC, max(0, target - time.time()))
        if sleep_sec <= 0:
            break
        time.sleep(sleep_sec)
        if time.time() >= target:
            break
        now = datetime.now(timezone.utc)
        engine.balance = _get_margin_balance(client)
        pos_amt, entry_price = _get_position_risk(client, SYMBOL)
        if pos_amt != 0:
            direction = "long" if pos_amt > 0 else "short"
            engine.position = Position(
                direction, entry_price, abs(pos_amt), now
            )
            engine.position.calculate_margin(params.MARGIN_LEVEL)
        else:
            engine.position = None
        price_for_sl = _get_current_price(client, SYMBOL)
        pos_amt_before = pos_amt
        ts8 = datetime.now(UTC8).strftime("%H:%M:%S")
        if engine.position is None:
            print(f"  [止损检查] [{ts8}] 无持仓")
            continue
        triggered = engine.check_stop_loss(price_for_sl, now)
        if triggered:
            pos_now, _ = _get_position_risk(client, SYMBOL)
            if pos_now == 0:
                print(f"  {YELLOW}[防护] 仓位已为 0，跳过重复平仓{RESET}")
                engine.position = None
                engine.balance = _get_margin_balance(client)
                return
            if _has_open_orders(client, SYMBOL):
                print(
                    f"  {YELLOW}[防护] 当前已有挂单，跳过止损平仓，下轮再检{RESET}"
                )
                return
            side = "SELL" if pos_amt_before > 0 else "BUY"
            qty = _round_quantity(abs(pos_amt_before))
            if not dry_run:
                try:
                    client.new_order(
                        symbol=SYMBOL,
                        side=side,
                        type="MARKET",
                        quantity=qty,
                        reduceOnly="true",
                    )
                    print(f"{GREEN}已市价平仓(止损) {side} {qty} {SYMBOL}{RESET}")
                except Exception as e:
                    print(f"{RED}平仓失败: {e}{RESET}")
            else:
                print(f"{MAGENTA}[DRY-RUN] 将平仓(止损) {side} {qty} {SYMBOL}{RESET}")
            engine.balance = _get_margin_balance(client)
            engine.position = None
            return
        fp = engine.position.calculate_profit(price_for_sl)
        total_bal = engine.balance + fp
        max_tb = max(engine.tb_list) if engine.tb_list else total_bal
        tb_threshold = max_tb * params.TB_LOST_LIMIT
        if engine.position.direction == "long":
            fp_stop_price = (
                engine.position.entry_price
                - engine.position.margin
                / (params.STOPLOSS_FACTOR * engine.position.amount)
            )
            tb_stop_price = (
                engine.position.entry_price
                + (tb_threshold - engine.balance) / engine.position.amount
            )
        else:
            fp_stop_price = (
                engine.position.entry_price
                + engine.position.margin
                / (params.STOPLOSS_FACTOR * engine.position.amount)
            )
            tb_stop_price = (
                engine.position.entry_price
                - (tb_threshold - engine.balance) / engine.position.amount
            )
        print(
            f"  [止损检查] [{ts8}] 现价={price_for_sl:.2f} "
            f"FP止损价={fp_stop_price:.2f} TB止损价={tb_stop_price:.2f} 未触发"
        )


def _has_open_orders(client, symbol):
    """
    查询当前是否已有该 symbol 的挂单（未成交/部分成交的订单）。
    用于下单前防护：避免在上一单尚未成交时重复下单。
    返回 True 表示有挂单应跳过，False 表示无挂单可下单。请求失败时返回 False 并打日志（不阻塞下单）。
    """
    try:
        orders = client.get_orders(symbol=symbol)
        if isinstance(orders, list) and len(orders) > 0:
            return True
        return False
    except Exception as e:
        print(f"  {YELLOW}[警告] 查询挂单失败: {e}，本轮回退为不校验{RESET}")
        return False


def run_live(testnet=True, dry_run=False):
    """
    连接 binance（测试网或实盘），循环拉取 K 线、同步账户、执行止损与信号逻辑并下单。

    testnet: 是否使用合约测试网（建议先用测试网）。
    dry_run: 若为 True，只打印将要下的单，不实际发单。
    """
    if UMFutures is None:
        print(
            f"{RED}未安装 binance-futures-connector，请执行: pip install binance-futures-connector{RESET}"
        )
        return

    config = get_config()
    ok, err = check_config(config)
    if not ok:
        print(f"{RED}{err}{RESET}")
        print(
            f"{YELLOW}示例（测试网）：export BINANCE_TESTNET=true BINANCE_API_KEY=xxx BINANCE_SECRET_KEY=xxx{RESET}"
        )
        return

    base_url = config["base_url"]
    client = UMFutures(
        key=config["api_key"],
        secret=config["secret"],
        base_url=base_url,
    )

    env_label = "测试网" if config["testnet"] else "实盘"
    print(f"\n{BLUE}{'='*70}{RESET}")
    print(f"{BLUE}Shadow Power Build - binance {env_label} 运行{RESET}")
    print(f"{BLUE}{'='*70}{RESET}\n")
    print(f"{CYAN}API 根地址: {base_url}{RESET}")
    print(f"{CYAN}交易对: {SYMBOL}  杠杆: {StrategyParams.MARGIN_LEVEL}x{RESET}")
    if dry_run:
        print(f"{YELLOW}当前为【仅模拟】模式，不会真实下单{RESET}\n")

    # 测试连通性（仅用公开 ping，不涉及 IP 硬编码）
    alert_mgr = AlertManager()
    try:
        client.ping()
        print(f"{GREEN}✓ 连接 binance 成功{RESET}\n")
        alert_mgr.alert_startup({
            "testnet": config["testnet"],
            "base_url": base_url,
            "leverage": StrategyParams.MARGIN_LEVEL,
        })
    except Exception as e:
        print(f"{RED}✗ 连接失败: {e}{RESET}\n")
        alert_mgr.alert_connection_error(e)
        return

    params = StrategyParams()
    engine = BacktestEngine(params=params, silent=True)

    # 设置杠杆
    try:
        client.change_leverage(symbol=SYMBOL, leverage=params.MARGIN_LEVEL)
    except Exception as e:
        print(f"{YELLOW}设置杠杆时提示（可忽略）: {e}{RESET}")

    # 用于“仅在新 4h 收线时”跑一次信号，避免重复
    last_4h_close_time = None
    # 策略内部状态：从交易所同步余额与持仓，sl_tp_list 等由本地维护
    engine.df_funding = None  # 可选：后续可拉 funding 做统计
    balance_low_alerted = False  # 余额过低告警每轮只发一次
    last_telegram_update_id = 0   # 用于轮询 Telegram 远程指令（如 /stop）
    stop_requested = False       # 是否收到 Telegram 停止指令

    while True:
        try:
            # ---------- 检查 Telegram 是否收到停止指令（每轮先查一次）----------
            if alert_mgr.telegram_bot_token and alert_mgr.telegram_chat_id:
                try:
                    last_telegram_update_id, commands = alert_mgr.get_recent_commands(last_telegram_update_id)
                    if "stop" in commands:
                        stop_requested = True
                        print(f"\n{YELLOW}收到 Telegram 停止指令，正在退出...{RESET}\n")
                        break
                    if "pos" in commands:
                        _handle_pos_command(client, engine, params, alert_mgr)
                except Exception:
                    pass
            if stop_requested:
                break

            # ---------- 统一调度：止损 15m+5s，开平仓 4h+30s ----------
            now_ts = time.time()
            target_stop = _next_15m_bar_close_sec(now_ts) + STOPLOSS_AFTER_15M_CLOSE_SEC
            target_open = _next_4h_bar_close_sec(now_ts, last_4h_close_time) + OPEN_CLOSE_AFTER_4H_CLOSE_SEC
            sleep_until = min(target_stop, target_open)
            if now_ts < sleep_until - 1:
                ts8 = datetime.now(UTC8).strftime("%H:%M:%S")
                msg_sched = (
                    f"下次事件 {sleep_until - now_ts:.0f}s 后"
                    f"（止损 {INTERVAL_15M}+{STOPLOSS_AFTER_15M_CLOSE_SEC}s / 开平仓 {INTERVAL_4H}+{OPEN_CLOSE_AFTER_4H_CLOSE_SEC}s）"
                )
                print(
                    f"  {CYAN}[调度] [{ts8}] {msg_sched}{RESET}"
                )
                # 调度信息仅 print，不推送到 Telegram
                # 分段 sleep 并轮询 Telegram，以便尽快响应 /stop（约 30s 内）
                while time.time() < sleep_until - 1:
                    if alert_mgr.telegram_bot_token and alert_mgr.telegram_chat_id:
                        try:
                            last_telegram_update_id, commands = alert_mgr.get_recent_commands(last_telegram_update_id)
                            if "stop" in commands:
                                stop_requested = True
                                print(f"\n{YELLOW}收到 Telegram 停止指令，正在退出...{RESET}\n")
                                break
                            if "pos" in commands:
                                _handle_pos_command(client, engine, params, alert_mgr)
                        except Exception:
                            pass
                    if stop_requested:
                        break
                    time.sleep(min(30, sleep_until - time.time()))
                if stop_requested:
                    break
            now_ts = time.time()
            if now_ts < target_open - 15:
                _run_stop_check_once(client, engine, params, dry_run, alert_mgr)
                continue
            # ---------- 开平仓轮：INTERVAL_4H 收盘后约 30s 内执行，拉 K 线、同步、止损、开平仓信号 ----------
            ts8_oc = datetime.now(UTC8).strftime("%H:%M:%S")
            print(f"  {CYAN}[开平仓] [{ts8_oc}] 开始本根 {INTERVAL_4H} K 线下单判断（拉K线→同步→止损→信号）{RESET}")
            # 有挂单时先等待再重检一次，避免：① 止损单尚未成交就整轮跳过；② 本轮因网络抖动被重复触发
            if _has_open_orders(client, SYMBOL):
                ts8 = datetime.now(UTC8).strftime("%H:%M:%S")
                print(
                    f"  {CYAN}[调度] [{ts8}] 当前有挂单，等待 {OPEN_ORDER_WAIT_AND_RECHECK_SEC}s 后重检再决定是否执行开平仓{RESET}"
                )
                time.sleep(OPEN_ORDER_WAIT_AND_RECHECK_SEC)
                if _has_open_orders(client, SYMBOL):
                    ts8b = datetime.now(UTC8).strftime("%H:%M:%S")
                    print(
                        f"  {YELLOW}[防护] [{ts8b}] 重检后仍有挂单，跳过本根K线开平仓，下轮再判{RESET}"
                    )
                    continue
            interval_sec = _interval_seconds(INTERVAL_4H)
            last_closed_bar_close_ms = None
            now = datetime.fromtimestamp(now_ts, tz=timezone.utc)
            now_ms = int(now_ts * 1000)

            # 1) 拉 K 线（失败则每 5s 重试，避免漏K线）
            need_len = params.DECISION_LEN + 10
            df_4h = _fetch_klines_with_retry(
                client, SYMBOL, INTERVAL_4H, need_len, now_ms + 60000
            )
            if len(df_4h) < params.DECISION_LEN + 1:
                print(f"{YELLOW}K 线不足，等待下一轮{RESET}")
                continue

            # 仅用已收盘的 K 线（close_time <= now）
            df_4h_closed = df_4h[df_4h["close_time"] <= now_ms].copy()
            # 去重：同一 open_time 只保留一条，避免重复K线
            df_4h_closed = (
                df_4h_closed.drop_duplicates(subset=["open_time"], keep="last")
                .sort_values("open_time")
                .reset_index(drop=True)
            )
            last_closed_bar_close_ms = (
                df_4h_closed.iloc[-1]["close_time"] if len(df_4h_closed) > 0 else None
            )
            if len(df_4h_closed) < params.DECISION_LEN + 1:
                continue

            # 跳K线修正：若上一根已处理时间为 last_4h_close_time，检查是否缺中间根，缺则拉取并合并
            interval_ms = interval_sec * 1000
            if last_4h_close_time is not None:
                expected_close_sec = last_4h_close_time + interval_sec
                current_last_close_ms = df_4h_closed.iloc[-1]["close_time"]
                current_last_close_sec = current_last_close_ms / 1000.0
                while expected_close_sec < current_last_close_sec - 0.001:
                    expected_close_ms = int(expected_close_sec * 1000)
                    while True:
                        try:
                            row = _fetch_single_bar_for_time(
                                client,
                                SYMBOL,
                                INTERVAL_4H,
                                expected_close_ms,
                                interval_sec,
                            )
                            if row is not None:
                                new_df = pd.DataFrame([row.to_dict()])
                                df_4h_closed = (
                                    pd.concat([df_4h_closed, new_df], ignore_index=True)
                                    .drop_duplicates(subset=["open_time"], keep="last")
                                    .sort_values("open_time")
                                    .reset_index(drop=True)
                                )
                                ts8 = datetime.now(UTC8).strftime("%H:%M:%S")
                                print(
                                    f"  {CYAN}[{ts8}] 已补漏K线 close_time={expected_close_sec}{RESET}"
                                )
                                break
                        except Exception as e:
                            ts8 = datetime.now(UTC8).strftime("%H:%M:%S")
                            print(
                                f"  {YELLOW}[{ts8}] 补漏K线失败，{RETRY_SLEEP_SEC}s 后重试: {e}{RESET}"
                            )
                            time.sleep(RETRY_SLEEP_SEC)
                    expected_close_sec += interval_sec

            last_closed_bar_close_ms = df_4h_closed.iloc[-1]["close_time"]
            engine.df_4h = df_4h_closed
            # 最后一根已收盘的 K 线作为 current_record（已去重，不会把 K_n-1 当 K_n）
            current_record = df_4h_closed.iloc[-1]
            current_time = current_record["datetime"]
            current_price = current_record["close"]
            i = len(df_4h_closed) - 1
            klines = df_4h_closed.iloc[i - params.DECISION_LEN : i + 1]
            volume_klines = klines.tail(params.VOLUME_DECISION_LEN)

            # 2) 同步账户
            balance = _get_margin_balance(client)
            pos_amt, entry_price = _get_position_risk(client, SYMBOL)
            engine.balance = balance

            if pos_amt != 0:
                direction = "long" if pos_amt > 0 else "short"
                engine.position = Position(
                    direction, entry_price, abs(pos_amt), current_time
                )
                engine.position.calculate_margin(params.MARGIN_LEVEL)
            else:
                engine.position = None

            # 余额过低告警（每轮只发一次）
            BALANCE_LOW_THRESHOLD = 100.0
            if engine.balance < BALANCE_LOW_THRESHOLD and not balance_low_alerted:
                try:
                    alert_mgr.alert_balance_low(engine.balance, BALANCE_LOW_THRESHOLD)
                    balance_low_alerted = True
                except Exception:
                    pass

            # 用当前最新价做止损检查
            price_for_sl = _get_current_price(client, SYMBOL)
            # 先保存交易所持仓，以便止损触发时发平仓单
            pos_amt_before, _ = _get_position_risk(client, SYMBOL)

            # 有头寸时预先算好止损相关数据（每轮都用于 check_stop_loss；打印仅在有新 K 时做一次，见下方）
            if engine.position is not None:
                fp = engine.position.calculate_profit(price_for_sl)
                total_bal = engine.balance + fp
                max_tb = max(engine.tb_list) if engine.tb_list else total_bal
                # FP 止损：浮动亏损 >= 保证金 * STOPLOSS_FACTOR 时触发
                if engine.position.direction == "long":
                    fp_stop_price = (
                        engine.position.entry_price
                        - engine.position.margin
                        / (params.STOPLOSS_FACTOR * engine.position.amount)
                    )
                else:
                    fp_stop_price = (
                        engine.position.entry_price
                        + engine.position.margin
                        / (params.STOPLOSS_FACTOR * engine.position.amount)
                    )
                tb_threshold = max_tb * params.TB_LOST_LIMIT
                if engine.position.direction == "long":
                    tb_stop_price = (
                        engine.position.entry_price
                        + (tb_threshold - engine.balance) / engine.position.amount
                    )
                else:
                    tb_stop_price = (
                        engine.position.entry_price
                        - (tb_threshold - engine.balance) / engine.position.amount
                    )
                # 仅在本轮回会判断新 K 线时打印一次 [止损]，避免无新 K 时每轮刷屏
                current_close = current_record["close_time"] / 1000.0
                is_new_bar = (
                    last_4h_close_time is None
                    or abs(last_4h_close_time - current_close) >= 0.001
                )
                if is_new_bar:
                    print(
                        f"  [止损] 头寸={engine.position.direction} amount={engine.position.amount:.3f} "
                        f"保证金={engine.position.margin:.2f} 浮动盈亏={fp:.2f} "
                        f"现价={price_for_sl:.2f} FP止损价={fp_stop_price:.2f} TB止损价={tb_stop_price:.2f}"
                    )

            # 3) 止损检查（逻辑与回测一致，触发则市价平仓）
            if engine.check_stop_loss(price_for_sl, now):
                # 重复止损防护：发单前再次确认仓位，若已为 0 则不再发平仓单（避免多单被误平两次变成空头）
                pos_now, _ = _get_position_risk(client, SYMBOL)
                if pos_now == 0:
                    print(f"  {YELLOW}[防护] 仓位已为 0，跳过重复平仓{RESET}")
                    engine.position = None
                    engine.balance = _get_margin_balance(client)
                    continue
                if _has_open_orders(client, SYMBOL):
                    print(
                        f"  {YELLOW}[防护] 当前已有挂单，跳过止损平仓，下轮再检{RESET}"
                    )
                    continue
                if pos_amt_before != 0:
                    sl_profit = engine.position.calculate_profit(price_for_sl)
                    sl_direction = engine.position.direction
                    tb_at_trigger = engine.balance + sl_profit
                    margin_at_trigger = _get_position_margin(client, SYMBOL, params.MARGIN_LEVEL)
                    ts8_sl = datetime.now(UTC8).strftime("%H:%M:%S")
                    trigger_msg = (
                        f"现价={price_for_sl:.2f} 当前TB={tb_at_trigger:.2f} FP={sl_profit:.2f} "
                        f"占用margin={margin_at_trigger:.2f} 触发止损"
                    )
                    print(f"  [止损检查] [{ts8_sl}] {trigger_msg}")
                    try:
                        alert_mgr.send_log("stop_check", trigger_msg)
                    except Exception:
                        pass
                    side = "SELL" if pos_amt_before > 0 else "BUY"
                    qty = _round_quantity(abs(pos_amt_before))
                    if not dry_run:
                        try:
                            client.new_order(
                                symbol=SYMBOL,
                                side=side,
                                type="MARKET",
                                quantity=qty,
                                reduceOnly="true",
                            )
                            print(f"{GREEN}已市价平仓 {side} {qty} {SYMBOL}{RESET}")
                        except Exception as e:
                            print(f"{RED}平仓失败: {e}{RESET}")
                    else:
                        print(f"{MAGENTA}[DRY-RUN] 将平仓 {side} {qty} {SYMBOL}{RESET}")
                    try:
                        alert_mgr.alert_stop_loss(
                            sl_direction, price_for_sl, sl_profit, "止损触发",
                            total_balance=tb_at_trigger, margin=margin_at_trigger,
                        )
                    except Exception:
                        pass
                engine.balance = _get_margin_balance(client)
                engine.position = None
                last_4h_close_time = None
                continue

            # 4) 每根K线只跑一次开仓信号（查重：用 close_time 判断是否已处理，避免重复K线被当成本根）
            current_close = current_record["close_time"] / 1000.0
            if (
                last_4h_close_time is not None
                and abs(last_4h_close_time - current_close) < 0.001
            ):
                continue
            last_4h_close_time = current_close
            # 本根K线最多 1 次平仓 + 1 次开仓，防止重复下单
            orders_placed_this_bar = 0
            _log_kline(current_record, alert_mgr)
            # 本根K线形态与位置判断结果（便于对照为何无/有信号）
            is_max = engine.current_is_max(klines, current_record)
            is_min = engine.current_is_min(klines, current_record)
            up_shadow = engine.is_up_shadow_record(current_record)
            down_shadow = engine.is_down_shadow_record(current_record)
            vol_max = engine.current_volume_is_max(volume_klines, current_record)

            # 5) 开仓信号（与回测同一套逻辑）
            fp_before_action = engine.fp_list[-1] if engine.fp_list else 0
            cft = engine.count_fp_trend()
            if cft == -3:
                cft = cft - 3
            elif cft >= 1:
                cft = cft + 2

            signal_direction = None
            amount = 0.0

            if engine.current_is_max(
                klines, current_record
            ) and engine.is_up_shadow_record(current_record):
                if engine.position and engine.position.direction == "long":
                    if fp_before_action > 0:
                        engine.sl_tp_list.append(1)
                    elif fp_before_action < 0:
                        engine.sl_tp_list.append(-1)
                signal_direction = "short"
                amount = engine.calculate_order_amount(current_price, cft)

            elif engine.current_is_min(
                klines, current_record
            ) and engine.is_down_shadow_record(current_record):
                if engine.position and engine.position.direction == "short":
                    if fp_before_action > 0:
                        engine.sl_tp_list.append(1)
                    elif fp_before_action < 0:
                        engine.sl_tp_list.append(-1)
                signal_direction = "long"
                amount = engine.calculate_order_amount(current_price, cft)

            if i > 0:
                prev_record = df_4h_closed.iloc[i - 1]
                if engine.current_is_max(klines, prev_record):
                    if current_record["close"] < current_record[
                        "open"
                    ] and engine.current_volume_is_max(volume_klines, current_record):
                        if engine.position and engine.position.direction == "long":
                            if fp_before_action > 0:
                                engine.sl_tp_list.append(1)
                            elif fp_before_action < 0:
                                engine.sl_tp_list.append(-1)
                        signal_direction = "short"
                        amount = engine.calculate_order_amount(current_price, cft)

                if engine.current_is_min(klines, prev_record):
                    if current_record["close"] > current_record[
                        "open"
                    ] and engine.current_volume_is_max(volume_klines, current_record):
                        if engine.position and engine.position.direction == "short":
                            if fp_before_action > 0:
                                engine.sl_tp_list.append(1)
                            elif fp_before_action < 0:
                                engine.sl_tp_list.append(-1)
                        signal_direction = "long"
                        amount = engine.calculate_order_amount(current_price, cft)

            # 判断结果（含是否符合开仓），print 且推送
            conform = "符合开仓" if (signal_direction and amount > 0) else "不符合开仓"
            judge_msg = (
                f"{conform} 是否最高点={is_max} 是否上影线={up_shadow} 是否最低点={is_min} "
                f"是否下影线={down_shadow} 是否量最大={vol_max}"
            )
            print(f"  [判断] {judge_msg}")
            try:
                alert_mgr.send_log("judge", judge_msg)
            except Exception:
                pass

            if signal_direction and amount > 0:
                # 用于「操作后」推送：是否本笔为反手、平仓时的 fp/方向/数量
                did_reverse = False
                close_fp = 0.0
                close_dir = None
                close_amt = 0.0
                # 若有现有持仓则先平仓（换仓），且本根K线未超下单次数
                if pos_amt_before != 0 and orders_placed_this_bar < 2:
                    if _has_open_orders(client, SYMBOL):
                        print(
                            f"  {YELLOW}[防护] 当前已有挂单，跳过本根K线开平仓，下轮再判{RESET}"
                        )
                        last_4h_close_time = None
                        continue
                    did_reverse = True
                    close_fp = engine.position.calculate_profit(current_price)
                    close_dir = "long" if pos_amt_before > 0 else "short"
                    close_amt = abs(pos_amt_before)
                    close_side = "SELL" if pos_amt_before > 0 else "BUY"
                    close_qty = _round_quantity(abs(pos_amt_before))
                    if not dry_run:
                        try:
                            client.new_order(
                                symbol=SYMBOL,
                                side=close_side,
                                type="MARKET",
                                quantity=close_qty,
                                reduceOnly="true",
                            )
                            print(
                                f"{GREEN}已市价平仓（换仓） {close_side} {close_qty} {SYMBOL}{RESET}"
                            )
                            orders_placed_this_bar += 1
                        except Exception as e:
                            print(f"{RED}平仓失败: {e}{RESET}")
                            last_4h_close_time = None  # 下轮重新判断本根K线并重试换仓
                            continue
                    else:
                        print(
                            f"{MAGENTA}[DRY-RUN] 将平仓 {close_side} {close_qty} {SYMBOL}{RESET}"
                        )
                        orders_placed_this_bar += 1
                    time.sleep(1)

                side = "BUY" if signal_direction == "long" else "SELL"
                qty = _round_quantity(amount)
                if qty <= 0:
                    continue
                did_open = False
                if orders_placed_this_bar < 2:
                    if _has_open_orders(client, SYMBOL):
                        print(
                            f"  {YELLOW}[防护] 当前已有挂单，跳过开仓，下轮再判{RESET}"
                        )
                        last_4h_close_time = None
                        continue
                    if not dry_run:
                        try:
                            client.new_order(
                                symbol=SYMBOL,
                                side=side,
                                type="MARKET",
                                quantity=qty,
                            )
                            print(
                                f"{GREEN}已市价开仓 {side} {qty} {SYMBOL} @ ~{current_price:.2f}{RESET}"
                            )
                            orders_placed_this_bar += 1
                            did_open = True
                        except Exception as e:
                            print(f"{RED}开仓失败: {e}{RESET}")
                    else:
                        print(
                            f"{MAGENTA}[DRY-RUN] 将开仓 {side} {qty} {SYMBOL} @ ~{current_price:.2f}{RESET}"
                        )
                        orders_placed_this_bar += 1
                        did_open = True
                else:
                    print(
                        f"  {YELLOW}[防护] 本根K线已下单 {orders_placed_this_bar} 次，不再开仓{RESET}"
                    )

                # 仅在实际发开仓单（或 dry-run 开仓）时更新本地持仓并发开仓告警，并推送操作结果
                if did_open:
                    try:
                        alert_mgr.alert_open_position(signal_direction, current_price, qty)
                    except Exception:
                        pass
                    engine.open_position(
                        signal_direction, current_price, qty, current_time
                    )
                    # 操作后：等待成交后查询持仓与权益，print 且推送
                    if not dry_run:
                        time.sleep(POST_OPEN_QUERY_DELAY_SEC)
                        pos_amt_after, entry_after = _get_position_risk(client, SYMBOL)
                        balance_after = _get_margin_balance(client)
                        if pos_amt_after != 0:
                            amt_after = abs(pos_amt_after)
                            margin_after = _get_position_margin(client, SYMBOL, params.MARGIN_LEVEL)
                        else:
                            margin_after = 0.0
                            amt_after = 0.0
                        if did_reverse:
                            close_label = "平多" if close_dir == "long" else "平空"
                            open_label = "开多" if signal_direction == "long" else "开空"
                            pos_result_msg = (
                                f"反手-{close_label}{open_label}. "
                                f"{close_label} amount={close_amt:.3f}, 平仓 fp={close_fp:.2f}. "
                                f"{open_label} amount={amt_after:.3f}, 当前 tb={balance_after:.2f}, margin={margin_after:.2f}"
                            )
                        else:
                            open_label = "开多" if signal_direction == "long" else "开空"
                            pos_result_msg = (
                                f"{open_label}. {open_label} amount={amt_after:.3f}, "
                                f"当前 tb={balance_after:.2f}, margin={margin_after:.2f}"
                            )
                        print(f"  [操作结果] {pos_result_msg}")
                        try:
                            alert_mgr.send_log("position_result", pos_result_msg)
                        except Exception:
                            pass
                    else:
                        if did_reverse:
                            close_label = "平多" if close_dir == "long" else "平空"
                            open_label = "开多" if signal_direction == "long" else "开空"
                            pos_result_msg = (
                                f"[DRY-RUN] 反手-{close_label}{open_label}. "
                                f"{close_label} amount={close_amt:.3f}, 平仓 fp={close_fp:.2f}. "
                                f"{open_label} amount={qty:.3f} (模拟，未实盘查询 tb/margin)"
                            )
                        else:
                            open_label = "开多" if signal_direction == "long" else "开空"
                            pos_result_msg = (
                                f"[DRY-RUN] {open_label}. {open_label} amount={qty:.3f} "
                                f"(模拟，未实盘查询 tb/margin)"
                            )
                        print(f"  [操作结果] {pos_result_msg}")
                        try:
                            alert_mgr.send_log("position_result", pos_result_msg)
                        except Exception:
                            pass
            else:
                # 本根K线检查完成且未触发开仓，打一条心跳便于确认脚本在运行（东八区时间）
                ts = now.astimezone(UTC8).strftime("%H:%M:%S")
                print(f"  [{ts}] 已检查最新K线，无开平仓信号")

        except KeyboardInterrupt:
            print(f"\n{YELLOW}用户中断{RESET}\n")
            try:
                alert_mgr.alert_shutdown("用户中断")
            except Exception:
                pass
            break
        except Exception as e:
            print(f"{RED}本轮异常: {e}{RESET}")
            err_str = str(e).lower()
            if "timeout" in err_str or "connection" in err_str or "connect" in err_str:
                try:
                    alert_mgr.alert_connection_error(e)
                except Exception:
                    pass
            time.sleep(60)

        # 正常结束开平仓轮后直接进入下一轮，由循环头统一调度

    # 若因 Telegram 指令退出：先确认已读该指令，再发停止告警（避免重启后再次读到同一条 /stop）
    if stop_requested:
        try:
            alert_mgr.ack_updates(last_telegram_update_id)
        except Exception:
            pass
        try:
            alert_mgr.alert_shutdown("用户通过 Telegram 指令停止")
        except Exception:
            pass


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Shadow Power Build - binance 实盘/测试网"
    )
    parser.add_argument(
        "--mainnet", action="store_true", help="使用实盘（默认使用测试网）"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="只打印下单逻辑，不真实下单"
    )
    args = parser.parse_args()
    testnet = not args.mainnet
    run_live(testnet=testnet, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
