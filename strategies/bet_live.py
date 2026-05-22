#!/usr/bin/env python3
"""
BET 组合策略 — Binance USDT 本位永续实盘（经典合约 fapi，非 Portfolio Margin）。

- 周期：1h（可改 bet_config.KLINE_INTERVAL）
- 无 Telegram，仅日志
- 状态：data/bet_state.json（连续计数、策略内部持仓），与交易所持仓通过每根 K 线后调仓对齐
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

UTC8 = timezone(timedelta(hours=8))

from bet_config import (
    ACCOUNT_MAX_POSITION_SYMBOLS,
    ACCOUNT_NEW_OPEN_MARGIN_RATIO_LIMIT,
    AFTER_BAR_CLOSE_SEC,
    DEFAULT_PARAMS,
    DELIST_BLOCK_NEW_OPEN_HOURS,
    KLINE_INTERVAL,
    LOG_PER_SYMBOL_BAR_DETAIL,
    MIN_DELTA_NOTIONAL_USDT,
    RECORD_DAILY_NAV_CSV,
    SYMBOLS,
    SYMBOLS_EXTRA,
    UNIVERSE_EXCLUDE,
    UNIVERSE_REFRESH_SEC,
    UNIVERSE_TOP_N,
    USE_VOLUME_UNIVERSE,
)
from bet_exchange import (
    BinanceUMAdapter,
    ReconcileExhaustedError,
    bulk_verify_positions_vs_engine,
    fetch_last_closed_kline,
    fetch_um_margin_usage_ratio,
    list_um_nonzero_positions,
    reconcile_market_order,
    reconcile_pending_sort_key,
)
from bet_live_config import check_config, get_config
from bet_nav import record_bet_nav_snapshot
from bet_strategy_engine import BetPortfolioEngine, LegPosition, SymbolStrategyState
from bet_universe import (
    MIN_UNIVERSE_REFRESH_SEC,
    build_active_symbol_list,
    fetch_volume_ranked_symbols,
    get_symbol_delivery_map,
    perpetual_trading_usdt_symbols,
    symbols_blocked_new_open_delist_horizon,
)

try:
    from binance.um_futures import UMFutures
except ImportError:
    UMFutures = None

THIS_DIR = Path(__file__).resolve().parent
STATE_PATH = THIS_DIR / "data" / "bet_state.json"
NAV_CSV_PATH = THIS_DIR / "data" / "bet_nav_daily.csv"


def _base_asset(sym: str) -> str:
    if sym.endswith("USDT"):
        return sym[: -len("USDT")] or sym
    return sym


# 与 surge_crash_live 类似：盈利粗体绿 / 亏损粗体红 / 接近零黄（终端且未设 NO_COLOR 时启用）
_ANSI_BOLD = "\033[1m"
_ANSI_DIM = "\033[2m"
_ANSI_GREEN = "\033[92m"
_ANSI_RED = "\033[91m"
_ANSI_YELLOW = "\033[93m"
_ANSI_CYAN = "\033[96m"
_ANSI_RESET = "\033[0m"


def _color_enabled() -> bool:
    if os.environ.get("NO_COLOR", "").strip():
        return False
    try:
        return sys.stdout.isatty()
    except Exception:
        return False


def _fmt_upnl_usdt(fp: float, *, color: bool) -> str:
    text = f"{fp:+.4f}"
    if not color:
        return f"{text} USDT"
    if fp > 1e-6:
        return f"{_ANSI_GREEN}{_ANSI_BOLD}{text}{_ANSI_RESET} USDT"
    if fp < -1e-6:
        return f"{_ANSI_RED}{_ANSI_BOLD}{text}{_ANSI_RESET} USDT"
    return f"{_ANSI_YELLOW}{text}{_ANSI_RESET} USDT"


def _print_hourly_positions_summary(
    adapter: BinanceUMAdapter,
    wallet_usdt: float,
    available_usdt: float,
    ts8: str,
) -> None:
    """每小时批次末尾：汇总当前账户全部非零 U 本位永续仓位（API uPnL，与 surge_crash 汇总行风格相近）。"""
    use_color = _color_enabled()
    rows = list_um_nonzero_positions(adapter)
    if not rows:
        dim = _ANSI_DIM if use_color else ""
        rst = _ANSI_RESET if use_color else ""
        print(
            f"汇总 - [持仓快照] [{ts8}] {dim}无非零 USDT-M 永续仓位{rst} "
            f"钱包≈{wallet_usdt:.2f} 可用≈{available_usdt:.2f}"
        )
        return

    n_long = sum(1 for r in rows if r["position_amt"] > 0)
    n_short = sum(1 for r in rows if r["position_amt"] < 0)
    sum_total = sum(float(r["unrealized_profit"]) for r in rows)
    sum_long = sum(float(r["unrealized_profit"]) for r in rows if r["position_amt"] > 0)
    sum_short = sum(float(r["unrealized_profit"]) for r in rows if r["position_amt"] < 0)

    cyan = _ANSI_CYAN if use_color else ""
    rst = _ANSI_RESET if use_color else ""
    print(
        f"汇总 - [持仓快照] [{ts8}] {cyan}钱包≈{wallet_usdt:.2f} 可用≈{available_usdt:.2f}{rst} "
        f"非零头寸={len(rows)}个 多{n_long} 空{n_short} "
        f"多uPnL={_fmt_upnl_usdt(sum_long, color=use_color)} "
        f"空uPnL={_fmt_upnl_usdt(sum_short, color=use_color)} "
        f"合计uPnL={_fmt_upnl_usdt(sum_total, color=use_color)}"
    )
    for r in rows:
        side = "LONG" if r["position_amt"] > 0 else "SHORT"
        amt = float(r["position_amt"])
        ep = float(r["entry_price"])
        lev = int(r["leverage"])
        up = float(r["unrealized_profit"])
        nom = abs(float(r["notional"] or 0.0))
        nom_txt = f"{nom:.2f}" if nom > 1e-8 else "—"
        print(
            f"  持仓 - [快照] [{ts8}] {r['symbol']} [{side}] "
            f"amt={amt:g} entry={ep:.8g} lev={lev}x 名义≈{nom_txt} USDT "
            f"uPnL={_fmt_upnl_usdt(up, color=use_color)}"
        )


def _fmt_open_judge_line(sym: str, leg: Optional[LegPosition], mark: float, lev: int) -> str:
    """结构化一行：[开仓判断][SYMBOL] …"""
    if leg is None:
        return f"[开仓判断][{sym}] 策略目标: 空仓"
    side = "LONG" if leg.direction == 1 else "SHORT"
    base = _base_asset(sym)
    amt = 0.0
    if mark and mark > 0:
        amt = leg.size * lev / mark
    return (
        f"[开仓判断][{sym}] 策略目标: {side} {base} "
        f"amount≈{amt:.6f} 保证金={leg.size:.4f} USDT entry={leg.entry_price:.6f}"
    )


def _sleep_until_next_interval_close(interval_sec: int, after_open_sec: int):
    """睡到下一个 UTC 周期边界（新 K 线开盘时刻）+ after_open_sec 秒。"""
    now = time.time()
    period = float(interval_sec)
    next_close = (int(now / period) + 1) * period
    wake = next_close + after_open_sec
    delay = max(1.0, wake - now)
    t8 = datetime.now(UTC8).strftime("%Y-%m-%d %H:%M:%S")
    print(
        f"[调度][休眠] {t8} 等待 {delay:.1f}s "
        f"(下一 UTC 边界后 {after_open_sec}s 执行)"
    )
    time.sleep(delay)


def _interval_seconds(s: str) -> int:
    m = {
        "1m": 60,
        "3m": 180,
        "5m": 300,
        "15m": 900,
        "30m": 1800,
        "1h": 3600,
        "2h": 7200,
        "4h": 14400,
    }
    return m.get(s, 3600)


def _fallback_symbols() -> List[str]:
    if USE_VOLUME_UNIVERSE:
        return list(dict.fromkeys(list(SYMBOLS) + list(SYMBOLS_EXTRA)))
    return list(SYMBOLS)


def load_engine(params) -> BetPortfolioEngine:
    fb = _fallback_symbols()
    if not STATE_PATH.is_file():
        return BetPortfolioEngine(fb, params)
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return BetPortfolioEngine.from_dict(data, fb, params)
    except Exception as e:
        print(f"[警告][状态] 加载失败，使用全新状态: {e}")
        return BetPortfolioEngine(fb, params)


def save_engine(engine: BetPortfolioEngine):
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(STATE_PATH.suffix + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(engine.to_dict(), f, indent=2, ensure_ascii=False)
        os.replace(tmp, STATE_PATH)
    except Exception:
        if tmp.is_file():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise


def main():
    parser = argparse.ArgumentParser(description="BET 策略实盘")
    parser.add_argument(
        "--mainnet",
        action="store_true",
        help="使用主网（仍可通过 BINANCE_TESTNET=false 控制；本参数可双重确认）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印将要下的单，不真实下单",
    )
    args = parser.parse_args()

    if UMFutures is None:
        print("[错误][依赖] 请先安装: pip install binance-futures-connector")
        sys.exit(1)

    cfg = get_config()
    if args.mainnet:
        os.environ["BINANCE_TESTNET"] = "false"
        cfg = get_config()

    ok, err = check_config(cfg)
    if not ok:
        print(f"[错误][配置] {err}")
        sys.exit(1)

    params = DEFAULT_PARAMS
    engine = load_engine(params)
    client = UMFutures(
        key=cfg["api_key"],
        secret=cfg["secret"],
        base_url=cfg["base_url"],
    )
    adapter = BinanceUMAdapter(client)

    try:
        w0, a0 = adapter.get_usdt_balances()
        t0 = datetime.now(UTC8).strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"[账户][余额][启动确认] {t0} 钱包USDT≈{w0:.2f} "
            f"可用≈{a0:.2f}（档位 sizing 用钱包余额，与锁仓后的可用解耦）"
        )
    except Exception as e:
        print(f"[错误][账户][启动确认] 读取合约 USDT 余额失败: {e}")
        sys.exit(1)

    exclude_set: Set[str] = set(UNIVERSE_EXCLUDE)
    active_symbols: List[str] = list(_fallback_symbols())
    prev_active_set: Set[str] = set(active_symbols)
    volume_rank_snapshot: List[str] = []
    last_universe_fetch = 0.0
    refresh_sec = max(MIN_UNIVERSE_REFRESH_SEC, int(UNIVERSE_REFRESH_SEC))

    if USE_VOLUME_UNIVERSE:
        try:
            volume_rank_snapshot = fetch_volume_ranked_symbols(
                client, UNIVERSE_TOP_N, exclude_set
            )
            active_symbols = build_active_symbol_list(
                volume_rank_snapshot,
                SYMBOLS_EXTRA,
                exclude_set,
                engine,
                adapter,
                prev_active_set,
            )
            engine.ensure_states_for(active_symbols)
            prev_active_set = set(active_symbols)
            last_universe_fetch = time.time()
            print(
                f"[标的池][初始化] Top{UNIVERSE_TOP_N} 排名内={len(volume_rank_snapshot)} 个 "
                f"本轮交易 {len(active_symbols)} 个标的（含必选/持仓保留）"
            )
        except Exception as e:
            print(f"[警告][标的池] 初始化失败，回退 SYMBOLS+EXTRA: {e}")
            active_symbols = list(_fallback_symbols())
            engine.ensure_states_for(active_symbols)
            prev_active_set = set(active_symbols)
    else:
        engine.ensure_states_for(active_symbols)

    try:
        pos_snapshot = adapter.get_all_position_amts()
    except Exception as e:
        pos_snapshot = {}
        print(f"[警告][对账][启动] 全量持仓拉取失败，将跳过所内有仓检查: {e}")

    for sym in active_symbols:
        try:
            amt = float(pos_snapshot.get(sym, 0.0))
            st = engine.states[sym]
            has_ex = abs(amt) > 1e-8
            has_int = st.position is not None
            if has_ex and not has_int:
                print(
                    f"[警告][对账][{sym}] 交易所有仓但本地 state 空仓，请手平或删 "
                    f"data/bet_state.json 后重启"
                )
            if not has_ex and has_int:
                print(
                    f"[警告][对账][{sym}] 本地 state 有目标仓但交易所无仓，下轮尝试市价对齐"
                )
        except Exception as e:
            print(f"[警告][对账][{sym}] 检查失败: {e}")

    print("[启动] ========== BET 策略实盘 ==========")
    print(f"[启动] USDT-M fapi 环境={'测试网' if cfg['testnet'] else '主网'} {cfg['base_url']}")
    if USE_VOLUME_UNIVERSE:
        print(
            f"[启动][标的池] 动态 24h 成交额 Top {UNIVERSE_TOP_N} "
            f"刷新≤{refresh_sec}s SYMBOLS_EXTRA={SYMBOLS_EXTRA or '[]'}"
        )
    else:
        print(f"[启动][标的池] 静态 {active_symbols}")
    cap_txt = ""
    if getattr(params, "max_margin_usdt_per_leg", 0.0) and params.max_margin_usdt_per_leg > 0:
        cap_txt = f" 单腿保证金上限={params.max_margin_usdt_per_leg:.0f}USDT"
    print(
        f"[启动][参数] K线={KLINE_INTERVAL} 杠杆={params.leverage}x "
        f"连续阈值={params.consecutive_threshold} 乘数={params.position_multiplier} "
        f"盈利K平仓={params.profit_candle_threshold}{cap_txt}"
    )
    print(
        f"[启动][调度] 新K线UTC整点后 {AFTER_BAR_CLOSE_SEC}s 执行 "
        f"DELIST禁新开={DELIST_BLOCK_NEW_OPEN_HOURS}h"
    )
    print(
        f"[启动][账户风控] 最多同时持仓品种={ACCOUNT_MAX_POSITION_SYMBOLS} "
        f"保证占用>{ACCOUNT_NEW_OPEN_MARGIN_RATIO_LIMIT:.0%} 时禁止空仓新开"
    )
    print(f"[启动] dry_run={args.dry_run}")
    print("[启动] ========================================")

    for sym in active_symbols:
        try:
            adapter.set_leverage(sym, params.leverage)
            print(f"[杠杆][{sym}] 已设 {params.leverage}x")
        except Exception as e:
            print(f"[警告][杠杆][{sym}] {e}")

    interval_sec = _interval_seconds(KLINE_INTERVAL)

    while True:
        try:
            _sleep_until_next_interval_close(interval_sec, AFTER_BAR_CLOSE_SEC)

            wallet, available = adapter.get_usdt_balances()
            t8 = datetime.now(UTC8).strftime("%Y-%m-%d %H:%M:%S")
            print(
                f"\n[账户][余额] {t8} 钱包USDT≈{wallet:.2f} "
                f"可用≈{available:.2f}（本根 K 线 apply_kline 档位基数=钱包≈{wallet:.2f}）"
            )

            if RECORD_DAILY_NAV_CSV:
                try:
                    record_bet_nav_snapshot(
                        NAV_CSV_PATH,
                        wallet,
                        available_usdt=available,
                        sapi_key=cfg["api_key"],
                        sapi_secret=cfg["secret"],
                        sapi_mainnet=not cfg["testnet"],
                        master_account_email=cfg.get("nav_master_account_email", ""),
                    )
                except Exception as nav_e:
                    print(f"[警告][净值CSV] {nav_e}")

            # 与 calculate_position_size 传入的「余额」：用钱包 USDT，避免因占用保证金导致可用过低、档位异常偏小
            bar_balance = wallet
            pending: List[Tuple[str, int, float, Optional[tuple]]] = []

            if USE_VOLUME_UNIVERSE and (
                time.time() - last_universe_fetch >= refresh_sec
            ):
                try:
                    volume_rank_snapshot = fetch_volume_ranked_symbols(
                        client, UNIVERSE_TOP_N, exclude_set
                    )
                    active_symbols = build_active_symbol_list(
                        volume_rank_snapshot,
                        SYMBOLS_EXTRA,
                        exclude_set,
                        engine,
                        adapter,
                        prev_active_set,
                    )
                    engine.ensure_states_for(active_symbols)
                    prev_active_set = set(active_symbols)
                    last_universe_fetch = time.time()
                    ts = datetime.now(UTC8).strftime("%H:%M:%S")
                    head = (
                        volume_rank_snapshot[0]
                        if volume_rank_snapshot
                        else "N/A"
                    )
                    print(
                        f"[标的池][刷新] {ts} Top{UNIVERSE_TOP_N} "
                        f"本轮{len(active_symbols)}标的 量第一={head}"
                    )
                    for sym in active_symbols:
                        try:
                            adapter.set_leverage(sym, params.leverage)
                        except Exception as lev_e:
                            print(f"[警告][杠杆][{sym}] 刷新池后设杠杆: {lev_e}")
                except Exception as e:
                    print(f"[警告][标的池] 刷新失败沿用上一轮: {e}")

            symbols_this_bar = active_symbols

            delivery_map = get_symbol_delivery_map(client, max_age_sec=3300)
            now_ms = int(time.time() * 1000)
            delist_no_open = symbols_blocked_new_open_delist_horizon(
                delivery_map, DELIST_BLOCK_NEW_OPEN_HOURS, now_ms
            )
            if delist_no_open and DELIST_BLOCK_NEW_OPEN_HOURS > 0:
                preview = ", ".join(sorted(delist_no_open)[:8])
                more = f"…等共{len(delist_no_open)}个" if len(delist_no_open) > 8 else ""
                print(
                    f"[下架保护] 未来{int(DELIST_BLOCK_NEW_OPEN_HOURS)}h 内交割/禁止新开 "
                    f"{preview}{more}"
                )

            perpetual_ok = perpetual_trading_usdt_symbols(client)
            if not perpetual_ok:
                print(
                    "[警告][标的池] TRADING 永续集合为空，本根 K 线仅按交割窗口禁止新开"
                )
            else:
                non_trading = [
                    s for s in symbols_this_bar if s not in perpetual_ok
                ]
                if non_trading:
                    head = ", ".join(sorted(non_trading)[:8])
                    tail = (
                        f" 等共{len(non_trading)}个"
                        if len(non_trading) > 8
                        else ""
                    )
                    print(f"[下架保护] 非 TRADING 标的禁止新开 {head}{tail}")

            pos_slots: Dict[str, float] = {}
            slot_limit_active = True
            try:
                pos_slots = adapter.get_all_position_amts()
            except Exception as e:
                slot_limit_active = False
                t_r = datetime.now(UTC8).strftime("%H:%M:%S")
                print(
                    f"[警告][账户风控] {t_r} 全量持仓快照失败，"
                    f"本根跳过「最多{ACCOUNT_MAX_POSITION_SYMBOLS}品种」限制: {e}"
                )
            if slot_limit_active:
                ex_ct = sum(1 for v in pos_slots.values() if abs(float(v)) > 1e-8)
                slots_left = max(0, ACCOUNT_MAX_POSITION_SYMBOLS - ex_ct)
                ex_s = str(ex_ct)
                slots_s = str(slots_left)
            else:
                ex_ct = 0
                slots_left = 10**9
                ex_s = "n/a"
                slots_s = "n/a"
            margin_ratio, margin_note = fetch_um_margin_usage_ratio(adapter)
            t_risk = datetime.now(UTC8).strftime("%H:%M:%S")
            if margin_ratio is None:
                print(
                    f"[账户风控] {t_risk} 所内持仓品种={ex_s} "
                    f"本根可新开槽位≈{slots_s} "
                    f"保证占用=n/a ({margin_note})"
                )
            else:
                print(
                    f"[账户风控] {t_risk} 所内持仓品种={ex_s} "
                    f"本根可新开槽位≈{slots_s} "
                    f"保证占用≈{margin_ratio:.2%} ({margin_note})"
                )

            for sym in symbols_this_bar:
                st_snap: Optional[SymbolStrategyState] = None
                try:
                    k = fetch_last_closed_kline(client, sym, KLINE_INTERVAL)
                    if not k:
                        print(f"[跳过][{sym}] 无已收盘K线")
                        continue

                    open_time_ms = int(k[0])
                    o = float(k[1])
                    c = float(k[4])
                    close_time_ms = int(k[6])

                    st = engine.states[sym]
                    if st.last_processed_open_time_ms == open_time_ms:
                        print(f"[跳过][{sym}] open_ms={open_time_ms} 已处理")
                        continue

                    st_snap = copy.deepcopy(st)

                    if LOG_PER_SYMBOL_BAR_DETAIL:
                        dir_txt = "阳" if c > o else "阴" if c < o else "平"
                        print(
                            f"[K线][{sym}] open_ms={open_time_ms} close_ms={close_time_ms} "
                            f"O={o:.6f} C={c:.6f} {dir_txt}"
                        )

                    allow_new_delist = (
                        DELIST_BLOCK_NEW_OPEN_HOURS <= 0
                        or sym not in delist_no_open
                    )
                    allow_new_trading = sym in perpetual_ok if perpetual_ok else True
                    allow_new = allow_new_delist and allow_new_trading

                    was_flat_eng = st.position is None
                    flat_ex = (
                        abs(float(pos_slots.get(sym, 0.0))) <= 1e-8
                        if slot_limit_active
                        else False
                    )
                    needs_new_symbol_slot = was_flat_eng and flat_ex
                    account_allow_new = True
                    if needs_new_symbol_slot:
                        if slot_limit_active and slots_left <= 0:
                            account_allow_new = False
                            if LOG_PER_SYMBOL_BAR_DETAIL:
                                print(
                                    f"[账户风控][{sym}] 禁止新开：已达 "
                                    f"{ACCOUNT_MAX_POSITION_SYMBOLS} 个持仓品种上限"
                                )
                        elif (
                            margin_ratio is not None
                            and margin_ratio
                            > ACCOUNT_NEW_OPEN_MARGIN_RATIO_LIMIT
                        ):
                            account_allow_new = False
                            if LOG_PER_SYMBOL_BAR_DETAIL:
                                print(
                                    f"[账户风控][{sym}] 禁止新开：保证占用 "
                                    f"{margin_ratio:.2%} > "
                                    f"{ACCOUNT_NEW_OPEN_MARGIN_RATIO_LIMIT:.0%}"
                                )
                    allow_new = allow_new and account_allow_new

                    engine.apply_kline(
                        sym,
                        open_time_ms,
                        o,
                        c,
                        bar_balance,
                        allow_new_position=allow_new,
                        bar_interval_ms=_interval_seconds(KLINE_INTERVAL) * 1000,
                    )

                    if (
                        needs_new_symbol_slot
                        and account_allow_new
                        and engine.states[sym].position is not None
                    ):
                        slots_left = max(0, slots_left - 1)

                    mark = adapter.mark_price(sym)
                    leg = engine.get_desired_position(sym)
                    if LOG_PER_SYMBOL_BAR_DETAIL:
                        print(_fmt_open_judge_line(sym, leg, mark, params.leverage))

                    desired: Optional[tuple] = None
                    if leg:
                        desired = (leg.direction, leg.size, leg.entry_price)

                    pending.append((sym, open_time_ms, mark, desired))
                except Exception as sym_e:
                    if st_snap is not None:
                        engine.states[sym] = st_snap
                    t_s = datetime.now(UTC8).strftime("%Y-%m-%d %H:%M:%S")
                    print(f"[警告][循环][{sym}] {t_s} {sym_e}")

            if pending:
                pos_fetch_ok = False
                pos_pre_batch: Dict[str, float] = {}
                try:
                    pos_pre_batch = adapter.get_all_position_amts()
                    pos_fetch_ok = True
                except Exception as e:
                    t_pre = datetime.now(UTC8).strftime("%Y-%m-%d %H:%M:%S")
                    print(
                        f"[警告][调仓] {t_pre} 调仓前全量持仓拉取失败，"
                        f"将 per-symbol 读仓位: {e}"
                    )

                t_batch = datetime.now(UTC8).strftime("%Y-%m-%d %H:%M:%S")
                print(
                    f"[账户][调仓批次] {t_batch} 本小时开仓判断已用上方余额快照；"
                    f"调仓前 positionRisk×1（待对齐 {len(pending)} 标的），"
                    f"动作完成后 bulk 复核×1"
                )

                if pos_fetch_ok:
                    pending.sort(
                        key=lambda it: reconcile_pending_sort_key(
                            it[0],
                            it[3],
                            it[2],
                            params.leverage,
                            float(pos_pre_batch.get(it[0], 0.0)),
                            adapter,
                        )
                    )
                    print(
                        f"[账户][调仓批次] {t_batch} 执行顺序：先全平/减仓/反手，再空仓开仓或加仓"
                    )
                else:
                    print(
                        f"[警告][调仓批次] {t_batch} 未拉到批次持仓快照，"
                        f"调仓顺序为标的循环顺序（未应用先平后开排序）"
                    )

                defer_verify = not args.dry_run

                # 调仓失败不回滚 apply_kline：所内持仓可在下一轮继续对齐到「当前」引擎目标
                # （例如本轮目标 1.1 未达成，下根 K 目标 1.2 时直接从实际仓位对到 1.2）
                for sym, open_time_ms, mark, desired in pending:
                    try:
                        init_pos = (
                            float(pos_pre_batch.get(sym, 0.0))
                            if pos_fetch_ok
                            else None
                        )
                        msg = reconcile_market_order(
                            adapter,
                            sym,
                            params.leverage,
                            desired,
                            mark,
                            MIN_DELTA_NOTIONAL_USDT,
                            args.dry_run,
                            defer_per_order_verify=defer_verify,
                            initial_pos_amt=init_pos,
                        )
                        if msg:
                            print(f"[调仓][{sym}] {msg}")
                    except ReconcileExhaustedError as sym_e:
                        t_s = datetime.now(UTC8).strftime("%Y-%m-%d %H:%M:%S")
                        print(f"[错误][调仓][{sym}] {t_s} {sym_e}")
                    except Exception as sym_e:
                        t_s = datetime.now(UTC8).strftime("%Y-%m-%d %H:%M:%S")
                        print(f"[警告][循环][{sym}] {t_s} {sym_e}")
                    finally:
                        engine.mark_bar_processed(sym, open_time_ms)
                        save_engine(engine)

                if not args.dry_run:
                    t_vf = datetime.now(UTC8).strftime("%Y-%m-%d %H:%M:%S")
                    issues = bulk_verify_positions_vs_engine(
                        adapter,
                        engine,
                        symbols_this_bar,
                        params.leverage,
                    )
                    if issues:
                        for it in issues:
                            print(f"[对账][复核] {t_vf} {it}")
                    else:
                        print(
                            f"[对账][复核] {t_vf} 共 {len(symbols_this_bar)} 个标的 "
                            "策略目标与所内持仓一致（容差内）"
                        )

            t_summary = datetime.now(UTC8).strftime("%Y-%m-%d %H:%M:%S")
            try:
                _print_hourly_positions_summary(adapter, wallet, available, t_summary)
            except Exception as summ_e:
                print(f"[汇总][持仓快照] [{t_summary}] 输出失败: {summ_e}")

        except KeyboardInterrupt:
            print("[关闭] KeyboardInterrupt 保存状态并退出")
            save_engine(engine)
            break
        except Exception as e:
            t8 = datetime.now(UTC8).strftime("%Y-%m-%d %H:%M:%S")
            print(f"[警告][循环] {t8} {e} 10s后重试")
            time.sleep(10)


if __name__ == "__main__":
    main()
