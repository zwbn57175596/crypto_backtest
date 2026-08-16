#!/usr/bin/env python3
"""Edge health monitor (measurement only — never places orders).

Measures the rolling, cost-adjusted, volatility-filtered edge of the
mean-reversion portfolio strategy on fresh market data, and emits a GO / HOLD
signal. Background: the edge was real and profitable across 2021-2024 but has
eroded to ~zero/negative in 2025-2026 even in high-volatility regimes. This
monitor watches for the edge to revive (e.g. as a new inefficiency window
opens) so real capital is only ever considered once the edge is measurably
back — it does not trade, hold positions, or need API keys.

Metric: edge_bps = sum(pnl - fees) / sum(opened notional) * 1e4  -- the
net-of-fee return per dollar traded. Compounding/leverage-invariant, so it
reflects trade quality rather than sizing. Trades are gated to high-vol
regime (BTC 7d realized vol above its ~60d EMA), matching where the edge lives.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import sqlite3
import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from backtest.mean_reversion import MRParams
from backtest.mean_reversion_engine import MeanReversionPortfolioEngine
from backtest.reporter import Reporter

# ── Champion configuration (locked from the walk-forward investigation) ──
STRUCT = dict(threshold=4, tp_pct=0.018, sl_pct=0.003, time_stop_bars=24)
DEFAULT_SYMBOLS = ["ETHUSDT", "DOGEUSDT", "WLDUSDT", "PEPEUSDT", "AAVEUSDT", "UNIUSDT"]
RISK_PCT = 0.001
MAX_POS = 5
OFFSET = 0.001
TTL_BARS = 2
INTERVAL = "1h"
EXCHANGE = "okx"
REGIME_SYMBOL = "BTCUSDT"  # market-wide vol regime driver
VOL_WINDOW = 168           # 7d realized-vol window (hourly bars)
VOL_EMA_SPAN = 1440        # ~60d EMA of realized vol
_HOUR_MS = 3_600_000


# ── Volatility-regime-gated engine (only OPENS in high-vol; exits always run) ──
class VolFilteredEngine(MeanReversionPortfolioEngine):
    regime_ok: dict | None = None
    _cur_ts: int = 0

    def _process_symbol(self, sym, bar, state, sim, newly_submitted):
        self._cur_ts = bar.timestamp
        return super()._process_symbol(sym, bar, state, sim, newly_submitted)

    def _account_allows_new_slot(self, sim, newly_submitted):
        if self.regime_ok is not None and not self.regime_ok.get(self._cur_ts, True):
            return False
        return super()._account_allows_new_slot(sim, newly_submitted)


def build_regime(db_path: str) -> dict[int, bool]:
    """Backward-looking BTC vol regime: 7d realized vol above its ~60d EMA."""
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT timestamp, close FROM klines WHERE exchange=? AND symbol=? AND interval=? "
        "ORDER BY timestamp ASC",
        (EXCHANGE, REGIME_SYMBOL, INTERVAL),
    ).fetchall()
    conn.close()
    regime: dict[int, bool] = {}
    if len(rows) < 2:
        return regime
    alpha = 2 / (VOL_EMA_SPAN + 1)
    win: deque[float] = deque(maxlen=VOL_WINDOW)
    s = s2 = 0.0
    ema: float | None = None
    prev = rows[0][1]
    regime[rows[0][0]] = True
    for ts, close in rows[1:]:
        r = (close / prev - 1.0) if prev else 0.0
        prev = close
        if len(win) == VOL_WINDOW:
            old = win[0]
            s -= old
            s2 -= old * old
        win.append(r)
        s += r
        s2 += r * r
        n = len(win)
        mean = s / n
        var = max(s2 / n - mean * mean, 0.0)
        vol = math.sqrt(var)
        ema = vol if ema is None else alpha * vol + (1 - alpha) * ema
        regime[ts] = vol > ema
    return regime


def _last_ts(db_path: str, symbols: list[str]) -> int | None:
    conn = sqlite3.connect(db_path)
    placeholders = ",".join("?" * len(symbols))
    row = conn.execute(
        f"SELECT MAX(timestamp) FROM klines WHERE exchange=? AND interval=? "
        f"AND symbol IN ({placeholders})",
        [EXCHANGE, INTERVAL, *symbols],
    ).fetchone()
    conn.close()
    return row[0] if row and row[0] else None


def data_freshness(db_path: str, symbols: list[str]) -> dict:
    """Per-symbol last bar time + total bar count (for the dashboard)."""
    conn = sqlite3.connect(db_path)
    out = {}
    for sym in [*symbols, REGIME_SYMBOL]:
        row = conn.execute(
            "SELECT COUNT(*), MAX(timestamp) FROM klines WHERE exchange=? AND symbol=? AND interval=?",
            (EXCHANGE, sym, INTERVAL),
        ).fetchone()
        last = row[1] or 0
        out[sym] = {
            "bars": row[0] or 0,
            "last_bar": datetime.fromtimestamp(last / 1000, tz=timezone.utc).isoformat() if last else None,
        }
    conn.close()
    return out


def collect_fresh(db_path: str, symbols: list[str], overlap_days: int = 3) -> None:
    """Fetch new klines from the last stored bar up to now (public data, no keys)."""
    from backtest.collector import COLLECTORS

    collector_cls = COLLECTORS.get(EXCHANGE)
    if collector_cls is None:
        print(f"[collect] no collector for {EXCHANGE}; skipping", file=sys.stderr)
        return
    collector = collector_cls(db_path)
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    for sym in [*symbols, REGIME_SYMBOL]:
        last = _last_ts(db_path, [sym])
        start_ms = (last - overlap_days * 86_400_000) if last else (now_ms - 400 * 86_400_000)
        try:
            asyncio.run(collector.fetch(sym, INTERVAL, start_ms, now_ms))
        except Exception as e:  # network hiccup on one symbol must not kill the run
            print(f"[collect] {sym}: {e}", file=sys.stderr)


def measure(db_path: str, symbols: list[str], regime: dict, start_ms: int, end_ms: int,
            slip_bps: float) -> dict:
    def ms_to_str(ms: int) -> str:
        return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    s = slip_bps / 1e4
    params = MRParams(risk_pct=RISK_PCT, entry_offset_pct=OFFSET, entry_ttl_bars=TTL_BARS, **STRUCT)
    eng = VolFilteredEngine(
        db_path=db_path, symbols=symbols, interval=INTERVAL, exchange=EXCHANGE, params=params,
        balance=10000.0, leverage=10, max_position_symbols=MAX_POS, margin_ratio_limit=0.8,
        maker_commission_rate=0.0002 + s, taker_commission_rate=0.0005 + s,
        start=ms_to_str(start_ms), end=ms_to_str(end_ms),
    )
    eng.regime_ok = regime
    result = eng.run()
    rep = Reporter.generate(result)
    trades = rep["trades"]
    net = sum(t["pnl"] - t["commission"] for t in trades)
    opened = sum(
        t["quantity"] for t in trades
        if not t["order_id"].startswith("bracket") and t["order_id"] != "liquidation"
        and t["pnl"] == 0.0
    )
    edge_bps = (net / opened * 1e4) if opened > 0 else 0.0
    return {
        "edge_bps": round(edge_bps, 2),
        "win_rate": round(rep["win_rate"], 4),
        "max_drawdown": round(rep["max_drawdown"], 4),
        "sharpe": round(rep["sharpe_ratio"], 2),
        "closing_trades": sum(1 for t in trades if t["pnl"] != 0.0),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Edge health monitor (measurement only, never trades)")
    ap.add_argument("--db", default=str(Path("data") / "klines.db"))
    ap.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS),
                    help="Comma-separated traded universe (default: edge-carrier basket)")
    ap.add_argument("--windows-days", default="30,60,90",
                    help="Trailing windows to measure, comma-separated (default 30,60,90)")
    ap.add_argument("--slippage-bps", type=float, default=5.0,
                    help="Realistic per-side slippage in bps (default 5, realistic for alts)")
    ap.add_argument("--go-threshold-bps", type=float, default=2.0,
                    help="90d edge at/above this (and 60d>0) => GO signal (default 2.0)")
    ap.add_argument("--min-trades", type=int, default=300,
                    help="Min closing trades in the longest window for a GO (else WATCH); "
                         "guards against small-sample false positives (default 300)")
    ap.add_argument("--no-collect", action="store_true", help="Skip fetching fresh klines")
    ap.add_argument("--status-file", default=str(Path("data") / "edge_monitor_status.json"))
    ap.add_argument("--history-file", default=str(Path("data") / "edge_monitor_history.jsonl"),
                    help="Append-only JSONL log of every run (for the dashboard trend)")
    ap.add_argument("--json", action="store_true", help="Print machine-readable JSON to stdout too")
    args = ap.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    windows = sorted(int(w) for w in args.windows_days.split(",") if w.strip())

    if not args.no_collect:
        collect_fresh(args.db, symbols)

    end_ms = _last_ts(args.db, symbols)
    if end_ms is None:
        print("ERROR: no kline data found; run collection first.", file=sys.stderr)
        return 2

    regime = build_regime(args.db)
    per_window = {}
    for d in windows:
        per_window[d] = measure(args.db, symbols, regime, end_ms - d * 86_400_000, end_ms,
                                args.slippage_bps)

    long_w = windows[-1]
    mid_w = windows[len(windows) // 2]
    e_long = per_window[long_w]["edge_bps"]
    e_mid = per_window[mid_w]["edge_bps"]
    n_long = per_window[long_w]["closing_trades"]
    edge_positive = e_long >= args.go_threshold_bps and e_mid > 0
    if edge_positive and n_long >= args.min_trades:
        signal = "GO"       # edge back AND on enough data -> worth human re-evaluation
    elif edge_positive:
        signal = "WATCH"    # promising but too few trades to trust -> keep watching, don't act
    else:
        signal = "HOLD"     # edge still eroded

    status = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "data_through": datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc).isoformat(),
        "config": {"struct": STRUCT, "symbols": symbols, "slippage_bps": args.slippage_bps,
                   "go_threshold_bps": args.go_threshold_bps, "min_trades": args.min_trades,
                   "interval": INTERVAL, "regime": "BTC 7d realized vol > ~60d EMA"},
        "windows": per_window,
        "data_freshness": data_freshness(args.db, symbols),
        "signal": signal,
    }
    Path(args.status_file).parent.mkdir(parents=True, exist_ok=True)
    Path(args.status_file).write_text(json.dumps(status, indent=2))

    # append compact record to the history log (drives the dashboard trend chart)
    hist = {"checked_at": status["checked_at"], "data_through": status["data_through"],
            "signal": signal, "n_long": n_long}
    for d in windows:
        hist[f"edge_{d}d"] = per_window[d]["edge_bps"]
    hist_path = Path(args.history_file)
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    with hist_path.open("a") as f:
        f.write(json.dumps(hist) + "\n")

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"=== Edge health check @ {stamp} (data through "
          f"{status['data_through'][:10]}) ===")
    print(f"  universe: {','.join(symbols)}   slippage: {args.slippage_bps:.0f}bps/side (vol-filtered)")
    for d in windows:
        w = per_window[d]
        print(f"  trailing {d:>3}d: edge={w['edge_bps']:+6.2f} bps  win={w['win_rate']:.0%}  "
              f"maxDD={w['max_drawdown']:.0%}  sharpe={w['sharpe']:+.2f}  trades={w['closing_trades']}")
    reason = {
        "GO": "edge revived on sufficient data — human re-evaluation warranted",
        "WATCH": f"edge positive but only {n_long} trades (<{args.min_trades}) — keep watching, do NOT act",
        "HOLD": "edge still eroded — do NOT deploy",
    }[signal]
    print(f"  SIGNAL: {signal}  ({reason})")
    if args.json:
        print(json.dumps(status))
    # exit code: 0 = HOLD (nominal), 5 = WATCH, 10 = GO (revival, for alerts); 2 = error
    return {"HOLD": 0, "WATCH": 5, "GO": 10}[signal]


if __name__ == "__main__":
    sys.exit(main())
