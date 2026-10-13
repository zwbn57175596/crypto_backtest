"""Viability report: does spot-long + perp-short funding carry pay after fees and borrow interest?"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone

from carry.funding_history import load_rates, median_interval_hours
from carry.funding_math import PERIODS_PER_YEAR
from carry.simulate import CarryParams, simulate_carry

SYMBOLS = {"binance": ("BTCUSDT", "ETHUSDT"), "okx": ("BTC-USDT-SWAP", "ETH-USDT-SWAP")}
FEES = {"taker": 0.0015, "maker": 0.0010}       # per side, both legs
BORROW = (0.05, 0.08, 0.12, 0.20)
HEADER = ("| {} | net APR | maxDD | in-mkt | trips | avg lev | lev flips |\n"
          "|---|---|---|---|---|---|---|")
PARTIAL_TOL_MS = 3 * 86_400_000     # a year is "partial" if it starts/ends >3 days off the boundary
COLD_START_NOTE = ("_Each year is simulated independently with a cold start (entry cost + "
                   "9-settlement warm-up per year), which biases per-year rows down by roughly "
                   "0.35-0.5pp; annualised figures for partial years are noisy._")


def _row(label: str, r) -> str:
    return (f"| {label} | {r.net_apr:+.2%} | {r.max_drawdown:.2%} | "
            f"{r.time_in_market:.0%} | {r.round_trips} | {r.avg_leverage:.2f} | {r.lev_changes} |")


def _is_partial_year(year: int, first_ms: int, last_ms: int) -> bool:
    start = int(datetime(year, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    end = int(datetime(year + 1, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    return first_ms - start > PARTIAL_TOL_MS or end - last_ms > PARTIAL_TOL_MS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/carry.db")
    ap.add_argument("--exchange", default="binance", choices=list(SYMBOLS))
    ap.add_argument("--enter", type=float, default=0.06)
    ap.add_argument("--exit", type=float, default=0.02)
    ap.add_argument("--lev-spread", type=float, default=0.03)
    ap.add_argument("--fee", type=float, default=0.0015, help="fee per side (both legs)")
    ap.add_argument("--borrow", type=float, default=0.08, help="borrow APR")
    args = ap.parse_args()
    base = CarryParams(enter_apr=args.enter, exit_apr=args.exit, lev_min_spread=args.lev_spread,
                       lev_exit_spread=min(CarryParams.lev_exit_spread, args.lev_spread))
    fb = f"fee/side {args.fee:.2%}, borrow {args.borrow:.0%}"

    for sym in SYMBOLS[args.exchange]:
        rows = load_rates(args.db, args.exchange, sym)
        if not rows:
            print(f"\n## {sym}: no data (run scripts/carry_fetch_funding.py)")
            continue
        assert abs(median_interval_hours(rows) - 8.0) < 0.01, "funding interval is not 8h"
        rates = [r for _, r in rows]
        first = datetime.fromtimestamp(rows[0][0] / 1000, tz=timezone.utc).date()
        last = datetime.fromtimestamp(rows[-1][0] / 1000, tz=timezone.utc).date()
        print(f"\n## {args.exchange} {sym}  {first} → {last}  "
              f"(mean funding APR {sum(rates) / len(rates) * PERIODS_PER_YEAR:.2%})")

        print("\n### Full period: fee × borrow APR × max leverage\n")
        print(HEADER.format("fee / borrow / maxL"))
        for fee_name, fee in FEES.items():
            for b in BORROW:
                for lev in (1.0, 1.5):
                    p = replace(base, fee_side=fee, borrow_apr=b, max_leverage=lev)
                    print(_row(f"{fee_name} / {b:.0%} / {lev}", simulate_carry(rates, p)))

        print(f"\n### Per calendar year ({fb}, maxL 1.5 vs 1.0; enter {args.enter:.0%}, "
              f"exit {args.exit:.0%}, lev-spread {args.lev_spread:.1%})\n")
        print(COLD_START_NOTE + "\n")
        print(HEADER.format("year / maxL"))
        by_year: dict[int, list[tuple[int, float]]] = {}
        for t, r in rows:
            by_year.setdefault(datetime.fromtimestamp(t / 1000, tz=timezone.utc).year, []).append((t, r))
        for year, yrows in sorted(by_year.items()):
            label = str(year)
            if _is_partial_year(year, yrows[0][0], yrows[-1][0]):
                label += " (partial)"
            rs = [r for _, r in yrows]
            for lev in (1.0, 1.5):
                p = replace(base, fee_side=args.fee, borrow_apr=args.borrow, max_leverage=lev)
                print(_row(f"{label} / {lev}", simulate_carry(rs, p)))

        print(f"\n### Entry-threshold sweep ({fb}, maxL 1.5, lev-spread {args.lev_spread:.1%}; "
              f"exit = enter/3)\n")
        print(HEADER.format("enter APR"))
        for e in (0.04, 0.06, 0.08, 0.10):
            p = replace(base, enter_apr=e, exit_apr=e / 3, fee_side=args.fee, borrow_apr=args.borrow)
            print(_row(f"{e:.0%}", simulate_carry(rates, p)))


if __name__ == "__main__":
    main()
