"""Viability report: does spot-long + perp-short funding carry pay after fees and borrow interest?"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone

from carry.funding_history import load_rates, median_interval_hours
from carry.simulate import CarryParams, simulate_carry

SYMBOLS = {"binance": ("BTCUSDT", "ETHUSDT"), "okx": ("BTC-USDT-SWAP", "ETH-USDT-SWAP")}
FEES = {"taker": 0.0015, "maker": 0.0010}       # per side, both legs
BORROW = (0.05, 0.08, 0.12, 0.20)
HEADER = "| {} | net APR | maxDD | in-mkt | trips | avg lev |\n|---|---|---|---|---|---|"


def _row(label: str, r) -> str:
    return (f"| {label} | {r.net_apr:+.2%} | {r.max_drawdown:.2%} | "
            f"{r.time_in_market:.0%} | {r.round_trips} | {r.avg_leverage:.2f} |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/carry.db")
    ap.add_argument("--exchange", default="binance", choices=list(SYMBOLS))
    ap.add_argument("--enter", type=float, default=0.06)
    ap.add_argument("--exit", type=float, default=0.02)
    ap.add_argument("--lev-spread", type=float, default=0.03)
    args = ap.parse_args()
    base = CarryParams(enter_apr=args.enter, exit_apr=args.exit, lev_min_spread=args.lev_spread)

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
              f"(mean funding APR {sum(rates) / len(rates) * 1095:.2%})")

        print("\n### Full period: fee × borrow APR × max leverage\n")
        print(HEADER.format("fee / borrow / maxL"))
        for fee_name, fee in FEES.items():
            for b in BORROW:
                for lev in (1.0, 1.5):
                    p = replace(base, fee_side=fee, borrow_apr=b, max_leverage=lev)
                    print(_row(f"{fee_name} / {b:.0%} / {lev}", simulate_carry(rates, p)))

        print("\n### Per calendar year (taker fee, borrow 8%, maxL 1.5 vs 1.0)\n")
        print(HEADER.format("year / maxL"))
        by_year: dict[int, list[float]] = {}
        for t, r in rows:
            by_year.setdefault(datetime.fromtimestamp(t / 1000, tz=timezone.utc).year, []).append(r)
        for year, rs in sorted(by_year.items()):
            for lev in (1.0, 1.5):
                p = replace(base, max_leverage=lev)
                print(_row(f"{year} / {lev}", simulate_carry(rs, p)))

        print("\n### Entry-threshold sweep (taker, borrow 8%, maxL 1.5; exit = enter/3)\n")
        print(HEADER.format("enter APR"))
        for e in (0.04, 0.06, 0.08, 0.10):
            p = replace(base, enter_apr=e, exit_apr=e / 3)
            print(_row(f"{e:.0%}", simulate_carry(rates, p)))


if __name__ == "__main__":
    main()
