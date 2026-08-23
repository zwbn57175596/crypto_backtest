#!/usr/bin/env python3
"""Fetch the top-N OKX USDT-perpetual symbols by 24h USD volume and write a
universe file for the edge monitor. Each entry gets a realistic per-side
slippage tier derived from its liquidity rank (more liquid -> tighter).

Output: data/universe.json = [{"symbol","instId","rank","vol_usd","slip_bps"}...]
Symbols are stored in the collector's format (e.g. BTCUSDT) so klines line up.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

TICKERS_URL = "https://www.okx.com/api/v5/market/tickers?instType=SWAP"


def slip_for_rank(rank: int) -> float:
    """Per-side slippage (bps) estimate by liquidity rank."""
    if rank <= 20:
        return 2.0
    if rank <= 60:
        return 3.0
    if rank <= 120:
        return 4.5
    return 6.0


def fetch_top(n: int) -> list[dict]:
    req = urllib.request.Request(TICKERS_URL, headers={"User-Agent": "edge-monitor/1.0"})
    with urllib.request.urlopen(req, timeout=20) as r:
        data = json.loads(r.read())
    if data.get("code") != "0":
        raise RuntimeError(f"OKX tickers error: {data.get('msg')}")
    rows = []
    for t in data["data"]:
        inst = t["instId"]                      # e.g. BTC-USDT-SWAP
        if not inst.endswith("-USDT-SWAP"):
            continue
        last = float(t["last"]) if t["last"] else 0.0
        volccy = float(t["volCcy24h"]) if t["volCcy24h"] else 0.0
        usd = volccy * last                     # volCcy24h is in base coin -> * price = USD notional
        if usd <= 0:
            continue
        symbol = inst.replace("-USDT-SWAP", "") + "USDT"
        rows.append({"symbol": symbol, "instId": inst, "vol_usd": usd})
    rows.sort(key=lambda x: x["vol_usd"], reverse=True)
    top = rows[:n]
    for i, row in enumerate(top, 1):
        row["rank"] = i
        row["slip_bps"] = slip_for_rank(i)
    return top


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch top-N OKX USDT-perp universe by 24h USD volume")
    ap.add_argument("--top", type=int, default=200)
    ap.add_argument("--out", default=str(Path("data") / "universe.json"))
    args = ap.parse_args()

    top = fetch_top(args.top)
    out = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "exchange": "okx",
        "count": len(top),
        "symbols": top,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))

    print(f"top {len(top)} OKX USDT-perps by 24h USD volume -> {args.out}")
    for r in top[:12]:
        print(f"  #{r['rank']:>3}  {r['symbol']:<14} {r['vol_usd']/1e9:8.2f} B USD   slip={r['slip_bps']}bps")
    print(f"  ... (#{top[-1]['rank']} {top[-1]['symbol']} {top[-1]['vol_usd']/1e6:.1f}M USD, slip={top[-1]['slip_bps']}bps)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
