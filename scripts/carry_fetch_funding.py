"""Download BTC/ETH funding history (OKX recent + Binance long history) into data/carry.db."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

import okx.PublicData as PublicData

from carry.funding_history import (
    fetch_binance_history, fetch_okx_history, make_binance_page_fetcher,
    make_okx_page_fetcher, median_interval_hours, save_rates,
)

PAIRS = [("BTC-USDT-SWAP", "BTCUSDT"), ("ETH-USDT-SWAP", "ETHUSDT")]


def _fmt(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/carry.db")
    ap.add_argument("--binance-start", default="2019-09-01")
    args = ap.parse_args()

    start_ms = int(datetime.strptime(args.binance_start, "%Y-%m-%d")
                   .replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    okx_fetch = make_okx_page_fetcher(PublicData.PublicAPI(flag="0", debug=False))
    bn_fetch = make_binance_page_fetcher()

    for okx_id, bn_sym in PAIRS:
        for exch, sym, rows in (
            ("okx", okx_id, fetch_okx_history(okx_fetch, okx_id)),
            ("binance", bn_sym, fetch_binance_history(bn_fetch, bn_sym, start_ms, end_ms)),
        ):
            save_rates(args.db, exch, sym, rows)
            if rows:
                print(f"{exch:8s} {sym:14s} n={len(rows):6d} {_fmt(rows[0][0])} → {_fmt(rows[-1][0])} "
                      f"median_interval={median_interval_hours(rows):.2f}h")
            else:
                print(f"{exch:8s} {sym:14s} NO DATA")


if __name__ == "__main__":
    main()
