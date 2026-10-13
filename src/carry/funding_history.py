"""Funding-rate history: paginated fetchers (network injected) and a SQLite store."""
from __future__ import annotations

import sqlite3
import statistics
import time
from pathlib import Path
from typing import Callable

Rows = list[tuple[int, float]]  # ascending (funding_time_ms, rate)


def fetch_okx_history(fetch_page: Callable[[str, int | None], list[dict]],
                      inst_id: str, max_pages: int = 500) -> Rows:
    """Walk backwards: each page holds records earlier than `after` (newest first)."""
    seen: dict[int, float] = {}
    after: int | None = None
    for _ in range(max_pages):
        page = fetch_page(inst_id, after)
        if not page:
            break
        for r in page:
            seen[int(r["fundingTime"])] = float(r["fundingRate"])
        oldest = min(int(r["fundingTime"]) for r in page)
        if after is not None and oldest >= after:
            break
        after = oldest
    return sorted(seen.items())


def fetch_binance_history(fetch_page: Callable[[str, int], list[dict]], symbol: str,
                          start_ms: int, end_ms: int, max_pages: int = 500) -> Rows:
    """Walk forwards from start_ms; each page is ascending."""
    seen: dict[int, float] = {}
    cursor = start_ms
    for _ in range(max_pages):
        page = fetch_page(symbol, cursor)
        if not page:
            break
        for r in page:
            t = int(r["fundingTime"])
            if t <= end_ms:
                seen[t] = float(r["fundingRate"])
        last = max(int(r["fundingTime"]) for r in page)
        if last >= end_ms:
            break
        cursor = last + 1
    return sorted(seen.items())


def make_okx_page_fetcher(public_api, pause: float = 0.25) -> Callable[[str, int | None], list[dict]]:
    def fetch(inst_id: str, after: int | None) -> list[dict]:
        time.sleep(pause)
        resp = public_api.funding_rate_history(
            instId=inst_id, after=str(after) if after else "", limit="100")
        if resp.get("code") != "0":
            raise RuntimeError(f"OKX funding history error: {resp}")
        return resp["data"]
    return fetch


def make_binance_page_fetcher(pause: float = 0.2) -> Callable[[str, int], list[dict]]:
    import httpx

    def fetch(symbol: str, start_ms: int) -> list[dict]:
        time.sleep(pause)
        r = httpx.get("https://fapi.binance.com/fapi/v1/fundingRate",
                      params={"symbol": symbol, "startTime": start_ms, "limit": 1000}, timeout=20)
        r.raise_for_status()
        return r.json()
    return fetch


def save_rates(db_path: str | Path, exchange: str, symbol: str, rows: Rows) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    with con:
        con.execute("CREATE TABLE IF NOT EXISTS funding_rates ("
                    "exchange TEXT, symbol TEXT, ts INTEGER, rate REAL, "
                    "PRIMARY KEY (exchange, symbol, ts))")
        con.executemany("INSERT OR REPLACE INTO funding_rates VALUES (?,?,?,?)",
                        [(exchange, symbol, t, r) for t, r in rows])
    con.close()


def load_rates(db_path: str | Path, exchange: str, symbol: str) -> Rows:
    if not Path(db_path).exists():
        return []
    con = sqlite3.connect(db_path)
    try:
        cur = con.execute("SELECT ts, rate FROM funding_rates WHERE exchange=? AND symbol=? ORDER BY ts",
                          (exchange, symbol))
        return [(int(t), float(r)) for t, r in cur.fetchall()]
    except sqlite3.OperationalError:
        return []
    finally:
        con.close()


def median_interval_hours(rows: Rows) -> float:
    diffs = [b[0] - a[0] for a, b in zip(rows, rows[1:])]
    return statistics.median(diffs) / 3_600_000 if diffs else 0.0
