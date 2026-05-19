"""One-shot script: sync Binance USDT-M futures trades/orders/position for BTCUSDT."""
import hashlib
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from backtest.live_connector import BinanceConnector, _retry
from backtest.live_history import LiveHistoryDB

KEY_FILE = Path(__file__).parent.parent / "data" / "bn_prod.key"
HISTORY_DB = Path(__file__).parent.parent / "data" / "live_history.db"
SYMBOL = "BTCUSDT"
START_MS = int(datetime(2026, 3, 1, tzinfo=timezone.utc).timestamp() * 1000)
PAGE_LIMIT = 500
WINDOW_MS = 7 * 24 * 3600 * 1000


def load_keys(path: Path) -> tuple[str, str]:
    env = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            env[k.strip()] = v.strip()
    return env["BINANCE_API_KEY"], env["BINANCE_SECRET"]


def _windowed_fetch(fetch_fn, normalize_fn, raw_id_field: str, since_ms: int) -> list[dict]:
    """Fetch records in 7-day windows, deduplicating by raw_id_field."""
    all_records: list[dict] = []
    seen_ids: set[str] = set()
    now_ms = int(time.time() * 1000)
    window_start = since_ms

    while window_start < now_ms:
        window_end = min(window_start + WINDOW_MS, now_ms)
        kwargs: dict = {"startTime": window_start, "endTime": window_end, "limit": PAGE_LIMIT}
        while True:
            raw = _retry(lambda: fetch_fn(**kwargs))
            if not raw:
                break
            new_raw = [r for r in raw if str(r[raw_id_field]) not in seen_ids]
            for r in new_raw:
                seen_ids.add(str(r[raw_id_field]))
            new = [normalize_fn(r) for r in new_raw]
            all_records.extend(new)
            if new:
                w0 = datetime.fromtimestamp(window_start / 1000, tz=timezone.utc).date()
                w1 = datetime.fromtimestamp(window_end / 1000, tz=timezone.utc).date()
                print(f"    {w0} ~ {w1}: +{len(new)} (total: {len(all_records)})")
            if len(raw) < PAGE_LIMIT:
                break
            kwargs = {"fromId": int(raw[-1][raw_id_field]) + 1, "limit": PAGE_LIMIT}
            time.sleep(0.1)
        window_start = window_end
        time.sleep(0.1)

    return all_records


def main() -> None:
    api_key, secret = load_keys(KEY_FILE)
    account_id = hashlib.sha256(api_key.encode()).hexdigest()[:16]
    print(f"account_id : {account_id}")
    print(f"symbol     : {SYMBOL}")
    print(f"since      : {datetime.fromtimestamp(START_MS/1000, tz=timezone.utc).date()}")
    print(f"history_db : {HISTORY_DB}")
    print()

    connector = BinanceConnector(api_key=api_key, secret=secret, testnet=False)
    db = LiveHistoryDB(str(HISTORY_DB))
    now_ms = int(time.time() * 1000)

    # --- trades ---
    print("=== Syncing trades ===")
    trades = _windowed_fetch(
        fetch_fn=lambda **kw: connector._client.get_account_trades(symbol=SYMBOL, **kw),
        normalize_fn=connector._normalize_trade,
        raw_id_field="id",
        since_ms=START_MS,
    )
    if trades:
        db.upsert_trades(account_id, "binance", trades)
        print(f"  => {len(trades)} trades upserted\n")
    else:
        print("  => no trades\n")

    # --- orders ---
    print("=== Syncing orders ===")
    orders = _windowed_fetch(
        fetch_fn=lambda **kw: connector._client.get_all_orders(symbol=SYMBOL, **kw),
        normalize_fn=connector._normalize_order,
        raw_id_field="orderId",
        since_ms=START_MS,
    )
    if orders:
        for o in orders:
            db.upsert_order(account_id, "binance", o)
        print(f"  => {len(orders)} orders upserted\n")
    else:
        print("  => no orders\n")

    # --- current position snapshot ---
    print("=== Syncing current position ===")
    pos_raw = connector.fetch_position(SYMBOL)
    balance = connector.fetch_balance()
    if pos_raw:
        from backtest.models import Position
        pos = Position(
            symbol=SYMBOL,
            side=pos_raw["side"],
            quantity=pos_raw["qty"],
            entry_price=pos_raw["entry_price"],
            leverage=1,
            unrealized_pnl=pos_raw["unrealized_pnl"],
            margin=0.0,
        )
        mark = connector.fetch_mark_price(SYMBOL)
        equity = balance + pos_raw["unrealized_pnl"]
        db.record_position(account_id, "binance", SYMBOL, pos, balance, equity, now_ms)
        print(f"  side={pos_raw['side']} qty={pos_raw['qty']} entry={pos_raw['entry_price']}"
              f" upnl={pos_raw['unrealized_pnl']:.2f} mark={mark}")
    else:
        db.record_position(account_id, "binance", SYMBOL, None, balance, balance, now_ms)
        print(f"  no open position, balance={balance:.2f}")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
