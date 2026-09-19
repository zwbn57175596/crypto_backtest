from carry.funding_history import (
    fetch_okx_history, fetch_binance_history, save_rates, load_rates, median_interval_hours,
)

H8 = 8 * 3600 * 1000


def _okx_fake(total: int):
    data = [{"fundingTime": str(1_700_000_000_000 + i * H8), "fundingRate": "0.0001"}
            for i in range(total)]
    data.sort(key=lambda r: -int(r["fundingTime"]))         # newest first, like OKX

    def fetch(inst_id, after):
        rows = [r for r in data if after is None or int(r["fundingTime"]) < after]
        return rows[:100]
    return fetch


def test_okx_pagination_collects_everything_ascending_unique():
    rows = fetch_okx_history(_okx_fake(250), "BTC-USDT-SWAP")
    assert len(rows) == 250
    assert [t for t, _ in rows] == sorted({t for t, _ in rows})
    assert rows[0][0] == 1_700_000_000_000


def test_binance_pagination_respects_end():
    start = 1_700_000_000_000
    data = [{"fundingTime": start + i * H8, "fundingRate": "0.0002"} for i in range(2500)]

    def fetch(symbol, cursor):
        return [r for r in data if r["fundingTime"] >= cursor][:1000]

    end = start + 1999 * H8
    rows = fetch_binance_history(fetch, "BTCUSDT", start, end)
    assert len(rows) == 2000 and rows[-1][0] == end


def test_sqlite_roundtrip_and_upsert(tmp_path):
    db = tmp_path / "sub" / "carry.db"
    save_rates(db, "okx", "BTC-USDT-SWAP", [(1, 0.1), (2, 0.2)])
    save_rates(db, "okx", "BTC-USDT-SWAP", [(2, 0.5), (3, 0.3)])
    assert load_rates(db, "okx", "BTC-USDT-SWAP") == [(1, 0.1), (2, 0.5), (3, 0.3)]
    assert load_rates(db, "binance", "BTCUSDT") == []


def test_median_interval_hours():
    rows = [(i * H8, 0.0) for i in range(10)]
    assert median_interval_hours(rows) == 8.0
