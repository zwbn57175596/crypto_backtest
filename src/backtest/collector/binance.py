import httpx
from backtest.collector.base import BaseCollector
from backtest.models import Bar

_BASE_URL = "https://fapi.binance.com"
_LIMIT = 1500


class BinanceCollector(BaseCollector):
    exchange_name = "binance"

    async def fetch(self, symbol: str, interval: str, start_ms: int, end_ms: int) -> None:
        async with httpx.AsyncClient(timeout=30) as client:
            current = start_ms
            while current < end_ms:
                params = {"symbol": symbol, "interval": interval,
                          "startTime": current, "endTime": end_ms, "limit": _LIMIT}
                resp = await client.get(f"{_BASE_URL}/fapi/v1/klines", params=params)
                resp.raise_for_status()
                data = resp.json()
                if not data:
                    break
                bars = [Bar(symbol=symbol, timestamp=int(row[0]), open=float(row[1]),
                           high=float(row[2]), low=float(row[3]), close=float(row[4]),
                           volume=float(row[5]), interval=interval) for row in data]
                self._save_bars(bars)
                current = int(data[-1][0]) + 1
                if len(data) < _LIMIT:
                    break
