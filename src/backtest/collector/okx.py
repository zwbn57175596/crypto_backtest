import asyncio
import httpx
from backtest.collector.base import BaseCollector
from backtest.models import Bar

_BASE_URL = "https://www.okx.com"
_LIMIT = 100
_DELAY = 0.3  # seconds between requests to avoid rate limiting


def _symbol_to_inst_id(symbol: str) -> str:
    base = symbol.replace("USDT", "")
    return f"{base}-USDT-SWAP"


class OkxCollector(BaseCollector):
    exchange_name = "okx"

    async def fetch(self, symbol: str, interval: str, start_ms: int, end_ms: int) -> None:
        inst_id = _symbol_to_inst_id(symbol)
        okx_bar = self._convert_interval(interval)
        async with httpx.AsyncClient(timeout=30) as client:
            current = end_ms
            while current > start_ms:
                params = {"instId": inst_id, "bar": okx_bar,
                          "after": str(current), "before": str(start_ms - 1),
                          "limit": str(_LIMIT)}
                for attempt in range(3):
                    resp = await client.get(f"{_BASE_URL}/api/v5/market/history-candles", params=params)
                    if resp.status_code == 429:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    resp.raise_for_status()
                    break
                data = resp.json().get("data", [])
                if not data:
                    break
                bars = [Bar(symbol=symbol, timestamp=int(row[0]), open=float(row[1]),
                           high=float(row[2]), low=float(row[3]), close=float(row[4]),
                           volume=float(row[5]), interval=interval) for row in data]
                self._save_bars(bars)
                current = min(int(row[0]) for row in data) - 1
                if len(data) < _LIMIT:
                    break
                await asyncio.sleep(_DELAY)

    @staticmethod
    def _convert_interval(interval: str) -> str:
        mapping = {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "1H", "4h": "4H", "1d": "1D"}
        return mapping.get(interval, interval)
