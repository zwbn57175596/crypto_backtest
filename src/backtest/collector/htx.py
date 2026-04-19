import httpx
from backtest.collector.base import BaseCollector
from backtest.models import Bar

_BASE_URL = "https://api.hbdm.com"
_LIMIT = 2000


def _symbol_to_contract(symbol: str) -> str:
    base = symbol.replace("USDT", "")
    return f"{base}-USDT"


class HtxCollector(BaseCollector):
    exchange_name = "htx"

    async def fetch(self, symbol: str, interval: str, start_ms: int, end_ms: int) -> None:
        contract = _symbol_to_contract(symbol)
        htx_period = self._convert_interval(interval)
        async with httpx.AsyncClient(timeout=30) as client:
            current = start_ms // 1000
            end_sec = end_ms // 1000
            while current < end_sec:
                params = {"contract_code": contract, "period": htx_period,
                          "from": current, "to": end_sec, "size": _LIMIT}
                resp = await client.get(f"{_BASE_URL}/linear-swap-ex/market/history/kline", params=params)
                resp.raise_for_status()
                data = resp.json().get("data", [])
                if not data:
                    break
                bars = [Bar(symbol=symbol, timestamp=int(row["id"]) * 1000, open=float(row["open"]),
                           high=float(row["high"]), low=float(row["low"]), close=float(row["close"]),
                           volume=float(row["vol"]), interval=interval) for row in data]
                self._save_bars(bars)
                current = max(int(row["id"]) for row in data) + 1
                if len(data) < _LIMIT:
                    break

    @staticmethod
    def _convert_interval(interval: str) -> str:
        mapping = {"1m": "1min", "5m": "5min", "15m": "15min", "1h": "60min", "4h": "4hour", "1d": "1day"}
        return mapping.get(interval, interval)
