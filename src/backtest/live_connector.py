# src/backtest/live_connector.py
import time
from abc import ABC, abstractmethod

try:
    from binance.um_futures import UMFutures
except ImportError:
    UMFutures = None

_TESTNET_URL = "https://testnet.binancefuture.com"
_MAINNET_URL = "https://fapi.binance.com"


def _retry(fn, attempts: int = 3, backoff: float = 2.0):
    last_exc = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if i < attempts - 1:
                time.sleep(backoff * (2 ** i))
    raise last_exc


class BaseExchangeConnector(ABC):
    @property
    @abstractmethod
    def exchange_name(self) -> str: ...

    @abstractmethod
    def server_time(self) -> int: ...

    @abstractmethod
    def exchange_info(self, symbol: str) -> dict: ...

    @abstractmethod
    def klines(self, symbol: str, interval: str, **kwargs) -> list: ...

    @abstractmethod
    def fetch_balance(self) -> float: ...

    @abstractmethod
    def fetch_position(self, symbol: str) -> dict | None:
        """Returns {"side", "qty", "entry_price", "unrealized_pnl"} or None if flat."""
        ...

    @abstractmethod
    def fetch_mark_price(self, symbol: str) -> float: ...

    @abstractmethod
    def fetch_orders(self, symbol: str, since_ms: int | None = None) -> list[dict]:
        """Returns list of normalized order dicts."""
        ...

    @abstractmethod
    def fetch_trades(self, symbol: str, since_ms: int | None = None) -> list[dict]:
        """Returns list of normalized trade dicts."""
        ...

    @abstractmethod
    def submit_order(self, symbol: str, side: str, type_: str,
                     quantity: float, price: float | None = None) -> dict:
        """Returns raw exchange response dict."""
        ...

    @abstractmethod
    def query_order(self, symbol: str, order_id: str) -> dict:
        """Returns normalized order dict with current status."""
        ...

    @abstractmethod
    def change_leverage(self, symbol: str, leverage: int) -> None: ...


class BinanceConnector(BaseExchangeConnector):
    def __init__(self, api_key: str = "", secret: str = "", testnet: bool = True):
        base_url = _TESTNET_URL if testnet else _MAINNET_URL
        probe = UMFutures(base_url=base_url)
        server_ms = probe.time()["serverTime"]
        local_ms = int(time.time() * 1000)
        offset_ms = server_ms - local_ms
        if abs(offset_ms) > 500:
            import binance.api as _binance_api
            print(f"[INFO] Clock skew detected: {offset_ms:+d}ms, applying correction")
            _binance_api.get_timestamp = lambda: int(time.time() * 1000) + offset_ms
        self._client = UMFutures(key=api_key, secret=secret, base_url=base_url)

    @property
    def exchange_name(self) -> str:
        return "binance"

    def server_time(self) -> int:
        return self._client.time()["serverTime"]

    def exchange_info(self, symbol: str) -> dict:
        info = _retry(lambda: self._client.exchange_info())
        for s in info["symbols"]:
            if s["symbol"] == symbol:
                return s
        raise ValueError(f"Symbol {symbol!r} not found in exchange_info")

    def klines(self, symbol: str, interval: str, **kwargs) -> list:
        return _retry(lambda: self._client.klines(symbol=symbol, interval=interval, **kwargs))

    def fetch_balance(self) -> float:
        balances = _retry(lambda: self._client.balance())
        for b in balances:
            if b["asset"] == "USDT":
                return float(b["availableBalance"])
        return 0.0

    def fetch_position(self, symbol: str) -> dict | None:
        positions = _retry(lambda: self._client.get_position_risk(symbol=symbol))
        for p in positions:
            qty = float(p["positionAmt"])
            if abs(qty) < 1e-8:
                continue
            return {
                "side": "long" if qty > 0 else "short",
                "qty": abs(qty),
                "entry_price": float(p["entryPrice"]),
                "unrealized_pnl": float(p["unrealizedProfit"]),
            }
        return None

    def fetch_mark_price(self, symbol: str) -> float:
        result = _retry(lambda: self._client.mark_price(symbol=symbol))
        return float(result["markPrice"])

    def fetch_orders(self, symbol: str, since_ms: int | None = None) -> list[dict]:
        kwargs: dict = {"symbol": symbol}
        if since_ms:
            kwargs["startTime"] = since_ms
        raw = _retry(lambda: self._client.get_all_orders(**kwargs))
        return [self._normalize_order(o) for o in raw]

    def fetch_trades(self, symbol: str, since_ms: int | None = None) -> list[dict]:
        kwargs: dict = {"symbol": symbol}
        if since_ms:
            kwargs["startTime"] = since_ms
        raw = _retry(lambda: self._client.get_account_trades(**kwargs))
        return [self._normalize_trade(t) for t in raw]

    def submit_order(self, symbol: str, side: str, type_: str,
                     quantity: float, price: float | None = None) -> dict:
        params: dict = {
            "symbol": symbol,
            "side": side.upper(),
            "type": "MARKET" if type_ == "market" else "LIMIT",
            "quantity": quantity,
        }
        if type_ == "limit" and price is not None:
            params["price"] = price
            params["timeInForce"] = "GTC"
        return _retry(lambda: self._client.new_order(**params))

    def query_order(self, symbol: str, order_id: str) -> dict:
        resp = _retry(lambda: self._client.query_order(symbol=symbol, orderId=int(order_id)))
        return self._normalize_order(resp)

    def change_leverage(self, symbol: str, leverage: int) -> None:
        _retry(lambda: self._client.change_leverage(symbol=symbol, leverage=leverage))

    @staticmethod
    def _normalize_order(o: dict) -> dict:
        avg = float(o.get("avgPrice", 0))
        status = o["status"].lower()
        return {
            "order_id": str(o["orderId"]),
            "symbol": o["symbol"],
            "side": o["side"].lower(),
            "type": "market" if o["type"] == "MARKET" else "limit",
            "quantity": float(o["origQty"]),
            "price": float(o["price"]) if float(o["price"]) > 0 else None,
            "status": status,
            "filled_price": avg if avg > 0 else None,
            "filled_qty": float(o["executedQty"]),
            "commission": None,
            "ts": int(o["time"]),
            "filled_at": int(o["updateTime"]) if status == "filled" else None,
        }

    @staticmethod
    def _normalize_trade(t: dict) -> dict:
        return {
            "trade_id": str(t["id"]),
            "order_id": str(t["orderId"]),
            "symbol": t["symbol"],
            "side": "buy" if t["buyer"] else "sell",
            "price": float(t["price"]),
            "qty": float(t["qty"]),
            "notional": float(t["quoteQty"]),
            "commission": float(t["commission"]),
            "realized_pnl": float(t.get("realizedPnl", 0)),
            "ts": int(t["time"]),
        }
