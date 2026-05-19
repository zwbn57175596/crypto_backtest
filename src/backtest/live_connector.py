# src/backtest/live_connector.py
import time
from abc import ABC, abstractmethod


def _nonzero_float(s: str | None) -> float | None:
    if s is None:
        return None
    v = float(s)
    return v if v > 0 else None

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
                     quantity: float, price: float | None = None,
                     client_order_id: str | None = None) -> dict:
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
        if since_ms is not None:
            kwargs["startTime"] = since_ms
        raw = _retry(lambda: self._client.get_all_orders(**kwargs))
        return [self._normalize_order(o) for o in raw]

    def fetch_trades(self, symbol: str, since_ms: int | None = None) -> list[dict]:
        kwargs: dict = {"symbol": symbol}
        if since_ms is not None:
            kwargs["startTime"] = since_ms
        raw = _retry(lambda: self._client.get_account_trades(**kwargs))
        return [self._normalize_trade(t) for t in raw]

    def submit_order(self, symbol: str, side: str, type_: str,
                     quantity: float, price: float | None = None,
                     client_order_id: str | None = None) -> dict:
        if type_ == "limit" and price is None:
            raise ValueError("price is required for limit orders")
        params: dict = {
            "symbol": symbol,
            "side": side.upper(),
            "type": "MARKET" if type_ == "market" else "LIMIT",
            "quantity": quantity,
        }
        if type_ == "limit" and price is not None:
            params["price"] = price
            params["timeInForce"] = "GTC"
        if client_order_id is not None:
            params["newClientOrderId"] = client_order_id
        return _retry(lambda: self._client.new_order(**params))

    def query_order(self, symbol: str, order_id: str) -> dict:
        resp = _retry(lambda: self._client.query_order(symbol=symbol, orderId=int(order_id)))
        return self._normalize_order(resp)

    def change_leverage(self, symbol: str, leverage: int) -> None:
        _retry(lambda: self._client.change_leverage(symbol=symbol, leverage=leverage))

    @staticmethod
    def _normalize_order(o: dict) -> dict:
        status = o["status"].lower()
        return {
            "order_id": str(o["orderId"]),
            "client_order_id": o.get("clientOrderId"),
            "symbol": o["symbol"],
            "side": o["side"].lower(),
            "type": "market" if o["type"] == "MARKET" else "limit",
            "quantity": float(o["origQty"]),
            "price": _nonzero_float(o.get("price", "0")),
            "status": status,
            "filled_price": _nonzero_float(o.get("avgPrice", "0")),
            "filled_qty": float(o["executedQty"]),
            "commission": None,  # commission not available at order level; see fetch_trades
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


try:
    import okx.PublicData as _OKXPublic
    import okx.MarketData as _OKXMarket
    import okx.Trade as _OKXTrade
    import okx.Account as _OKXAccount
except ImportError:
    _OKXPublic = _OKXMarket = _OKXTrade = _OKXAccount = None


class OKXConnector(BaseExchangeConnector):
    _INTERVAL_MAP: dict[str, str] = {
        "1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m", "30m": "30m",
        "1h": "1H", "2h": "2H", "4h": "4H", "6h": "6H", "8h": "8H", "12h": "12H",
        "1d": "1D",
    }
    _STATUS_MAP: dict[str, str] = {
        "live": "pending", "partially_filled": "pending",
        "filled": "filled", "canceled": "canceled", "mmp_canceled": "canceled",
    }

    def __init__(self, api_key: str = "", secret: str = "", passphrase: str = "", demo: bool = True):
        if _OKXPublic is None:
            raise ImportError("python-okx is not installed. Run: pip install python-okx")
        flag = "1" if demo else "0"
        self._ct_val: float = 0.01
        self._lot_sz: float = 1.0
        self._public = _OKXPublic.PublicAPI(flag=flag, debug=False)
        self._market = _OKXMarket.MarketAPI(flag=flag, debug=False)
        self._trade = _OKXTrade.TradeAPI(api_key, secret, passphrase, False, flag=flag, debug=False)
        self._account = _OKXAccount.AccountAPI(api_key, secret, passphrase, False, flag=flag, debug=False)

    @staticmethod
    def _inst_id(symbol: str) -> str:
        if symbol.endswith("USDT"):
            return f"{symbol[:-4]}-USDT-SWAP"
        raise ValueError(f"Cannot map {symbol!r} to OKX instId (only USDT perps supported)")

    @classmethod
    def _okx_bar(cls, interval: str) -> str:
        if interval not in cls._INTERVAL_MAP:
            raise ValueError(f"Unsupported interval: {interval!r}")
        return cls._INTERVAL_MAP[interval]

    @property
    def exchange_name(self) -> str:
        return "okx"

    def server_time(self) -> int:
        resp = _retry(lambda: self._public.get_system_time())
        return int(resp["data"][0]["ts"])

    def exchange_info(self, symbol: str) -> dict:
        inst_id = self._inst_id(symbol)
        resp = _retry(lambda: self._public.get_instruments(instType="SWAP", instId=inst_id))
        inst = resp["data"][0]
        self._ct_val = float(inst["ctVal"])
        self._lot_sz = float(inst["lotSz"])
        # Express minimum order as base-asset step for live_exchange._fetch_lot_step
        step_base = self._ct_val * self._lot_sz
        return {
            "filters": [{"filterType": "LOT_SIZE", "stepSize": str(step_base)}],
            "ctVal": self._ct_val,
            "lotSz": self._lot_sz,
        }

    def klines(self, symbol: str, interval: str, **kwargs) -> list:
        inst_id = self._inst_id(symbol)
        bar = self._okx_bar(interval)
        params: dict[str, str | int] = {"instId": inst_id, "bar": bar}
        if "limit" in kwargs:
            params["limit"] = kwargs["limit"]
        # OKX: after = ts > after (exclusive lower bound), before = ts < before
        if "startTime" in kwargs:
            params["after"] = str(int(kwargs["startTime"]) - 1)
        if "endTime" in kwargs:
            params["before"] = str(int(kwargs["endTime"]) + 1)
        resp = _retry(lambda: self._market.get_candlesticks(**params))
        # OKX returns newest-first; reverse to oldest-first (Binance convention)
        bars = list(reversed(resp["data"]))
        # Normalise to [ts, open, high, low, close, vol] matching _kline_to_bar
        return [[int(b[0]), b[1], b[2], b[3], b[4], b[5]] for b in bars]

    def fetch_balance(self) -> float:
        resp = _retry(lambda: self._account.get_account_balance(ccy="USDT"))
        for detail in resp["data"][0]["details"]:
            if detail["ccy"] == "USDT":
                return float(detail.get("availEq") or detail.get("cashBal") or 0)
        return 0.0

    def fetch_position(self, symbol: str) -> dict | None:
        inst_id = self._inst_id(symbol)
        resp = _retry(lambda: self._account.get_positions(instType="SWAP", instId=inst_id))
        for p in resp["data"]:
            pos_contracts = float(p.get("pos", 0))
            if abs(pos_contracts) < 1e-8:
                continue
            base_qty = abs(pos_contracts) * self._ct_val
            return {
                "side": "long" if pos_contracts > 0 else "short",
                "qty": base_qty,
                "entry_price": float(p["avgPx"]) if p.get("avgPx") else 0.0,
                "unrealized_pnl": float(p["upl"]) if p.get("upl") else 0.0,
            }
        return None

    def fetch_mark_price(self, symbol: str) -> float:
        inst_id = self._inst_id(symbol)
        resp = _retry(lambda: self._market.get_mark_price(instType="SWAP", instId=inst_id))
        return float(resp["data"][0]["markPx"])

    def fetch_orders(self, symbol: str, since_ms: int | None = None) -> list[dict]:
        inst_id = self._inst_id(symbol)
        params: dict = {"instType": "SWAP", "instId": inst_id}
        if since_ms is not None:
            params["begin"] = str(since_ms)
        resp = _retry(lambda: self._trade.get_orders_history(**params))
        return [self._normalize_order(o) for o in resp.get("data", [])]

    def fetch_trades(self, symbol: str, since_ms: int | None = None) -> list[dict]:
        inst_id = self._inst_id(symbol)
        params: dict = {"instType": "SWAP", "instId": inst_id}
        if since_ms is not None:
            params["begin"] = str(since_ms)
        resp = _retry(lambda: self._trade.get_fills_history(**params))
        return [self._normalize_trade(t) for t in resp.get("data", [])]

    def submit_order(self, symbol: str, side: str, type_: str,
                     quantity: float, price: float | None = None,
                     client_order_id: str | None = None) -> dict:
        if type_ == "limit" and price is None:
            raise ValueError("price is required for limit orders")
        inst_id = self._inst_id(symbol)
        # quantity is base-asset amount; convert to OKX contracts
        contracts = max(int(self._lot_sz), round(quantity / self._ct_val))
        params: dict = {
            "instId": inst_id,
            "tdMode": "cross",
            "side": side,
            "ordType": "market" if type_ == "market" else "limit",
            "sz": str(contracts),
        }
        if type_ == "limit" and price is not None:
            params["px"] = str(price)
        if client_order_id is not None:
            params["clOrdId"] = client_order_id
        resp = _retry(lambda: self._trade.place_order(**params))
        return {"orderId": resp["data"][0]["ordId"]}

    def query_order(self, symbol: str, order_id: str) -> dict:
        inst_id = self._inst_id(symbol)
        resp = _retry(lambda: self._trade.get_order(instId=inst_id, ordId=order_id))
        return self._normalize_order(resp["data"][0])

    def change_leverage(self, symbol: str, leverage: int) -> None:
        inst_id = self._inst_id(symbol)
        _retry(lambda: self._account.set_leverage(
            instId=inst_id, lever=str(leverage), mgnMode="cross"
        ))

    def _normalize_order(self, o: dict) -> dict:
        status = self._STATUS_MAP.get(o.get("state", ""), "pending")
        sz = float(o.get("sz", 0)) * self._ct_val
        fill_sz = float(o.get("fillSz", 0)) * self._ct_val
        return {
            "order_id": str(o["ordId"]),
            "client_order_id": o.get("clOrdId"),
            "symbol": o.get("instId", ""),
            "side": o.get("side", ""),
            "type": "market" if o.get("ordType") == "market" else "limit",
            "quantity": sz,
            "price": _nonzero_float(o.get("px")),
            "status": status,
            "filled_price": _nonzero_float(o.get("avgPx")),
            "filled_qty": fill_sz,
            "commission": float(o.get("fee", 0)),
            "ts": int(o["cTime"]) if o.get("cTime") else 0,
            "filled_at": int(o["uTime"]) if status == "filled" and o.get("uTime") else None,
        }

    def _normalize_trade(self, t: dict) -> dict:
        qty = float(t.get("fillSz", 0)) * self._ct_val
        price = float(t.get("fillPx", 0))
        return {
            "trade_id": str(t["tradeId"]),
            "order_id": str(t["ordId"]),
            "symbol": t.get("instId", ""),
            "side": t.get("side", ""),
            "price": price,
            "qty": qty,
            "notional": qty * price,
            "commission": abs(float(t.get("fee", 0))),
            "realized_pnl": float(t.get("pnl", 0)),
            "ts": int(t["ts"]),
        }
