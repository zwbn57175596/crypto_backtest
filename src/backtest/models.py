from dataclasses import dataclass, field


@dataclass
class Bar:
    symbol: str
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    interval: str


@dataclass
class Order:
    id: str
    symbol: str
    side: str  # "buy" | "sell"
    type: str  # "market" | "limit"
    quantity: float
    price: float | None = None
    status: str = "pending"
    filled_price: float = 0.0
    filled_at: int = 0
    commission: float = 0.0


@dataclass
class Position:
    symbol: str
    side: str  # "long" | "short"
    quantity: float
    entry_price: float
    leverage: int
    unrealized_pnl: float = 0.0
    margin: float = 0.0


@dataclass
class Trade:
    id: str
    order_id: str
    symbol: str
    side: str
    price: float
    quantity: float
    pnl: float
    commission: float
    timestamp: int
