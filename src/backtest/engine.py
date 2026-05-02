from datetime import datetime, timezone
from backtest.data_feed import DataFeed
from backtest.exchange import SimExchange
from backtest.strategy import BaseStrategy


class BacktestEngine:
    def __init__(
        self, db_path: str, symbol: str, interval: str, exchange: str,
        strategy_class: type[BaseStrategy], balance: float = 10000.0,
        leverage: int = 10, commission_rate: float = 0.0004,
        funding_rate: float = 0.0001, maintenance_margin: float = 0.005,
        start: str | None = None, end: str | None = None, margin_mode: str = "isolated",
    ):
        self.db_path = db_path
        self.symbol = symbol
        self.interval = interval
        self.exchange_name = exchange
        self.strategy_class = strategy_class
        self.balance = balance
        self.leverage = leverage
        self.commission_rate = commission_rate
        self.funding_rate = funding_rate
        self.maintenance_margin = maintenance_margin
        self.margin_mode = margin_mode
        self.start_ts = self._parse_time(start) if start else None
        self.end_ts = self._parse_time(end) if end else None

    @staticmethod
    def _parse_time(s: str) -> int:
        dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)

    def run(self) -> dict:
        sim_exchange = SimExchange(
            balance=self.balance, leverage=self.leverage,
            commission_rate=self.commission_rate, funding_rate=self.funding_rate,
            maintenance_margin=self.maintenance_margin, margin_mode=self.margin_mode,
        )
        strategy = self.strategy_class(exchange=sim_exchange, symbol=self.symbol)
        strategy.on_init()

        feed = DataFeed(
            db_path=self.db_path, symbol=self.symbol, interval=self.interval,
            exchange=self.exchange_name, start_ts=self.start_ts, end_ts=self.end_ts,
        )

        for bar in feed:
            sim_exchange.on_new_bar(bar)
            strategy._push_bar(bar)

        trades = sim_exchange.get_trades()
        return {
            "trades": trades,
            "trades_count": len(trades),
            "equity_curve": sim_exchange.get_equity_curve(),
            "final_equity": sim_exchange.equity,
            "initial_balance": self.balance,
        }
