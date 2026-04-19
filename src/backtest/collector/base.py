import sqlite3
from abc import ABC, abstractmethod
from backtest.models import Bar


class BaseCollector(ABC):
    exchange_name: str = ""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS klines (
                symbol TEXT, interval TEXT, timestamp INTEGER,
                open REAL, high REAL, low REAL, close REAL,
                volume REAL, exchange TEXT,
                PRIMARY KEY (exchange, symbol, interval, timestamp)
            )
        """)
        conn.commit()
        conn.close()

    def _save_bars(self, bars: list[Bar]) -> None:
        if not bars:
            return
        conn = sqlite3.connect(self.db_path)
        conn.executemany(
            "INSERT OR IGNORE INTO klines VALUES (?,?,?,?,?,?,?,?,?)",
            [(b.symbol, b.interval, b.timestamp, b.open, b.high,
              b.low, b.close, b.volume, self.exchange_name) for b in bars],
        )
        conn.commit()
        conn.close()

    def _get_latest_timestamp(self, symbol: str, interval: str) -> int | None:
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT MAX(timestamp) FROM klines WHERE symbol=? AND interval=? AND exchange=?",
            (symbol, interval, self.exchange_name),
        ).fetchone()
        conn.close()
        return row[0] if row and row[0] is not None else None

    @abstractmethod
    async def fetch(self, symbol: str, interval: str, start_ms: int, end_ms: int) -> None:
        pass
