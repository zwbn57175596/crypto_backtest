import sqlite3
from collections.abc import Iterator
from backtest.models import Bar


class DataFeed:
    def __init__(self, db_path: str, symbol: str, interval: str, exchange: str,
                 start_ts: int | None = None, end_ts: int | None = None):
        self.db_path = db_path
        self.symbol = symbol
        self.interval = interval
        self.exchange = exchange
        self.start_ts = start_ts
        self.end_ts = end_ts

    def __iter__(self) -> Iterator[Bar]:
        conn = sqlite3.connect(self.db_path)
        query = ("SELECT symbol, interval, timestamp, open, high, low, close, volume "
                 "FROM klines WHERE symbol = ? AND interval = ? AND exchange = ?")
        params: list = [self.symbol, self.interval, self.exchange]
        if self.start_ts is not None:
            query += " AND timestamp >= ?"
            params.append(self.start_ts)
        if self.end_ts is not None:
            query += " AND timestamp <= ?"
            params.append(self.end_ts)
        query += " ORDER BY timestamp ASC"
        cursor = conn.execute(query, params)
        for row in cursor:
            yield Bar(symbol=row[0], interval=row[1], timestamp=row[2],
                      open=row[3], high=row[4], low=row[5], close=row[6], volume=row[7])
        conn.close()
