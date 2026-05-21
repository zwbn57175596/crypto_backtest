import sqlite3
from collections.abc import Iterator
from backtest.models import Bar


class MultiSymbolDataFeed:
    """
    Yields (timestamp, {symbol: Bar}) groups aligned across N symbols.
    Uses the union of all timestamps — symbols missing at a given timestamp
    are simply absent from the dict for that step.
    """

    def __init__(
        self,
        db_path: str,
        symbols: list[str],
        interval: str,
        exchange: str,
        start_ts: int | None = None,
        end_ts: int | None = None,
    ):
        self.db_path = db_path
        self.symbols = symbols
        self.interval = interval
        self.exchange = exchange
        self.start_ts = start_ts
        self.end_ts = end_ts

    def __iter__(self) -> Iterator[tuple[int, dict[str, Bar]]]:
        conn = sqlite3.connect(self.db_path)
        placeholders = ",".join("?" * len(self.symbols))
        query = (
            f"SELECT symbol, interval, timestamp, open, high, low, close, volume "
            f"FROM klines "
            f"WHERE symbol IN ({placeholders}) AND interval = ? AND exchange = ?"
        )
        params: list = list(self.symbols) + [self.interval, self.exchange]
        if self.start_ts is not None:
            query += " AND timestamp >= ?"
            params.append(self.start_ts)
        if self.end_ts is not None:
            query += " AND timestamp <= ?"
            params.append(self.end_ts)
        query += " ORDER BY timestamp ASC, symbol ASC"

        current_ts: int | None = None
        current_bars: dict[str, Bar] = {}

        for row in conn.execute(query, params):
            bar = Bar(
                symbol=row[0], interval=row[1], timestamp=row[2],
                open=row[3], high=row[4], low=row[5], close=row[6], volume=row[7],
            )
            if bar.timestamp != current_ts:
                if current_ts is not None:
                    yield current_ts, current_bars
                current_ts = bar.timestamp
                current_bars = {}
            current_bars[bar.symbol] = bar

        if current_ts is not None:
            yield current_ts, current_bars

        conn.close()
