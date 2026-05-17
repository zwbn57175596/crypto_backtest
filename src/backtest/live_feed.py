# src/backtest/live_feed.py
import time
from collections.abc import Iterator

from backtest.live_connector import BaseExchangeConnector
from backtest.models import Bar

_INTERVAL_SECONDS: dict[str, int] = {
    "1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
    "1h": 3600, "2h": 7200, "4h": 14400, "6h": 21600,
    "8h": 28800, "12h": 43200, "1d": 86400,
}


def _interval_to_seconds(interval: str) -> int:
    if interval not in _INTERVAL_SECONDS:
        raise ValueError(f"Unsupported interval: {interval!r}. Choose from {list(_INTERVAL_SECONDS)}")
    return _INTERVAL_SECONDS[interval]


def _bar_close_time(interval_sec: int, ref_time: float | None = None) -> float:
    now = ref_time if ref_time is not None else time.time()
    last_close = (int(now) // interval_sec) * interval_sec
    return float(last_close + interval_sec)


def _kline_to_bar(symbol: str, interval: str, k: list) -> Bar:
    return Bar(
        symbol=symbol,
        timestamp=int(k[0]),
        open=float(k[1]),
        high=float(k[2]),
        low=float(k[3]),
        close=float(k[4]),
        volume=float(k[5]),
        interval=interval,
    )


class LiveFeed:
    def __init__(self, connector: BaseExchangeConnector, symbol: str, interval: str,
                 close_buffer_sec: float = 5.0):
        self._connector = connector
        self._symbol = symbol
        self._interval = interval
        self._interval_sec = _interval_to_seconds(interval)
        self._close_buffer_sec = close_buffer_sec
        self._last_bar_ts: int | None = None

    def __iter__(self) -> Iterator[Bar]:
        while True:
            next_close = _bar_close_time(self._interval_sec)
            sleep_until = next_close + self._close_buffer_sec
            now = time.time()
            if sleep_until > now:
                time.sleep(sleep_until - now)

            klines = self._connector.klines(
                symbol=self._symbol,
                interval=self._interval,
                limit=2,
            )
            if len(klines) < 2:
                continue

            closed_bar = _kline_to_bar(self._symbol, self._interval, klines[-2])

            if self._last_bar_ts is not None:
                expected_ts = self._last_bar_ts + self._interval_sec * 1000
                if closed_bar.timestamp > expected_ts:
                    yield from self._backfill(expected_ts, closed_bar.timestamp)

            self._last_bar_ts = closed_bar.timestamp
            yield closed_bar

    def _backfill(self, from_ts: int, to_ts: int) -> Iterator[Bar]:
        klines = self._connector.klines(
            symbol=self._symbol,
            interval=self._interval,
            startTime=from_ts,
            endTime=to_ts - 1,
            limit=1000,
        )
        for k in klines:
            bar = _kline_to_bar(self._symbol, self._interval, k)
            if bar.timestamp < to_ts:
                self._last_bar_ts = bar.timestamp
                yield bar
