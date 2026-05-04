from unittest.mock import MagicMock, patch
from backtest.live_feed import _interval_to_seconds, _bar_close_time, _kline_to_bar


def _make_kline(ts_ms: int, open_=100.0, high=110.0, low=90.0, close=105.0, vol=10.0) -> list:
    close_ts = ts_ms + 3_599_999
    return [ts_ms, str(open_), str(high), str(low), str(close), str(vol),
            close_ts, "0", 0, "0", "0", "0"]


class TestHelpers:
    def test_interval_to_seconds_1h(self):
        assert _interval_to_seconds("1h") == 3600

    def test_interval_to_seconds_4h(self):
        assert _interval_to_seconds("4h") == 14400

    def test_interval_to_seconds_1m(self):
        assert _interval_to_seconds("1m") == 60

    def test_interval_unsupported_raises(self):
        import pytest
        with pytest.raises(ValueError):
            _interval_to_seconds("99x")

    def test_bar_close_time_aligns_to_interval(self):
        # ref_time at 02:00:00 UTC (7200s), interval=4h (14400s)
        close = _bar_close_time(14400, ref_time=7200.0)
        assert close == 14400  # 04:00:00 UTC

    def test_bar_close_time_when_already_past_close(self):
        # ref_time at 05:00:00 UTC (18000s), interval=4h (14400s)
        # last_close = floor(18000/14400)*14400 = 1*14400 = 14400
        # next_close = 14400 + 14400 = 28800 (08:00 UTC)
        close = _bar_close_time(14400, ref_time=18000.0)
        assert close == 28800

    def test_kline_to_bar_fields(self):
        k = _make_kline(1_700_000_000_000)
        bar = _kline_to_bar("BTCUSDT", "1h", k)
        assert bar.symbol == "BTCUSDT"
        assert bar.interval == "1h"
        assert bar.timestamp == 1_700_000_000_000
        assert bar.open == 100.0
        assert bar.high == 110.0
        assert bar.low == 90.0
        assert bar.close == 105.0
        assert bar.volume == 10.0


class TestLiveFeedIteration:
    def test_yields_second_to_last_kline_as_closed_bar(self):
        from backtest.live_feed import LiveFeed
        ts1 = 1_700_000_000_000
        ts2 = ts1 + 3_600_000
        connector = MagicMock()
        connector.klines.return_value = [_make_kline(ts1), _make_kline(ts2)]
        feed = LiveFeed(connector, "BTCUSDT", "1h", close_buffer_sec=0)
        with patch("backtest.live_feed.time.sleep"), \
             patch("backtest.live_feed.time.time", return_value=float(ts2 // 1000 + 10)):
            bar = next(iter(feed))
        assert bar.timestamp == ts1
        assert bar.symbol == "BTCUSDT"

    def test_updates_last_bar_ts_after_yield(self):
        from backtest.live_feed import LiveFeed
        ts1 = 1_700_000_000_000
        ts2 = ts1 + 3_600_000
        connector = MagicMock()
        connector.klines.return_value = [_make_kline(ts1), _make_kline(ts2)]
        feed = LiveFeed(connector, "BTCUSDT", "1h", close_buffer_sec=0)
        with patch("backtest.live_feed.time.sleep"), \
             patch("backtest.live_feed.time.time", return_value=float(ts2 // 1000 + 10)):
            next(iter(feed))
        assert feed._last_bar_ts == ts1


class TestLiveFeedBackfill:
    def test_backfill_yields_missing_bar(self):
        from backtest.live_feed import LiveFeed
        ts_base = 1_700_000_000_000
        interval_ms = 3_600_000
        ts1, ts2, ts3 = ts_base, ts_base + interval_ms, ts_base + 2 * interval_ms
        connector = MagicMock()
        connector.klines.return_value = [_make_kline(ts2, close=200.0)]
        feed = LiveFeed(connector, "BTCUSDT", "1h", close_buffer_sec=0)
        feed._last_bar_ts = ts1
        bars = list(feed._backfill(ts2, ts3))
        assert len(bars) == 1
        assert bars[0].timestamp == ts2
        assert bars[0].close == 200.0

    def test_backfill_excludes_current_bar_timestamp(self):
        from backtest.live_feed import LiveFeed
        ts_base = 1_700_000_000_000
        interval_ms = 3_600_000
        ts2, ts3 = ts_base + interval_ms, ts_base + 2 * interval_ms
        connector = MagicMock()
        connector.klines.return_value = [_make_kline(ts2), _make_kline(ts3)]
        feed = LiveFeed(connector, "BTCUSDT", "1h", close_buffer_sec=0)
        feed._last_bar_ts = ts_base
        bars = list(feed._backfill(ts2, ts3))
        assert len(bars) == 1
        assert bars[0].timestamp == ts2

    def test_backfill_updates_last_bar_ts(self):
        from backtest.live_feed import LiveFeed
        ts_base = 1_700_000_000_000
        interval_ms = 3_600_000
        ts2, ts3 = ts_base + interval_ms, ts_base + 2 * interval_ms
        connector = MagicMock()
        connector.klines.return_value = [_make_kline(ts2)]
        feed = LiveFeed(connector, "BTCUSDT", "1h", close_buffer_sec=0)
        feed._last_bar_ts = ts_base
        list(feed._backfill(ts2, ts3))
        assert feed._last_bar_ts == ts2

    def test_no_gap_no_backfill_called(self):
        from backtest.live_feed import LiveFeed
        ts1 = 1_700_000_000_000
        ts2 = ts1 + 3_600_000
        connector = MagicMock()
        connector.klines.return_value = [_make_kline(ts1), _make_kline(ts2)]
        feed = LiveFeed(connector, "BTCUSDT", "1h", close_buffer_sec=0)
        feed._last_bar_ts = ts1
        with patch("backtest.live_feed.time.sleep"), \
             patch("backtest.live_feed.time.time", return_value=float(ts2 // 1000 + 10)):
            bar = next(iter(feed))
        assert connector.klines.call_count == 1
        assert bar.timestamp == ts1
