from datetime import datetime, timezone

from scripts.edge_monitor import compute_stability


def _row(checked_at: str, symbols: list[str] | None) -> dict:
    row = {"checked_at": checked_at, "signal": "GO"}
    if symbols is not None:
        row["tradeable_symbols"] = symbols
    return row


def test_streak_counts_consecutive_days_symbol_stayed_on_list():
    rows = [
        _row("2026-08-20T10:00:00+00:00", ["ALGOUSDT"]),
        _row("2026-08-21T10:00:00+00:00", ["ALGOUSDT"]),
        _row("2026-08-22T10:00:00+00:00", ["ALGOUSDT"]),
    ]
    now = datetime(2026, 8, 22, 12, 0, tzinfo=timezone.utc)

    result = compute_stability(rows, now)

    assert result["symbol_streaks"]["ALGOUSDT"] == 3


def test_streak_resets_on_gap_day():
    rows = [
        _row("2026-08-20T10:00:00+00:00", ["ALGOUSDT"]),
        # 2026-08-21 has no data at all (gap)
        _row("2026-08-22T10:00:00+00:00", ["ALGOUSDT"]),
    ]
    now = datetime(2026, 8, 22, 12, 0, tzinfo=timezone.utc)

    result = compute_stability(rows, now)

    assert result["symbol_streaks"]["ALGOUSDT"] == 1


def test_old_format_rows_without_tradeable_symbols_do_not_count_as_a_day():
    rows = [
        _row("2026-08-19T10:00:00+00:00", None),  # pre-migration row, no field
        _row("2026-08-20T10:00:00+00:00", ["ALGOUSDT"]),
        _row("2026-08-21T10:00:00+00:00", ["ALGOUSDT"]),
    ]
    now = datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc)

    result = compute_stability(rows, now)

    assert result["symbol_streaks"]["ALGOUSDT"] == 2


def test_multiple_runs_same_day_use_latest_snapshot():
    rows = [
        _row("2026-08-22T01:00:00+00:00", ["ALGOUSDT", "XLMUSDT"]),
        _row("2026-08-22T09:00:00+00:00", ["ALGOUSDT"]),  # XLMUSDT dropped later same day
    ]
    now = datetime(2026, 8, 22, 12, 0, tzinfo=timezone.utc)

    result = compute_stability(rows, now)

    assert "XLMUSDT" not in result["symbol_streaks"]
    assert result["symbol_streaks"]["ALGOUSDT"] == 1


def test_symbol_streaks_only_includes_symbols_currently_tradeable():
    rows = [
        _row("2026-08-20T10:00:00+00:00", ["ALGOUSDT"]),
        _row("2026-08-21T10:00:00+00:00", ["ALGOUSDT"]),
        _row("2026-08-22T10:00:00+00:00", []),  # ALGOUSDT dropped today
    ]
    now = datetime(2026, 8, 22, 12, 0, tzinfo=timezone.utc)

    result = compute_stability(rows, now)

    assert "ALGOUSDT" not in result["symbol_streaks"]


def test_week_over_week_reports_added_and_dropped_symbols():
    rows = [
        _row("2026-08-15T10:00:00+00:00", ["ALGOUSDT", "XLMUSDT"]),
        _row("2026-08-22T10:00:00+00:00", ["ALGOUSDT", "INJUSDT"]),
    ]
    now = datetime(2026, 8, 22, 12, 0, tzinfo=timezone.utc)

    result = compute_stability(rows, now)

    wow = result["week_over_week"]
    assert wow["available"] is True
    assert wow["overlap"] == 1
    assert wow["added"] == ["INJUSDT"]
    assert wow["dropped"] == ["XLMUSDT"]


def test_week_over_week_unavailable_when_no_data_seven_days_ago():
    rows = [
        _row("2026-08-20T10:00:00+00:00", ["ALGOUSDT"]),
        _row("2026-08-22T10:00:00+00:00", ["ALGOUSDT"]),
    ]
    now = datetime(2026, 8, 22, 12, 0, tzinfo=timezone.utc)

    result = compute_stability(rows, now)

    assert result["week_over_week"]["available"] is False


def test_no_data_returns_empty_result():
    result = compute_stability([], datetime(2026, 8, 22, tzinfo=timezone.utc))

    assert result["as_of_day"] is None
    assert result["symbol_streaks"] == {}
    assert result["week_over_week"]["available"] is False
