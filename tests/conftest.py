import pytest
from backtest.models import Bar, Order, Position, Trade


@pytest.fixture
def sample_bar():
    return Bar(
        symbol="BTCUSDT",
        timestamp=1704067200000,
        open=42000.0,
        high=42500.0,
        low=41800.0,
        close=42300.0,
        volume=1500.0,
        interval="1h",
    )
