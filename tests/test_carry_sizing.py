import math

import pytest

from carry.sizing import ceil_to, drift, floor_to, plan_position
from carry.types import BTC, ETH


def test_floor_and_ceil_handle_float_noise():
    assert floor_to(0.20833333, 0.01) == pytest.approx(0.20)
    assert floor_to(1.4999999999999998, 0.01) == pytest.approx(1.5)
    assert ceil_to(0.015015015015, 1e-8) == pytest.approx(0.01501502)


def test_plan_btc():
    s = plan_position(900.0, 60000.0, BTC, 0.001)
    assert s.contracts == pytest.approx(1.5)
    assert s.coin_qty == pytest.approx(0.015)
    assert s.spot_qty == pytest.approx(0.01501502)
    assert s.notional == pytest.approx(900.0)


def test_plan_eth():
    s = plan_position(300.0, 3000.0, ETH, 0.001)
    assert s.contracts == pytest.approx(1.0)
    assert s.coin_qty == pytest.approx(0.1)
    assert s.spot_qty == pytest.approx(0.100101)


def test_plan_rounds_contracts_down():
    assert plan_position(905.0, 60000.0, BTC, 0.001).contracts == pytest.approx(1.5)


def test_plan_rejects_too_small():
    with pytest.raises(ValueError):
        plan_position(5.0, 60000.0, BTC, 0.001)


def test_drift():
    assert drift(0.01501502, 1.5, BTC) == pytest.approx(0.001, rel=0.02)
    assert drift(0.015, 0.0, BTC) == math.inf
    assert drift(0.0, 0.0, BTC) == 0.0
