import pytest

from carry.config import CarryConfig


def test_defaults_are_valid():
    c = CarryConfig()
    assert c.per_coin_notional == pytest.approx(125.0)
    assert c.exit_apr < c.enter_apr


def test_rejects_inverted_apr_thresholds():
    with pytest.raises(ValueError):
        CarryConfig(enter_apr=0.02, exit_apr=0.06)


def test_rejects_bad_mgn_thresholds():
    with pytest.raises(ValueError):
        CarryConfig(mgn_warn=2.0, mgn_close=2.5, mgn_emergency=1.5)


def test_rejects_notional_above_cap():
    with pytest.raises(ValueError):
        CarryConfig(deploy_usdt=1000.0, max_order_notional=300.0)


def test_rejects_nonpositive_deploy():
    with pytest.raises(ValueError):
        CarryConfig(deploy_usdt=0.0)
