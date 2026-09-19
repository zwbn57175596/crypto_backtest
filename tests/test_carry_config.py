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


def test_open_retry_defaults():
    c = CarryConfig()
    assert c.open_cooldown_ms == 1_800_000 and c.max_open_fails == 3


def test_rejects_negative_open_cooldown():
    with pytest.raises(ValueError):
        CarryConfig(open_cooldown_ms=-1)


def test_rejects_nonpositive_max_open_fails():
    with pytest.raises(ValueError):
        CarryConfig(max_open_fails=0)


def test_rejects_empty_coins():
    with pytest.raises(ValueError):
        CarryConfig(coins=())


def test_rejects_bad_window():
    with pytest.raises(ValueError):
        CarryConfig(window=0)


def test_rejects_negative_min_hold():
    with pytest.raises(ValueError):
        CarryConfig(min_hold=-1)


def test_settle_sec_default_and_validation():
    assert CarryConfig().settle_sec == 0.0
    with pytest.raises(ValueError):
        CarryConfig(settle_sec=-1.0)
