from carry.config import CarryConfig
from carry.risk import Tier, classify

C = CarryConfig()  # warn 4.0, close 2.5, emergency 1.5


def test_tiers():
    assert classify(5.0, True, C) == Tier.NORMAL
    assert classify(3.0, True, C) == Tier.WARN
    assert classify(2.5, True, C) == Tier.CLOSE
    assert classify(1.4, True, C) == Tier.EMERGENCY


def test_missing_ratio_with_position_is_warn():
    assert classify(None, True, C) == Tier.WARN


def test_no_position_is_normal_whatever_the_ratio():
    assert classify(None, False, C) == Tier.NORMAL
    assert classify(0.1, False, C) == Tier.NORMAL


def test_unit_divisor_for_percent_form():
    c = CarryConfig(mgn_unit=100.0)
    assert classify(250.0, True, c) == Tier.CLOSE
    assert classify(450.0, True, c) == Tier.NORMAL


def test_tier_ordering():
    assert Tier.EMERGENCY > Tier.CLOSE > Tier.WARN > Tier.NORMAL
