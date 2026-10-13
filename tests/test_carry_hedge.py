from carry.hedge import HedgeStatus, check_hedge
from carry.types import BTC


def test_statuses():
    assert check_hedge(0.015, 1.5, BTC, 0.02) == HedgeStatus.OK
    assert check_hedge(0.0152, 1.5, BTC, 0.02) == HedgeStatus.OK
    assert check_hedge(0.015, 1.0, BTC, 0.02) == HedgeStatus.DRIFT
    assert check_hedge(0.015, 0.0, BTC, 0.02) == HedgeStatus.SPOT_ONLY
    assert check_hedge(0.0, 1.5, BTC, 0.02) == HedgeStatus.PERP_ONLY
    assert check_hedge(0.0, 0.0, BTC, 0.02) == HedgeStatus.FLAT


def test_dust_below_minimums_counts_as_flat():
    assert check_hedge(1e-6, 0.005, BTC, 0.02) == HedgeStatus.FLAT
