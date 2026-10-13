from carry.config import CarryConfig
from carry.decision import decide

P = CarryConfig()  # window 9, enter 6%, exit 2%, min_hold 9


def test_opens_when_trailing_funding_high():
    assert decide([0.0001] * 9, False, 0, P) == "open"          # 10.95% APR


def test_holds_when_below_entry():
    assert decide([0.00005] * 9, False, 0, P) == "hold"         # 5.5% APR


def test_holds_with_short_history():
    assert decide([0.0001] * 5, False, 0, P) == "hold"


def test_closes_when_faded_after_min_hold():
    assert decide([0.0] * 9, True, 9, P) == "close"


def test_min_hold_delays_close():
    assert decide([0.0] * 9, True, 3, P) == "hold"


def test_hysteresis_band_holds_position():
    assert decide([0.00003] * 9, True, 20, P) == "hold"         # 3.3% APR, between exit and enter


def test_no_double_open():
    assert decide([0.0001] * 9, True, 20, P) == "hold"
