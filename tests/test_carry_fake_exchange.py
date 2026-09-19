import pytest

from carry.fake_exchange import FakeExchange


def test_buy_spot_credits_net_of_fee():
    ex = FakeExchange()
    r = ex.buy_spot("BTC", 0.01)
    assert r.ok and r.filled == 0.01 and r.avg_price == 60000.0
    assert ex.spot_balance("BTC") == pytest.approx(0.01 * 0.999)


def test_swap_open_close_and_reduce_only():
    ex = FakeExchange()
    ex.sell_swap("BTC", 1.5)
    assert ex.swap_contracts("BTC") == 1.5
    r = ex.buy_swap("BTC", 2.0)          # cannot buy more than the short
    assert r.filled == 1.5 and ex.swap_contracts("BTC") == 0.0


def test_sell_spot_capped_at_balance():
    ex = FakeExchange()
    ex.buy_spot("ETH", 1.0)
    r = ex.sell_spot("ETH", 5.0)
    assert r.filled == pytest.approx(0.999) and ex.spot_balance("ETH") == pytest.approx(0.0)


def test_fail_next_counts_down():
    ex = FakeExchange()
    ex.fail_next("sell_swap", 2)
    assert not ex.sell_swap("BTC", 1.0).ok
    assert not ex.sell_swap("BTC", 1.0).ok
    assert ex.sell_swap("BTC", 1.0).ok
    assert ex.swap_contracts("BTC") == 1.0


def test_partial_next_applies_once():
    ex = FakeExchange()
    ex.partial_next("sell_swap", 0.5)
    assert ex.sell_swap("BTC", 2.0).filled == 1.0
    assert ex.sell_swap("BTC", 2.0).filled == 2.0


def test_trigger_adl_shrinks_perp():
    ex = FakeExchange()
    ex.sell_swap("BTC", 1.5)
    ex.trigger_adl("BTC", 0.3)
    assert ex.swap_contracts("BTC") == pytest.approx(1.05)


def test_funding_history_returns_last_n():
    ex = FakeExchange(funding={"BTC": [1, 2, 3, 4]})
    assert ex.funding_history("BTC", 2) == [3, 4]
    assert ex.funding_history("ETH", 2) == []


def test_calls_are_logged():
    ex = FakeExchange()
    ex.buy_spot("BTC", 0.01)
    ex.sell_swap("BTC", 1.0)
    assert [(c[0], c[1]) for c in ex.calls] == [("buy_spot", "BTC"), ("sell_swap", "BTC")]


def test_fault_injection_rejects_unknown_op():
    ex = FakeExchange()
    with pytest.raises(ValueError):
        ex.fail_next("buy_swp")
    with pytest.raises(ValueError):
        ex.partial_next("nope", 0.5)


def test_funding_history_zero_is_empty():
    ex = FakeExchange(funding={"BTC": [0.1, 0.2, 0.3]})
    assert ex.funding_history("BTC", 0) == []
    assert ex.funding_history("BTC", 2) == [0.2, 0.3]
