import json

import pytest

from carry.cli import Clients, load_config, load_env_file, main
from carry.fake_exchange import FakeExchange
from carry.runner import InstanceLock
from carry.state import CarryState, load_state, save_state
from tests.test_carry_preflight import FakeAccount, FakePublic

HIGH = {"BTC": [0.0001] * 9, "ETH": [0.0001] * 9}


def setup(tmp_path, funding=None, **cfg_over):
    cfg = {"state_path": str(tmp_path / "state.json"), **cfg_over}
    cfg_path = tmp_path / "carry.json"
    cfg_path.write_text(json.dumps(cfg))
    ex = FakeExchange(funding=dict(funding or {}))
    alerts, out = [], []
    kw = dict(make_clients=lambda env, c, a: Clients(ex, FakeAccount(), FakePublic()),
              make_alerter=lambda env, log: alerts.append,
              out=out.append, sleep=lambda s: None)
    argv = ["--config", str(cfg_path)]
    return ex, alerts, out, kw, argv


class FlushAlerter:
    """Records messages and flush() calls."""

    def __init__(self):
        self.msgs, self.flushed = [], 0

    def __call__(self, msg):
        self.msgs.append(msg)

    def flush(self, timeout=20.0):
        self.flushed += 1
        return True


class PendingExchange(FakeExchange):
    """FakeExchange exposing the OkxExchange pending-order helpers."""

    def __init__(self, pending=(), **kw):
        super().__init__(**kw)
        self.pending, self.events = list(pending), []

    def pending_carry_orders(self):
        return list(self.pending)

    def adopt_pending_orders(self):
        self.events.append("adopt")
        return list(self.pending)

    def clear_unconfirmed(self):
        self.events.append("clear")

    def spot_balance(self, coin):
        self.events.append("spot_balance")
        return super().spot_balance(coin)


def test_load_env_file(tmp_path):
    p = tmp_path / "e.env"
    p.write_text("# c\nA=1\nexport B='two'\nC=\"three\"\n\nbad line\nD=a=b\n")
    assert load_env_file(p) == {"A": "1", "B": "two", "C": "three", "D": "a=b"}


def test_load_config_validates(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"coins": ["BTC"], "deploy_usdt": 100.0}))
    c = load_config(p)
    assert c.coins == ("BTC",) and c.deploy_usdt == 100.0
    p.write_text(json.dumps({"bogus": 1}))
    with pytest.raises(ValueError):
        load_config(p)


def test_config_error_exit_code(tmp_path):
    p = tmp_path / "c.json"
    p.write_text("{bad")
    out = []
    assert main(["--config", str(p), "status"], env={}, out=out.append) == 2 and "config error" in out[0]


def test_alert_test_sends_a_message(tmp_path):
    ex, alerts, out, kw, argv = setup(tmp_path)
    assert main(argv + ["alert-test"], env={}, **kw) == 0
    assert any("alert test" in a for a in alerts)


def test_preflight_command_exit_codes(tmp_path):
    ex, alerts, out, kw, argv = setup(tmp_path)
    assert main(argv + ["preflight"], env={}, **kw) == 0
    bad = dict(kw, make_clients=lambda env, c, a: Clients(ex, FakeAccount(acct_lv="2"), FakePublic()))
    assert main(argv + ["preflight"], env={}, **bad) == 1
    assert any("acctLv" in line for line in out)


def test_status_prints_exposure(tmp_path):
    ex, alerts, out, kw, argv = setup(tmp_path)
    assert main(argv + ["status"], env={}, **kw) == 0
    assert "mode=FLAT" in "\n".join(out)


def test_force_open_close_and_rearm_persist_state(tmp_path):
    ex, alerts, out, kw, argv = setup(tmp_path)
    assert main(argv + ["force-open", "BTC", "100"], env={}, **kw) == 0
    assert "BTC" in load_state(tmp_path / "state.json").legs
    assert main(argv + ["force-close", "BTC"], env={}, **kw) == 0
    assert load_state(tmp_path / "state.json").legs == {}
    ex.trigger_adl("BTC", 1.0)                      # nothing open, harmless
    assert main(argv + ["rearm"], env={}, **kw) == 0


def test_force_open_error_returns_1(tmp_path):
    ex, alerts, out, kw, argv = setup(tmp_path)
    assert main(argv + ["force-open", "XRP", "100"], env={}, **kw) == 1


def test_manual_commands_refuse_while_the_service_holds_the_lock(tmp_path):
    ex, alerts, out, kw, argv = setup(tmp_path)
    with InstanceLock(str(tmp_path / "state.json") + ".lock"):
        assert main(argv + ["force-open", "BTC", "100"], env={}, **kw) == 3
    assert "lock" in "\n".join(out)


def test_run_bounded_steps_opens_and_saves_state_and_heartbeats(tmp_path):
    ex, alerts, out, kw, argv = setup(tmp_path, funding=HIGH)
    assert main(argv + ["run", "--max-steps", "3"], env={}, **kw) == 0
    st = load_state(tmp_path / "state.json")
    assert st.mode == "HEDGED" and set(st.legs) == {"BTC", "ETH"}
    assert any(a.startswith("[HEARTBEAT]") for a in alerts)


def test_run_restores_signal_handlers(tmp_path):
    import signal
    before = signal.getsignal(signal.SIGTERM)
    ex, alerts, out, kw, argv = setup(tmp_path)
    main(argv + ["run", "--max-steps", "1"], env={}, **kw)
    assert signal.getsignal(signal.SIGTERM) == before


def test_default_clients_requires_keys():
    from carry.cli import default_clients
    from carry.config import CarryConfig
    with pytest.raises(SystemExit):
        default_clients({}, CarryConfig())


# --- amendment tests ---------------------------------------------------------------

def test_flush_called_after_alert_test_and_bounded_run(tmp_path):
    ex, _, out, kw, argv = setup(tmp_path)
    fa = FlushAlerter()
    kw["make_alerter"] = lambda env, log: fa
    assert main(argv + ["alert-test"], env={}, **kw) == 0
    assert fa.flushed == 1
    assert main(argv + ["run", "--max-steps", "1"], env={}, **kw) == 0
    assert fa.flushed == 2


@pytest.mark.parametrize("cmd", [["preflight"], ["status"], ["force-open", "BTC", "100"],
                                 ["force-close", "BTC"], ["rearm"], ["force-open", "XRP", "100"]])
def test_flush_called_on_every_other_command_path(tmp_path, cmd):
    ex, _, out, kw, argv = setup(tmp_path)
    fa = FlushAlerter()
    kw["make_alerter"] = lambda env, log: fa
    main(argv + cmd, env={}, **kw)
    assert fa.flushed == 1


def test_flush_called_when_lock_is_held_and_when_flush_raises(tmp_path):
    ex, _, out, kw, argv = setup(tmp_path)
    fa = FlushAlerter()
    kw["make_alerter"] = lambda env, log: fa
    with InstanceLock(str(tmp_path / "state.json") + ".lock"):
        assert main(argv + ["rearm"], env={}, **kw) == 3
    assert fa.flushed == 1

    def boom(timeout=20.0):
        raise RuntimeError("flush broke")
    fa.flush = boom
    assert main(argv + ["status"], env={}, **kw) == 0          # a broken flush never changes the exit code


def test_rearm_clears_unconfirmed_orders_when_supported(tmp_path):
    ex = PendingExchange()
    _, alerts, out, kw, argv = setup(tmp_path)
    kw["make_clients"] = lambda env, c, a: Clients(ex, FakeAccount(), FakePublic())
    assert main(argv + ["rearm"], env={}, **kw) == 0
    assert "clear" in ex.events


def test_preflight_reports_each_pending_carry_order(tmp_path):
    ex = PendingExchange(pending=["carryabc", "carrydef"])
    _, alerts, out, kw, argv = setup(tmp_path)
    kw["make_clients"] = lambda env, c, a: Clients(ex, FakeAccount(), FakePublic())
    assert main(argv + ["preflight"], env={}, **kw) == 1
    problems = [line for line in out if line.startswith("PROBLEM:")]
    assert len(problems) == 2 and "carryabc" in problems[0] and "carrydef" in problems[1]
    ex.pending = []
    assert main(argv + ["preflight"], env={}, **kw) == 0


@pytest.mark.parametrize("cmd", [["run", "--max-steps", "1"], ["force-close", "BTC"], ["rearm"],
                                 ["force-open", "BTC", "100"]])
def test_adopt_pending_orders_runs_before_reconcile_on_locked_commands(tmp_path, cmd):
    ex = PendingExchange(pending=["carryabc"])
    _, alerts, out, kw, argv = setup(tmp_path)
    kw["make_clients"] = lambda env, c, a: Clients(ex, FakeAccount(), FakePublic())
    main(argv + cmd, env={}, **kw)
    assert ex.events[0] == "adopt"
    assert "spot_balance" in ex.events and ex.events.index("adopt") < ex.events.index("spot_balance")
    assert any("carryabc" in line for line in out)


def test_status_and_preflight_do_not_adopt(tmp_path):
    ex = PendingExchange(pending=["carryabc"])
    _, alerts, out, kw, argv = setup(tmp_path)
    kw["make_clients"] = lambda env, c, a: Clients(ex, FakeAccount(), FakePublic())
    main(argv + ["status"], env={}, **kw)
    main(argv + ["preflight"], env={}, **kw)
    assert "adopt" not in ex.events


def test_startup_state_is_saved_before_the_engine_runs(tmp_path):
    ex, alerts, out, kw, argv = setup(tmp_path)
    assert main(argv + ["run", "--max-steps", "0"], env={}, **kw) == 0
    assert (tmp_path / "state.json").exists()               # saved immediately, before any step


def test_startup_exchange_failure_starts_halted_persisted_and_places_no_order(tmp_path):
    class Broken(FakeExchange):
        def spot_balance(self, coin):
            raise ConnectionError("exchange down")

        def swap_contracts(self, coin):
            raise ConnectionError("exchange down")

    ex = Broken(funding=dict(HIGH))
    _, alerts, out, kw, argv = setup(tmp_path, funding=HIGH)
    kw["make_clients"] = lambda env, c, a: Clients(ex, FakeAccount(), FakePublic())
    assert main(argv + ["run", "--max-steps", "2"], env={}, **kw) == 0
    assert load_state(tmp_path / "state.json").mode == "HALTED"
    assert any("[CRITICAL] startup check failed (ConnectionError); starting HALTED" in a for a in alerts)
    assert ex.calls == []                                   # no order was placed


def test_startup_halted_state_is_not_overwritten_by_reconcile_success(tmp_path):
    ex, alerts, out, kw, argv = setup(tmp_path, funding=HIGH)
    save_state(tmp_path / "state.json", CarryState(mode="HALTED", halted_reason="earlier"))
    assert main(argv + ["run", "--max-steps", "2"], env={}, **kw) == 0
    assert load_state(tmp_path / "state.json").mode == "HALTED" and ex.calls == []
