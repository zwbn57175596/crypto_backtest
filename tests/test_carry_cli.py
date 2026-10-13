import json

import pytest

from carry.cli import Clients, _build_ping, load_config, load_env_file, main
from carry.fake_exchange import FakeExchange
from carry.runner import InstanceLock
from carry.state import CarryState, load_state, save_state
from tests.test_carry_preflight import FakeAccount, FakePublic

HIGH = {"BTC": [0.0001] * 9, "ETH": [0.0001] * 9}


def setup(tmp_path, funding=None, **cfg_over):
    cfg = {"state_path": str(tmp_path / "state.json"), "settle_sec": 0.0,
           # deliberately bypasses the C1 calibration gate (the gate has its own tests); no real sleeping
           "mgn_calibrated": True, **cfg_over}
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


# --- fix round 1 -------------------------------------------------------------------

import sys
import types

import httpx

from carry.cli import default_clients
from carry.config import CarryConfig


def _clients_with_mode(ex, mode):
    return lambda env, c, a: Clients(ex, FakeAccount(), FakePublic(), mode=mode)


# I1: force-close with no recorded leg never trades holdings it has no record of
@pytest.mark.parametrize("holding", [("spot", 0.01), ("swap", 5.0)])
def test_force_close_without_recorded_leg_refuses_when_exchange_holds_something(tmp_path, holding):
    ex, alerts, out, kw, argv = setup(tmp_path)
    getattr(ex, holding[0])["BTC"] = holding[1]
    assert main(argv + ["force-close", "BTC"], env={}, **kw) == 1
    line = next(x for x in out if x.startswith("PROBLEM:"))
    assert "no recorded leg for BTC" in line and "OKX UI" in line and "rearm" in line
    assert ex.calls == []                                   # nothing was traded


def test_force_close_without_recorded_leg_and_flat_exchange_is_a_noop_success(tmp_path):
    ex, alerts, out, kw, argv = setup(tmp_path)
    assert main(argv + ["force-close", "BTC"], env={}, **kw) == 0
    assert "force-close BTC: nothing to close" in out and ex.calls == []


def test_force_close_unknown_coin_is_an_error(tmp_path):
    ex, alerts, out, kw, argv = setup(tmp_path)
    assert main(argv + ["force-close", "XRP"], env={}, **kw) == 1
    assert any(line.startswith("error:") for line in out)


# I2: demo vs live is always visible
@pytest.mark.parametrize("cmd", [["preflight"], ["status"], ["force-open", "BTC", "100"],
                                 ["force-close", "BTC"], ["rearm"], ["run", "--max-steps", "1"]])
@pytest.mark.parametrize("mode", ["DEMO", "LIVE"])
def test_every_command_prints_okx_mode_first(tmp_path, cmd, mode):
    ex, alerts, out, kw, argv = setup(tmp_path)
    kw["make_clients"] = _clients_with_mode(ex, mode)
    main(argv + cmd, env={}, **kw)
    assert out[0] == f"okx mode: {mode}"


def test_clients_default_mode_is_live():
    assert Clients(None, None, None).mode == "LIVE"


def test_alert_test_prints_okx_mode_from_env(tmp_path):
    ex, alerts, out, kw, argv = setup(tmp_path)
    assert main(argv + ["alert-test"], env={"OKX_DEMO": "1"}, **kw) == 0
    assert out[0] == "okx mode: DEMO"
    out.clear()
    assert main(argv + ["alert-test"], env={}, **kw) == 0
    assert out[0] == "okx mode: LIVE"


def test_run_startup_alert_and_heartbeat_carry_the_mode(tmp_path):
    ex, alerts, out, kw, argv = setup(tmp_path)
    kw["make_clients"] = _clients_with_mode(ex, "DEMO")
    assert main(argv + ["run", "--max-steps", "1"], env={}, **kw) == 0
    assert any(a.startswith("[INFO] carry starting (okx mode: DEMO") for a in alerts)
    assert any(a.startswith("[HEARTBEAT]\nokx=DEMO mode=") for a in alerts)


# I4: relative state_path would defeat the lock
def test_relative_state_path_in_config_file_is_a_config_error(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"state_path": "live_state/x.json"}))
    out = []
    assert main(["--config", str(p), "status"], env={}, out=out.append) == 2
    assert out[0].startswith("config error:") and "absolute" in out[0]
    with pytest.raises(ValueError):
        load_config(p)
    p.write_text(json.dumps({"state_path": str(tmp_path / "x.json")}))
    assert load_config(p).state_path == str(tmp_path / "x.json")


def test_default_relative_state_path_lock_is_resolved_to_an_absolute_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ex = FakeExchange()
    out = []
    kw = dict(make_clients=lambda env, c, a: Clients(ex, FakeAccount(), FakePublic()),
              make_alerter=lambda env, log: (lambda m: None), out=out.append, sleep=lambda s: None)
    with InstanceLock(str(tmp_path / "live_state" / "carry_state.json") + ".lock"):
        assert main(["run", "--max-steps", "1"], env={}, **kw) == 3
    assert "lock" in "\n".join(out)


def test_run_refuses_while_another_process_holds_the_lock(tmp_path):
    ex, alerts, out, kw, argv = setup(tmp_path)
    with InstanceLock(str(tmp_path / "state.json") + ".lock"):
        assert main(argv + ["run", "--max-steps", "1"], env={}, **kw) == 3
    assert not any(a.startswith("[HEARTBEAT]") for a in alerts)


# minors
def test_startup_save_failure_alerts_and_continues(tmp_path, monkeypatch):
    def boom(path, state):
        raise OSError("disk full")
    monkeypatch.setattr("carry.cli.save_state", boom)
    ex, alerts, out, kw, argv = setup(tmp_path)
    assert main(argv + ["run", "--max-steps", "1"], env={}, **kw) == 0
    assert any("[CRITICAL] could not persist startup state" in a for a in alerts)


def test_client_creation_systemexit_prints_message_returns_2_and_flushes(tmp_path):
    ex, _, out, kw, argv = setup(tmp_path)
    fa = FlushAlerter()

    def refuse(env, c, a):
        raise SystemExit("OKX_API_KEY must be set")
    kw.update(make_alerter=lambda env, log: fa, make_clients=refuse)
    assert main(argv + ["preflight"], env={}, **kw) == 2
    assert "OKX_API_KEY must be set" in "\n".join(out) and fa.flushed == 1


def test_status_never_mutates_a_corrupt_state_file(tmp_path):
    ex, alerts, out, kw, argv = setup(tmp_path)
    p = tmp_path / "state.json"
    p.write_text("{bad")
    assert main(argv + ["status"], env={}, **kw) == 0
    assert p.read_text() == "{bad" and [f.name for f in tmp_path.iterdir() if "corrupt" in f.name] == []
    text = "\n".join(out)
    assert "state file unreadable (JSONDecodeError) — showing HALTED" in text and "mode=HALTED" in text
    assert alerts == []


def test_load_env_file_bom_crlf_hash_in_value_export_and_equals(tmp_path):
    p = tmp_path / "e.env"
    p.write_bytes("﻿# c\r\nA=1\r\nexport B=two\r\nC=pa#ss\r\nD=x=y==\r\n".encode("utf-8"))
    assert load_env_file(p) == {"A": "1", "B": "two", "C": "pa#ss", "D": "x=y=="}


def test_secrets_never_reach_output_or_alerts(tmp_path):
    env = {"TELEGRAM_BOT_TOKEN": "SENTINEL-TOKEN", "SMTP_PASSWORD": "SENTINEL-PW", "OKX_API_SECRET": "SENTINEL-SECRET"}
    ex, alerts, out, kw, argv = setup(tmp_path, funding=HIGH)
    for cmd in (["alert-test"], ["preflight"], ["status"], ["run", "--max-steps", "2"]):
        assert main(argv + cmd, env=env, **kw) == 0
    blob = "\n".join(out + alerts)
    assert alerts and out
    for s in ("SENTINEL-TOKEN", "SENTINEL-PW", "SENTINEL-SECRET"):
        assert s not in blob


class _FakeApi:
    made: list = []

    def __init__(self, *a, **kw):
        self.args, self.kw, self.timeout = a, kw, None
        _FakeApi.made.append(self)


def _inject_fake_okx(monkeypatch):
    _FakeApi.made = []
    mods = {"okx": types.ModuleType("okx")}
    for sub, cls in (("Trade", "TradeAPI"), ("Account", "AccountAPI"),
                     ("PublicData", "PublicAPI"), ("MarketData", "MarketAPI")):
        m = types.ModuleType(f"okx.{sub}")
        setattr(m, cls, type(cls, (_FakeApi,), {}))
        setattr(mods["okx"], sub, m)
        mods[f"okx.{sub}"] = m
    for name, m in mods.items():
        monkeypatch.setitem(sys.modules, name, m)


KEYS = {"OKX_API_KEY": "k", "OKX_API_SECRET": "s", "OKX_API_PASSPHRASE": "p"}


def test_default_clients_wiring_live_defaults(monkeypatch):
    _inject_fake_okx(monkeypatch)
    def sent(msg):
        pass
    c = default_clients(dict(KEYS, OKX_SPOT_TD_MODE=""), CarryConfig(), sent)
    assert c.mode == "LIVE" and len(_FakeApi.made) == 4
    assert all(a.kw["flag"] == "0" and a.kw["debug"] is False for a in _FakeApi.made)
    assert all(a.timeout == httpx.Timeout(10.0, connect=5.0) for a in _FakeApi.made)
    assert c.exchange.spot_td_mode == "cross" and c.exchange._alert_fn is sent


@pytest.mark.parametrize("demo", ["1", "true", "YES"])
def test_default_clients_demo_flag_and_cash_mode_override(monkeypatch, demo):
    _inject_fake_okx(monkeypatch)
    c = default_clients(dict(KEYS, OKX_DEMO=demo, OKX_SPOT_TD_MODE="cash"), CarryConfig())
    assert c.mode == "DEMO" and all(a.kw["flag"] == "1" for a in _FakeApi.made)
    assert c.exchange.spot_td_mode == "cash"


def test_default_clients_missing_keys_exit_before_importing_okx(monkeypatch):
    monkeypatch.setitem(sys.modules, "okx", None)            # any import of okx would raise ImportError
    with pytest.raises(SystemExit):
        default_clients({"OKX_API_KEY": "k"}, CarryConfig())


# --- fix round 2 -------------------------------------------------------------------

from carry.state import CoinLeg


def test_force_close_closes_a_recorded_leg_even_if_coin_was_removed_from_config(tmp_path):
    ex, alerts, out, kw, argv = setup(tmp_path)
    assert main(argv + ["force-open", "BTC", "100"], env={}, **kw) == 0
    assert "BTC" in load_state(tmp_path / "state.json").legs
    eth_only = tmp_path / "eth.json"
    eth_only.write_text(json.dumps({"state_path": str(tmp_path / "state.json"), "coins": ["ETH"], "settle_sec": 0.0}))
    assert main(["--config", str(eth_only), "force-close", "BTC"], env={}, **kw) == 0
    assert load_state(tmp_path / "state.json").legs == {}
    from carry.hedge import HedgeStatus, check_hedge
    from carry.types import DEFAULT_INSTRUMENTS
    assert check_hedge(ex.spot_balance("BTC"), ex.swap_contracts("BTC"), DEFAULT_INSTRUMENTS["BTC"], 0.02) \
        == HedgeStatus.FLAT                                 # flat (fee dust below the minimum lot is fine)


def test_force_close_recorded_leg_without_instrument_definition_is_a_problem_not_a_crash(tmp_path):
    ex, alerts, out, kw, argv = setup(tmp_path)
    save_state(tmp_path / "state.json",
               CarryState(mode="HEDGED", legs={"XRP": CoinLeg(contracts=1.0, spot_qty=1.0, entry_ts=0)}))
    assert main(argv + ["force-close", "XRP"], env={}, **kw) == 1
    assert "PROBLEM: no instrument definition for XRP; close it in the OKX UI and hand-edit the state" in out
    assert ex.calls == []


def test_force_close_unrecorded_coin_not_in_config_is_still_unknown(tmp_path):
    ex, alerts, out, kw, argv = setup(tmp_path)
    eth_only = tmp_path / "eth.json"
    eth_only.write_text(json.dumps({"state_path": str(tmp_path / "state.json"), "coins": ["ETH"], "settle_sec": 0.0}))
    assert main(["--config", str(eth_only), "force-close", "BTC"], env={}, **kw) == 1
    assert any(line.startswith("error: unknown coin") for line in out)


# --- final fix wave: config-file layer live-safe defaults ---------------------------
def test_load_config_file_defaults_are_live_safe(tmp_path):
    p = tmp_path / "c.json"
    p.write_text("{}")
    c = load_config(p)
    assert c.mgn_calibrated is False and c.settle_sec == 1.0
    p.write_text(json.dumps({"mgn_calibrated": True, "settle_sec": 0.5}))
    c = load_config(p)
    assert c.mgn_calibrated is True and c.settle_sec == 0.5


def test_uncalibrated_config_run_alert_preflight_warning_and_force_open_warning(tmp_path):
    ex, alerts, out, kw, argv = setup(tmp_path, mgn_calibrated=False, funding=HIGH)
    assert main(argv + ["preflight"], env={}, **kw) == 0
    assert any(line.startswith("warning:") and "not calibrated" in line for line in out)
    out.clear()
    assert main(argv + ["force-open", "BTC", "100"], env={}, **kw) == 0          # force-open is NOT blocked
    assert out[1].startswith("WARNING: mgn not calibrated") and "BTC" in load_state(tmp_path / "state.json").legs
    assert main(argv + ["force-close", "BTC"], env={}, **kw) == 0
    assert main(argv + ["run", "--max-steps", "2"], env={}, **kw) == 0
    assert any("mgn_calibrated=False" in a and a.startswith("[INFO] carry starting") for a in alerts)
    assert load_state(tmp_path / "state.json").legs == {}                          # signal opens stayed disabled


# --- dead-man's-switch ping -------------------------------------------------

def test_build_ping_returns_none_without_url():
    assert _build_ping({}) is None


def test_build_ping_calls_get_with_url_and_timeout():
    calls = []
    ping = _build_ping({"HEALTHCHECK_PING_URL": "https://hc-ping.com/abc"},
                       get=lambda url, timeout: calls.append((url, timeout)))
    assert ping is not None
    ping()
    assert calls == [("https://hc-ping.com/abc", 10.0)]
