import os

import pytest

from carry.bootstrap import exposure_report
from carry.config import CarryConfig
from carry.engine import Engine
from carry.fake_exchange import FakeExchange
from carry.runner import InstanceLock, Runner
from carry.state import CarryState
from carry.types import DEFAULT_INSTRUMENTS


class Clock:
    def __init__(self):
        self.t = 1000.0

    def now(self):
        return self.t

    def sleep(self, s):
        self.t += s


class FakeEngine:
    def __init__(self, script=None):
        self.calls, self.script = [], list(script or [])

    def step(self, now_ms):
        self.calls.append(now_ms)
        item = self.script.pop(0) if self.script else []
        if isinstance(item, Exception):
            raise item
        return item


def stop_after(n):
    state = {"i": 0}

    def should_stop():
        state["i"] += 1
        return state["i"] > n
    return should_stop


def make(engine, alerts, report=lambda: "REPORT", heartbeat=5.0):
    clock = Clock()
    return Runner(engine, report, alerts.append, CarryConfig(poll_sec=2.0), heartbeat_sec=heartbeat,
                  clock=clock.now, mono=clock.now, sleep=clock.sleep), clock


def test_steps_once_per_poll_and_advances_time():
    eng, alerts = FakeEngine(), []
    runner, _ = make(eng, alerts)
    runner.run(stop_after(3))
    assert len(eng.calls) == 3
    assert eng.calls[1] - eng.calls[0] == 2000


def test_actions_are_alerted():
    eng, alerts = FakeEngine([["opened BTC"]]), []
    make(eng, alerts)[0].run(stop_after(1))
    assert "[INFO] opened BTC" in alerts


def test_heartbeat_cadence():
    eng, alerts = FakeEngine(), []
    make(eng, alerts, heartbeat=5.0)[0].run(stop_after(5))       # t = 1000,1002,1004,1006,1008
    assert sum(a.startswith("[HEARTBEAT]") for a in alerts) == 2  # at 1000 and 1006
    assert alerts[0] == "[HEARTBEAT]\nREPORT"


def test_step_exception_is_alerted_and_loop_continues():
    eng, alerts = FakeEngine([RuntimeError("boom"), ["ok"]]), []
    make(eng, alerts)[0].run(stop_after(2))
    assert len(eng.calls) == 2
    assert any("[CRITICAL] runner: step raised" in a and "boom" in a for a in alerts)
    assert "[INFO] ok" in alerts


def test_report_failure_does_not_stop_the_loop():
    def bad():
        raise ConnectionError("down")
    eng, alerts = FakeEngine(), []
    make(eng, alerts, report=bad)[0].run(stop_after(2))
    assert len(eng.calls) == 2


def test_instance_lock_is_exclusive(tmp_path):
    p = tmp_path / "carry.lock"
    a, b = InstanceLock(p), InstanceLock(p)
    a.acquire()
    with pytest.raises(RuntimeError):
        b.acquire()
    a.release()
    b.acquire()
    b.release()


def test_instance_lock_context_manager(tmp_path):
    p = tmp_path / "carry.lock"
    with InstanceLock(p):
        with pytest.raises(RuntimeError):
            InstanceLock(p).acquire()
    with InstanceLock(p):
        pass


def test_raising_alert_does_not_stop_the_loop_on_step_error_and_heartbeat():
    def bad_alert(msg):
        raise ConnectionError("telegram down")

    eng = FakeEngine([RuntimeError("boom"), ["opened BTC"], []])
    clock = Clock()
    Runner(eng, lambda: "REPORT", bad_alert, CarryConfig(poll_sec=2.0), heartbeat_sec=1.0,
           clock=clock.now, mono=clock.now, sleep=clock.sleep).run(stop_after(3))
    assert len(eng.calls) == 3


def test_heartbeat_pacing_uses_monotonic_not_wall_clock():
    wall = {"t": 1_000_000.0}
    mono = {"t": 0.0}

    def sleep(s):
        mono["t"] += s
        wall["t"] -= 3600.0          # wall clock keeps jumping backward (NTP step / DST-like)

    alerts = []
    eng = FakeEngine()
    Runner(eng, lambda: "REPORT", alerts.append, CarryConfig(poll_sec=2.0), heartbeat_sec=5.0,
           clock=lambda: wall["t"], mono=lambda: mono["t"], sleep=sleep).run(stop_after(5))
    assert sum(a.startswith("[HEARTBEAT]") for a in alerts) == 2     # mono 0 and 6
    assert eng.calls[1] < eng.calls[0]                               # engine still gets wall-clock ms


def test_real_halted_engine_still_heartbeats():
    ex, alerts = FakeExchange(), []
    cfg = CarryConfig(poll_sec=2.0)
    eng = Engine(ex, DEFAULT_INSTRUMENTS, cfg, alerts.append, state=CarryState(mode="HALTED", halted_reason="x"))
    clock = Clock()
    Runner(eng, lambda: exposure_report(ex, DEFAULT_INSTRUMENTS, cfg, eng.state), alerts.append, cfg,
           heartbeat_sec=5.0, clock=clock.now, mono=clock.now, sleep=clock.sleep).run(stop_after(4))
    hb = [a for a in alerts if a.startswith("[HEARTBEAT]")]
    assert len(hb) == 2 and "mode=HALTED" in hb[0]
    assert eng.state.mode == "HALTED"


def test_ping_fires_once_per_successful_heartbeat():
    eng, alerts, pings = FakeEngine(), [], []
    clock = Clock()
    Runner(eng, lambda: "REPORT", alerts.append, CarryConfig(poll_sec=2.0), heartbeat_sec=5.0,
           clock=clock.now, mono=clock.now, sleep=clock.sleep, ping=lambda: pings.append(1)).run(stop_after(5))
    assert len(pings) == 2          # same cadence as the heartbeat (t=1000 and 1006)


def test_ping_not_called_when_heartbeat_report_fails():
    def bad():
        raise ConnectionError("down")
    eng, alerts, pings = FakeEngine(), [], []
    make(eng, alerts, report=bad)[0]
    clock = Clock()
    Runner(eng, bad, alerts.append, CarryConfig(poll_sec=2.0), heartbeat_sec=1.0,
           clock=clock.now, mono=clock.now, sleep=clock.sleep, ping=lambda: pings.append(1)).run(stop_after(2))
    assert pings == []


def test_no_ping_configured_does_not_raise():
    eng, alerts = FakeEngine(), []
    make(eng, alerts, heartbeat=1.0)[0].run(stop_after(2))   # ping=None by default


def test_raising_ping_does_not_stop_the_loop():
    def bad_ping():
        raise RuntimeError("monitor unreachable")
    eng, alerts = FakeEngine(), []
    clock = Clock()
    Runner(eng, lambda: "REPORT", alerts.append, CarryConfig(poll_sec=2.0), heartbeat_sec=1.0,
           clock=clock.now, mono=clock.now, sleep=clock.sleep, ping=bad_ping).run(stop_after(3))
    assert len(eng.calls) == 3


def test_lock_file_contains_pid(tmp_path):
    p = tmp_path / "carry.lock"
    a = InstanceLock(p)
    a.acquire()
    assert p.read_text() == str(os.getpid())
    with pytest.raises(RuntimeError):
        InstanceLock(p).acquire()
    assert p.read_text() == str(os.getpid())          # the loser must not wipe the holder's PID
    a.release()
