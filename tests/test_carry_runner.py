import pytest

from carry.config import CarryConfig
from carry.runner import InstanceLock, Runner


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
                  clock=clock.now, sleep=clock.sleep), clock


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
