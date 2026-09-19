"""Run loop with heartbeat, plus a single-instance lock."""
from __future__ import annotations

import fcntl
import time
from pathlib import Path
from typing import Callable

from carry.config import CarryConfig


class InstanceLock:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._fh = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self.path, "w")
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fh.close()
            raise RuntimeError("another carry instance holds the lock") from None
        self._fh = fh

    def release(self) -> None:
        if self._fh is not None:
            try:
                fcntl.flock(self._fh, fcntl.LOCK_UN)
            finally:
                self._fh.close()
                self._fh = None

    def __enter__(self) -> "InstanceLock":
        self.acquire()
        return self

    def __exit__(self, *exc) -> None:
        self.release()


class Runner:
    def __init__(self, engine, report_fn: Callable[[], str], alert: Callable[[str], None], cfg: CarryConfig,
                 *, heartbeat_sec: float = 3600.0, clock: Callable[[], float] = time.time,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.engine, self.report_fn, self.alert, self.cfg = engine, report_fn, alert, cfg
        self.heartbeat_sec, self.clock, self.sleep = heartbeat_sec, clock, sleep

    def run(self, should_stop: Callable[[], bool]) -> None:
        last_hb = float("-inf")
        while not should_stop():
            now = self.clock()
            try:
                for action in self.engine.step(int(now * 1000)):
                    self.alert(f"[INFO] {action}")
            except Exception as e:  # noqa: BLE001 - the engine halts itself; keep the loop (and watchdog) alive
                self.alert(f"[CRITICAL] runner: step raised {e!r}")
            if now - last_hb >= self.heartbeat_sec:
                last_hb = now
                try:
                    self.alert("[HEARTBEAT]\n" + self.report_fn())
                except Exception as e:  # noqa: BLE001
                    self.alert(f"[CRITICAL] runner: heartbeat failed {e!r}")
            self.sleep(self.cfg.poll_sec)
