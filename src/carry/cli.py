"""Command line for the carry engine."""
from __future__ import annotations

import argparse
import json
import os
import signal
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from carry.alerting import build_alerter
from carry.bootstrap import exposure_report, load_state_or_halt, reconcile
from carry.config import CarryConfig
from carry.engine import Engine
from carry.exchange import Exchange
from carry.hedge import HedgeStatus, check_hedge
from carry.preflight import preflight
from carry.runner import InstanceLock, Runner
from carry.state import CarryState, load_state, save_state
from carry.types import DEFAULT_INSTRUMENTS


@dataclass
class Clients:
    exchange: Exchange
    account: Any
    public: Any
    mode: str = "LIVE"                                        # "DEMO" | "LIVE": which OKX environment


def _okx_flag(env: Mapping[str, str]) -> str:
    return "1" if str(env.get("OKX_DEMO", "")).lower() in ("1", "true", "yes") else "0"


def load_env_file(path: str | Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:]
        k, v = line.split("=", 1)
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]
        out[k.strip()] = v
    return out


def load_config(path: str | Path) -> CarryConfig:
    data = json.loads(Path(path).read_text())
    unknown = set(data) - {f.name for f in fields(CarryConfig)}
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    if "state_path" in data and not os.path.isabs(str(data["state_path"])):
        raise ValueError("state_path must be an absolute path (a relative path would move the state and lock "
                         f"with the working directory), got {data['state_path']!r}")
    if "coins" in data:
        data["coins"] = tuple(data["coins"])
    return CarryConfig(**data)


def default_clients(env: Mapping[str, str], cfg: CarryConfig, alert: Callable[[str], None] | None = None) -> Clients:
    key, secret, phrase = (env.get(k) for k in ("OKX_API_KEY", "OKX_API_SECRET", "OKX_API_PASSPHRASE"))
    if not (key and secret and phrase):
        raise SystemExit("OKX_API_KEY / OKX_API_SECRET / OKX_API_PASSPHRASE must be set")
    flag = _okx_flag(env)
    import httpx
    import okx.Account as Account
    import okx.MarketData as MarketData
    import okx.PublicData as PublicData
    import okx.Trade as Trade
    from carry.okx_exchange import OkxExchange
    trade = Trade.TradeAPI(key, secret, phrase, False, flag=flag, debug=False)
    account = Account.AccountAPI(key, secret, phrase, False, flag=flag, debug=False)
    public = PublicData.PublicAPI(flag=flag, debug=False)
    market = MarketData.MarketAPI(flag=flag, debug=False)
    for client in (trade, account, public, market):            # python-okx clients are httpx.Client subclasses
        client.timeout = httpx.Timeout(10.0, connect=5.0)
    ex = OkxExchange(trade, account, public, market, DEFAULT_INSTRUMENTS,
                     spot_td_mode=env.get("OKX_SPOT_TD_MODE") or "cash", alert=alert)
    return Clients(ex, account, public, mode="DEMO" if flag == "1" else "LIVE")


class _StopFlag:
    def __init__(self, max_steps: int | None) -> None:
        self.max_steps, self.calls, self.signalled = max_steps, 0, False

    def trigger(self) -> None:
        self.signalled = True

    def __call__(self) -> bool:
        self.calls += 1
        return self.signalled or (self.max_steps is not None and self.calls > self.max_steps)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="carry")
    ap.add_argument("--config")
    ap.add_argument("--env-file")
    ap.add_argument("--log", default="live_state/carry.log")
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--max-steps", type=int)
    run.add_argument("--heartbeat-sec", type=float, default=3600.0)
    for name in ("status", "preflight", "rearm", "alert-test"):
        sub.add_parser(name)
    fo = sub.add_parser("force-open")
    fo.add_argument("coin")
    fo.add_argument("notional", type=float)
    fc = sub.add_parser("force-close")
    fc.add_argument("coin")
    return ap


def _safe_alert(alerter: Callable[[str], None], msg: str) -> None:
    try:
        alerter(msg)
    except Exception:  # noqa: BLE001 - a broken alerter must not abort startup
        pass


def _engine(clients: Clients, cfg: CarryConfig, alerter: Callable[[str], None],
            clock: Callable[[], float], out: Callable[[str], None]) -> Engine:
    """Startup order: load state -> adopt pending orders -> reconcile -> save -> Engine (same state object).

    Any exception in the adopt+reconcile step (e.g. a transient exchange error) starts the bot HALTED.
    """
    ex = clients.exchange
    state = load_state_or_halt(cfg.state_path, alerter, clock)
    try:
        adopt = getattr(ex, "adopt_pending_orders", None)
        if callable(adopt):
            adopted = adopt()
            if adopted:
                out(f"adopted pending orders: {list(adopted)}")
        notes = reconcile(ex, DEFAULT_INSTRUMENTS, cfg, state, alerter, int(clock() * 1000))
    except Exception as e:  # noqa: BLE001 - never start tradeable when the startup check could not complete
        name = type(e).__name__
        _safe_alert(alerter, f"[CRITICAL] startup check failed ({name}); starting HALTED")
        out(f"startup check failed ({name}); starting HALTED")
        state.mode = "HALTED"
        reason = f"startup check failed: {name}"
        if reason not in state.halted_reason:
            state.halted_reason = f"{state.halted_reason}; {reason}" if state.halted_reason else reason
    else:
        for note in notes:
            out(f"reconcile: {note}")
    try:
        save_state(cfg.state_path, state)
    except OSError:
        _safe_alert(alerter, "[CRITICAL] could not persist startup state")
    return Engine(ex, DEFAULT_INSTRUMENTS, cfg, alerter, state,
                  save=lambda s: save_state(cfg.state_path, s))


def _status_state(cfg: CarryConfig, out: Callable[[str], None]) -> CarryState:
    """Read-only view of the state file: status must never rename, rewrite or alert."""
    try:
        return load_state(cfg.state_path)
    except Exception as e:  # noqa: BLE001
        out(f"state file unreadable ({type(e).__name__}) \u2014 showing HALTED")
        return CarryState(mode="HALTED", halted_reason=f"state file unreadable: {type(e).__name__}")


def _force_close(eng: Engine, coin: str, clients: Clients, cfg: CarryConfig, out: Callable[[str], None]) -> int:
    if coin not in cfg.coins:
        out(f"error: unknown coin {coin!r}")
        return 1
    if coin not in eng.state.legs:
        try:
            spot, perp = clients.exchange.spot_balance(coin), clients.exchange.swap_contracts(coin)
            status = check_hedge(spot, perp, DEFAULT_INSTRUMENTS[coin], cfg.drift_tol)
        except Exception as e:  # noqa: BLE001
            out(f"error: could not read {coin} holdings ({type(e).__name__})")
            return 1
        if status != HedgeStatus.FLAT:
            out(f"PROBLEM: no recorded leg for {coin} but the exchange holds spot={spot:.8g} perp={perp:.8g} "
                "contracts \u2014 close it in the OKX UI, then run rearm")
            return 1
        out(f"force-close {coin}: nothing to close")
        return 0
    ok = eng.force_close(coin)
    out(f"force-close {coin}: {'ok' if ok else 'FAILED'}")
    return 0 if ok else 1


def _preflight_command(clients: Clients, cfg: CarryConfig, out: Callable[[str], None]) -> int:
    res = preflight(clients.account, clients.public, cfg, DEFAULT_INSTRUMENTS)
    problems = list(res.problems)
    pending_fn = getattr(clients.exchange, "pending_carry_orders", None)
    if callable(pending_fn):
        try:
            problems += [f"pending carry order still live on the exchange: {cl}" for cl in pending_fn()]
        except Exception as e:  # noqa: BLE001
            problems.append(f"could not read pending orders: {type(e).__name__}: {e}")
    for p in problems:
        out(f"PROBLEM: {p}")
    for w in res.warnings:
        out(f"warning: {w}")
    out("preflight OK" if not problems else "preflight FAILED")
    return 0 if not problems else 1


def _locked_command(args: argparse.Namespace, clients: Clients, cfg: CarryConfig,
                    alerter: Callable[[str], None], clock: Callable[[], float],
                    sleep: Callable[[float], None], out: Callable[[str], None]) -> int:
    lock = InstanceLock(os.path.abspath(cfg.state_path) + ".lock")
    try:
        lock.acquire()
    except RuntimeError as e:
        out(str(e))
        return 3
    try:
        eng = _engine(clients, cfg, alerter, clock, out)
        if args.cmd == "rearm":
            eng.rearm()
            clear = getattr(clients.exchange, "clear_unconfirmed", None)
            if callable(clear):
                clear()
            out("rearmed")
            return 0
        if args.cmd == "force-open":
            ok = eng.force_open(args.coin, args.notional, int(clock() * 1000))
            out(f"force-open {args.coin}: {'ok' if ok else 'FAILED'}")
            return 0 if ok else 1
        if args.cmd == "force-close":
            return _force_close(eng, args.coin, clients, cfg, out)
        stop = _StopFlag(args.max_steps)                       # cmd == "run"
        _safe_alert(alerter, f"[INFO] carry starting (okx mode: {clients.mode}, coins={list(cfg.coins)}, "
                             f"state={eng.state.mode}, poll_sec={cfg.poll_sec})")
        old = {s: signal.signal(s, lambda *_: stop.trigger()) for s in (signal.SIGTERM, signal.SIGINT)}
        try:
            Runner(eng, lambda: f"okx={clients.mode} "
                   + exposure_report(clients.exchange, DEFAULT_INSTRUMENTS, cfg, eng.state), alerter,
                   cfg, heartbeat_sec=args.heartbeat_sec, clock=clock, sleep=sleep).run(stop)
        finally:
            for s, h in old.items():
                signal.signal(s, h)
        return 0
    except (ValueError, RuntimeError) as e:
        out(f"error: {e}")
        return 1
    finally:
        lock.release()


def main(argv: Sequence[str] | None = None, env: Mapping[str, str] | None = None, *,
         make_clients: Callable[..., Clients] | None = None,
         make_alerter: Callable[..., Callable[[str], None]] | None = None,
         out: Callable[[str], None] = print, clock: Callable[[], float] = time.time,
         sleep: Callable[[float], None] = time.sleep) -> int:
    args = build_parser().parse_args(argv)
    env = dict(os.environ if env is None else env)
    try:
        if args.env_file:
            env.update(load_env_file(args.env_file))
        cfg = load_config(args.config) if args.config else CarryConfig()
    except (OSError, ValueError, TypeError) as e:
        out(f"config error: {e}")
        return 2
    alerter = (make_alerter or build_alerter)(env, args.log)
    try:
        if args.cmd == "alert-test":
            out(f"okx mode: {'DEMO' if _okx_flag(env) == '1' else 'LIVE'}")
            alerter("[INFO] carry alert test")
            out(f"channels: {getattr(alerter, 'channels', [])}")
            return 0

        try:
            clients = (make_clients or default_clients)(env, cfg, alerter)
        except SystemExit as e:                                # e.g. missing OKX credentials
            out(str(e))
            return 2
        out(f"okx mode: {clients.mode}")

        if args.cmd == "preflight":
            return _preflight_command(clients, cfg, out)

        if args.cmd == "status":
            out(exposure_report(clients.exchange, DEFAULT_INSTRUMENTS, cfg, _status_state(cfg, out)))
            return 0

        return _locked_command(args, clients, cfg, alerter, clock, sleep, out)
    finally:
        flush = getattr(alerter, "flush", None)              # deliver queued notifications before exiting
        if callable(flush):
            try:
                flush()
            except Exception:  # noqa: BLE001 - a broken flush must not change the exit code
                pass
