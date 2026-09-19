# Funding Carry — Phase 4 (Live Plumbing, no keys needed) + Phase 5 checklist Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Everything between the tested offline core (Phase 3) and a real OKX account, built and unit-tested WITHOUT any API key: alerting (log + Telegram + email), the OKX `Exchange` adapter, startup reconcile, the run loop, preflight checks, CLI and deployment files. The OKX demo cycle and small live run (Phase 5) are a manual checklist at the end, done last, once the user creates keys.

**Architecture:** `alerting.py` (throttled fan-out, never raises) · `okx_exchange.py` (adapter over injected python-okx-shaped API objects, so every code path is tested with fakes) · `bootstrap.py` (state fallback, reconcile, exposure report) · `runner.py` (loop, heartbeat, instance lock) · `preflight.py` · `cli.py` (+ `__main__.py`) · deploy files. The Phase 3 `Engine` is reused unchanged except for one small `Executor`/`CarryConfig` addition (Task 1).

**Tech Stack:** Python 3.11+, stdlib (`logging`, `smtplib`, `fcntl`, `argparse`, `json`), `httpx` (Telegram), `python-okx` (only imported by the default client factory), pytest.

**Spec / prior work:** `docs/superpowers/specs/2026-09-19-funding-carry-design.md`, Phase 3 plan `2026-09-19-funding-carry-phase3.md`, review carry-forward list in `.superpowers/sdd/2026-09-19-funding-carry-phase3/progress.md`.

## Execution batches (token economy: one implementer + one review per batch)

- Batch A: Task 1 (executor settle delay) + Task 2 (alerting)
- Batch B: Task 3 (OKX read side) + Task 4 (OKX write side) — safety-critical, review deeply
- Batch C: Task 5 (bootstrap) + Task 6 (runner)
- Batch D: Task 7 (preflight + CLI) + Task 8 (deploy files + runbook)

## Global Constraints

- Python >= 3.11, type hints on all signatures, `from __future__ import annotations`.
- NO credentials in code, tests, logs or committed files. Secrets only from env (`OKX_API_KEY`, `OKX_API_SECRET`, `OKX_API_PASSPHRASE`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`, `ALERT_EMAIL_FROM`, `ALERT_EMAIL_TO`, optional `OKX_DEMO=1`, `OKX_SPOT_TD_MODE`). Alert code must never log a secret or a URL containing a token (log the exception CLASS NAME only).
- No test performs network I/O or real sleeps; time and sleep are injected.
- Real python-okx signatures (verified in this repo's venv): `TradeAPI.place_order(instId, tdMode, side, ordType, sz, ccy='', clOrdId='', tag='', posSide='', px='', reduceOnly='', tgtCcy='', ...)`, `get_order(instId, ordId='', clOrdId='')`, `cancel_order(instId, ordId='', clOrdId='')`; `AccountAPI.get_positions(instType='', instId='', posId='')`, `get_account_balance(ccy='')`, `get_account_config()`; `PublicAPI.funding_rate_history(instId, after='', before='', limit='')` (NO `get_` prefix), `get_instruments(instType, ...)`; `MarketAPI.get_ticker(instId)`. ADL rank = the `adl` field of a positions row (no separate endpoint). Existing project convention: swaps use `tdMode="cross"`; API objects are built as `TradeAPI(key, secret, passphrase, False, flag=flag, debug=False)` with `flag="0"` live / `"1"` demo.
- Facts from the OKX probe: BTC swap ctVal 0.01, ETH 0.1; swap lot/min 0.01 contracts; spot BTC lot 1e-8 / min 1e-5, ETH lot 1e-6 / min 1e-4; account acctLv=3, `autoLoan` false, `posMode` net_mode.
- Tests: worktree has no venv: `export PY=/home/airwolf/Github/crypto_backtest/.venv/bin/python`, run `$PY -m pytest tests/test_carry_*.py -q` from the worktree root (currently 145 passing).
- Commit trailer exactly: `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>` (whichever model you are).

## File Structure

| File | Responsibility |
|---|---|
| `src/carry/executor.py` (modify) | settle delay after each order (Task 1) |
| `src/carry/config.py` (modify) | `settle_sec` (Task 1) |
| `src/carry/alerting.py` | `setup_logger`, `TelegramSink`, `EmailSink`, `Alerter`, `build_alerter` |
| `src/carry/okx_exchange.py` | `ExchangeError`, `OkxExchange` (read side Task 3, write side Task 4) |
| `src/carry/bootstrap.py` | `load_state_or_halt`, `reconcile`, `exposure_report` |
| `src/carry/runner.py` | `Runner`, `InstanceLock` |
| `src/carry/preflight.py` | `PreflightResult`, `preflight` |
| `src/carry/cli.py`, `src/carry/__main__.py` | commands, env/config loading, client factory |
| `config/carry.example.json`, `deploy/carry.service`, `deploy/carry.env.example`, `deploy/README-carry.md` | deployment + runbook |
| `tests/test_carry_<module>.py` | one test file per module |

---

### Task 1: Executor settle delay (read-after-write safety)

**Files:**
- Modify: `src/carry/executor.py`, `src/carry/config.py`
- Test: `tests/test_carry_executor.py` (append), `tests/test_carry_config.py` (append)

**Interfaces:**
- Produces: `CarryConfig.settle_sec: float = 0.0` (validated `>= 0`); `Executor(ex, insts, cfg, alert, sleep: Callable[[float], None] = time.sleep)` with a private `_settle()` that sleeps `cfg.settle_sec` when it is `> 0`. `_settle()` is called (a) in `_open_legs` right after the `sell_swap` try/except and before `held` is read, (b) right after the `buy_spot` try/except and before `bought` is read, (c) in `_drain` after every `order(excess)` attempt (after its try/except). Rationale: a real exchange may lag between an order and the position read; without a delay `_drain` could re-send the same order (oversell) or `open` could believe the perp did not fill.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_carry_executor.py`:

```python
def test_settle_delay_after_each_order():
    ex = FakeExchange()
    slept: list[float] = []
    exe = Executor(ex, DEFAULT_INSTRUMENTS, CarryConfig(settle_sec=0.5), [].append, sleep=slept.append)
    leg = exe.open("BTC", 125.0, now_ms=0)
    assert leg is not None
    assert slept == [0.5, 0.5]            # after the perp order and after the spot order
    slept.clear()
    assert exe.close("BTC", leg)
    assert slept == [0.5, 0.5]            # one sell_spot attempt + one buy_swap attempt


def test_no_sleep_when_settle_is_zero():
    ex = FakeExchange()
    slept: list[float] = []
    exe = Executor(ex, DEFAULT_INSTRUMENTS, CarryConfig(), [].append, sleep=slept.append)
    leg = exe.open("BTC", 125.0, now_ms=0)
    exe.close("BTC", leg)
    assert slept == []
```

Append to `tests/test_carry_config.py`:

```python
def test_settle_sec_default_and_validation():
    assert CarryConfig().settle_sec == 0.0
    with pytest.raises(ValueError):
        CarryConfig(settle_sec=-1.0)
```

- [ ] **Step 2: Run to verify failure**

Run: `$PY -m pytest tests/test_carry_executor.py tests/test_carry_config.py -q`
Expected: FAIL (`TypeError` unexpected keyword `settle_sec` / `sleep`).

- [ ] **Step 3: Implement** — in `config.py` add the field `settle_sec: float = 0.0  # pause after each order before re-reading positions` and in `__post_init__` `if self.settle_sec < 0: raise ValueError("settle_sec must be >= 0")`. In `executor.py`: `import time`; add the `sleep` constructor parameter stored as `self._sleep`; add

```python
    def _settle(self) -> None:
        if self.cfg.settle_sec > 0:
            self._sleep(self.cfg.settle_sec)
```

and the three call sites described in Interfaces. Nothing else changes.

- [ ] **Step 4: Run to verify pass**

Run: `$PY -m pytest tests/test_carry_*.py -q`
Expected: all pass (145 previous + 3 new).

- [ ] **Step 5: Commit**

```bash
git add src/carry/executor.py src/carry/config.py tests/test_carry_executor.py tests/test_carry_config.py
git commit -m "feat(carry): settle delay between an order and the position re-read

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 2: Alerting (log + Telegram + email)

**Files:**
- Create: `src/carry/alerting.py`
- Test: `tests/test_carry_alerting.py`

**Interfaces:**
- Produces: `setup_logger(path, level=logging.INFO) -> logging.Logger` (logger name `"carry"`, rotating file handler 5 MB x 5 + stream handler, idempotent); `TelegramSink(token, chat_id, post=None, timeout=10.0)` and `EmailSink(host, port, user, password, sender, recipients, smtp_factory=None, timeout=15.0)`, each with `name` and `send(text)`; `Alerter(log, sinks=(), throttle_sec=300.0, clock=time.time)` — a callable `alerter(msg)` that ALWAYS logs (`[CRITICAL]` prefix -> `log.critical`, else `log.info`), then sends to each sink unless the identical message was sent within `throttle_sec`; never raises; a failing sink is logged by sink name and exception CLASS NAME only (never `str(e)`, which for httpx contains the bot token); `alerter.channels -> list[str]`. `build_alerter(env, log_path, *, post=None, smtp_factory=None, clock=time.time, throttle_sec=300.0) -> Alerter` adds a Telegram sink only when both `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` are set, an email sink only when `SMTP_HOST`, `SMTP_USER`, `SMTP_PASSWORD`, `ALERT_EMAIL_TO` are set (`SMTP_PORT` default 465, `ALERT_EMAIL_FROM` default `SMTP_USER`, recipients comma-separated); with no sink it logs a warning "no notification channel configured (log only)".

- [ ] **Step 1: Write the failing test**

```python
import logging

import pytest

from carry.alerting import Alerter, EmailSink, TelegramSink, build_alerter, setup_logger


class Sink:
    def __init__(self, name="fake", fail=False):
        self.name, self.fail, self.sent = name, fail, []

    def send(self, text):
        if self.fail:
            raise RuntimeError("secret-token-123 leaked in message")
        self.sent.append(text)


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def make(sinks, clock=None, throttle=300.0):
    return Alerter(logging.getLogger("carry.test"), sinks, throttle_sec=throttle, clock=clock or Clock())


def test_message_is_logged_and_sent(caplog):
    s = Sink()
    with caplog.at_level(logging.INFO, logger="carry.test"):
        make([s])("[CRITICAL] hedge broken")
    assert s.sent == ["[CRITICAL] hedge broken"]
    assert any(r.levelno == logging.CRITICAL for r in caplog.records)


def test_identical_message_is_throttled_but_still_logged(caplog):
    s, clock = Sink(), Clock()
    a = make([s], clock)
    with caplog.at_level(logging.INFO, logger="carry.test"):
        a("same")
        clock.t += 10
        a("same")
        assert len(s.sent) == 1 and len(caplog.records) == 2
        clock.t += 400
        a("same")
    assert len(s.sent) == 2


def test_failing_sink_is_isolated_and_secret_never_logged(caplog):
    bad, good = Sink("telegram", fail=True), Sink("email")
    with caplog.at_level(logging.INFO, logger="carry.test"):
        make([bad, good])("hello")          # must not raise
    assert good.sent == ["hello"]
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "telegram" in text and "RuntimeError" in text and "secret-token-123" not in text


def test_channels_lists_sink_names():
    assert make([Sink("telegram"), Sink("email")]).channels == ["telegram", "email"]


def test_telegram_sink_posts_json_and_checks_status():
    calls = []

    class Resp:
        def raise_for_status(self):
            calls.append("checked")

    def post(url, json, timeout):
        calls.append((url, json, timeout))
        return Resp()

    TelegramSink("TOKEN", "42", post=post).send("x" * 5000)
    url, payload, timeout = calls[0]
    assert url == "https://api.telegram.org/botTOKEN/sendMessage"
    assert payload["chat_id"] == "42" and len(payload["text"]) == 4000 and timeout == 10.0
    assert calls[1] == "checked"


class FakeSMTP:
    instances = []

    def __init__(self, host, port, timeout):
        self.host, self.port, self.log = host, port, []
        FakeSMTP.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def starttls(self):
        self.log.append("starttls")

    def login(self, u, p):
        self.log.append(("login", u))

    def send_message(self, msg):
        self.log.append(("send", msg["Subject"], msg["To"]))


def test_email_sink_starttls_login_send():
    FakeSMTP.instances.clear()
    EmailSink("smtp.example.com", 587, "u", "p", "u@example.com", ["a@x.com", "b@x.com"],
              smtp_factory=FakeSMTP).send("[CRITICAL] first line\nsecond line")
    log = FakeSMTP.instances[0].log
    assert log[0] == "starttls" and log[1] == ("login", "u")
    assert log[2] == ("send", "[carry] [CRITICAL] first line", "a@x.com, b@x.com")


def test_email_sink_port_465_skips_starttls():
    FakeSMTP.instances.clear()
    EmailSink("h", 465, "u", "p", "u@x", ["a@x"], smtp_factory=FakeSMTP).send("m")
    assert "starttls" not in FakeSMTP.instances[0].log


@pytest.fixture
def clean_carry_logger():
    yield
    lg = logging.getLogger("carry")
    for h in list(lg.handlers):
        lg.removeHandler(h)
        h.close()


def test_build_alerter_detects_channels(tmp_path, clean_carry_logger):
    log = tmp_path / "carry.log"
    assert build_alerter({}, str(log)).channels == []
    tg = {"TELEGRAM_BOT_TOKEN": "t", "TELEGRAM_CHAT_ID": "1"}
    assert build_alerter(tg, str(log)).channels == ["telegram"]
    mail = {"SMTP_HOST": "h", "SMTP_USER": "u", "SMTP_PASSWORD": "p", "ALERT_EMAIL_TO": "a@x.com"}
    assert build_alerter({**tg, **mail}, str(log)).channels == ["telegram", "email"]
    assert build_alerter({"TELEGRAM_BOT_TOKEN": "t"}, str(log)).channels == []      # chat id missing


def test_setup_logger_is_idempotent_and_writes_file(tmp_path, clean_carry_logger):
    p = tmp_path / "sub" / "carry.log"
    lg = setup_logger(p)
    n = len(lg.handlers)
    assert len(setup_logger(p).handlers) == n
    lg.info("hello file")
    for h in lg.handlers:
        h.flush()
    assert "hello file" in p.read_text()
```

- [ ] **Step 2: Run to verify it fails**

Run: `$PY -m pytest tests/test_carry_alerting.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'carry.alerting'`

- [ ] **Step 3: Write the implementation**

```python
"""Alerting: a durable log plus throttled Telegram/email notifications. Never raises."""
from __future__ import annotations

import logging
import smtplib
import time
from email.message import EmailMessage
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Callable, Mapping, Protocol, Sequence

LOGGER_NAME = "carry"


def setup_logger(path: str | Path, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    if not any(isinstance(h, RotatingFileHandler) and Path(h.baseFilename) == p.resolve()
               for h in logger.handlers):
        fh = RotatingFileHandler(p, maxBytes=5_000_000, backupCount=5, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    if not any(type(h) is logging.StreamHandler for h in logger.handlers):
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    return logger


class Sink(Protocol):
    name: str

    def send(self, text: str) -> None: ...


class TelegramSink:
    name = "telegram"

    def __init__(self, token: str, chat_id: str, post: Callable | None = None, timeout: float = 10.0) -> None:
        self.token, self.chat_id, self.timeout, self._post = token, chat_id, timeout, post

    def send(self, text: str) -> None:
        post = self._post
        if post is None:
            import httpx
            post = httpx.post
        r = post(f"https://api.telegram.org/bot{self.token}/sendMessage",
                 json={"chat_id": self.chat_id, "text": text[:4000]}, timeout=self.timeout)
        r.raise_for_status()


class EmailSink:
    name = "email"

    def __init__(self, host: str, port: int, user: str, password: str, sender: str,
                 recipients: Sequence[str], smtp_factory: Callable | None = None, timeout: float = 15.0) -> None:
        self.host, self.port, self.user, self.password = host, port, user, password
        self.sender, self.recipients, self.timeout, self._factory = sender, list(recipients), timeout, smtp_factory

    def send(self, text: str) -> None:
        msg = EmailMessage()
        msg["Subject"] = "[carry] " + text.splitlines()[0][:100]
        msg["From"] = self.sender
        msg["To"] = ", ".join(self.recipients)
        msg.set_content(text)
        factory = self._factory or (smtplib.SMTP_SSL if self.port == 465 else smtplib.SMTP)
        with factory(self.host, self.port, timeout=self.timeout) as s:
            if self.port != 465:
                s.starttls()
            s.login(self.user, self.password)
            s.send_message(msg)


class Alerter:
    def __init__(self, log: logging.Logger, sinks: Sequence[Sink] = (), throttle_sec: float = 300.0,
                 clock: Callable[[], float] = time.time) -> None:
        self.log, self.sinks, self.throttle_sec, self.clock = log, list(sinks), throttle_sec, clock
        self._last: dict[str, float] = {}

    @property
    def channels(self) -> list[str]:
        return [s.name for s in self.sinks]

    def __call__(self, msg: str) -> None:
        try:
            (self.log.critical if msg.startswith("[CRITICAL]") else self.log.info)(msg)
        except Exception:  # noqa: BLE001
            pass
        now = self.clock()
        self._last = {m: t for m, t in self._last.items() if now - t < self.throttle_sec}
        if msg in self._last:
            return
        self._last[msg] = now
        for s in self.sinks:
            try:
                s.send(msg)
            except Exception as e:  # noqa: BLE001 - never log str(e): it can contain a token/URL
                try:
                    self.log.error("alert sink %s failed: %s", s.name, type(e).__name__)
                except Exception:  # noqa: BLE001
                    pass


def build_alerter(env: Mapping[str, str], log_path: str, *, post: Callable | None = None,
                  smtp_factory: Callable | None = None, clock: Callable[[], float] = time.time,
                  throttle_sec: float = 300.0) -> Alerter:
    logger = setup_logger(log_path)
    sinks: list[Sink] = []
    token, chat = env.get("TELEGRAM_BOT_TOKEN"), env.get("TELEGRAM_CHAT_ID")
    if token and chat:
        sinks.append(TelegramSink(token, chat, post=post))
    host, user, pw, to = (env.get(k) for k in ("SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD", "ALERT_EMAIL_TO"))
    if host and user and pw and to:
        sinks.append(EmailSink(host, int(env.get("SMTP_PORT") or 465), user, pw,
                               env.get("ALERT_EMAIL_FROM") or user,
                               [a.strip() for a in to.split(",") if a.strip()], smtp_factory=smtp_factory))
    if not sinks:
        logger.warning("no notification channel configured (log only)")
    return Alerter(logger, sinks, throttle_sec, clock)
```

- [ ] **Step 4: Run to verify it passes**

Run: `$PY -m pytest tests/test_carry_alerting.py -q`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add src/carry/alerting.py tests/test_carry_alerting.py
git commit -m "feat(carry): alerting with durable log, Telegram and email sinks

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 3: OKX adapter — read side

**Files:**
- Create: `src/carry/okx_exchange.py`
- Test: `tests/test_carry_okx_read.py`

**Interfaces:**
- Consumes: `DEFAULT_INSTRUMENTS`, `Instrument`, `OrderResult` (`carry.types`).
- Produces: `ExchangeError(msg, code=None)` (RuntimeError with `.code`); `TRANSIENT_CODES = {"50001","50004","50011","50013","50026"}` (retryable OKX codes; verify on demo); `OkxExchange(trade, account, public, market, insts=None, *, spot_td_mode="cross", max_basis=0.01, max_tick_age_sec=30.0, order_timeout_sec=10.0, poll_sec=0.5, retries=3, backoff_sec=0.5, read_ttl_sec=1.0, clock=time.time, sleep=time.sleep)`. `clock()` returns SECONDS. Read methods implementing the `Exchange` protocol: `price(coin)` (spot last; both spot and swap tickers must be fresh — `ts` within `max_tick_age_sec` — finite, > 0, and within `max_basis` relative difference, else `ExchangeError`); `funding_history(coin, n)` (last n SETTLED rates oldest-first via `public.funding_rate_history`, reversed from newest-first; cached per coin until `latest fundingTime + 8h + 60s`; refetched when a larger `n` is requested; `n <= 0 -> []`; `n > 100 -> ExchangeError`; empty -> `ExchangeError`); `spot_balance(coin)` (`cashBal` of that currency from `get_account_balance(ccy=coin)`; missing/'' -> 0.0); `swap_contracts(coin)` (UNSIGNED short size in contracts from `get_positions(instType="SWAP", instId=...)`: net mode `pos < 0` -> `-pos`, `pos > 0` (a LONG) raises `ExchangeError`; long_short mode rows use `posSide` (`short` -> `abs(pos)`, `long` with non-zero pos raises); no rows -> 0.0); `swap_adl_rank(coin)` (`adl` field of the same position row as int, missing/'' -> 0); `mgn_ratio()` (`mgnRatio` of `get_account_balance()`; ''/missing/non-numeric/NaN/inf -> `None`, never NaN). Internals: `_call(fn, what, **kw)` retries exceptions and `TRANSIENT_CODES` up to `retries` times with `sleep(backoff_sec * 2**attempt)`, raises `ExchangeError(code=...)` immediately for any other non-"0" code; a short read cache (`read_ttl_sec`) for balance and position reads, cleared by any write (Task 4).

- [ ] **Step 1: Write the failing test**

```python
import pytest

from carry.okx_exchange import ExchangeError, OkxExchange

H8 = 8 * 3600 * 1000
NOW_MS = 1_800_000_000_000


class Clock:
    def __init__(self, t=NOW_MS / 1000):
        self.t, self.slept = t, []

    def now(self):
        return self.t

    def sleep(self, s):
        self.slept.append(s)
        self.t += s


def ok(data):
    return {"code": "0", "msg": "", "data": data}


class FakeMarket:
    def __init__(self, last=None, ts=NOW_MS):
        self.last = last or {"BTC-USDT": "60000", "BTC-USDT-SWAP": "60050", "ETH-USDT": "3000", "ETH-USDT-SWAP": "3001"}
        self.ts, self.calls, self.errors = ts, [], []

    def get_ticker(self, instId):
        self.calls.append(instId)
        if self.errors:
            raise self.errors.pop(0)
        return ok([{"instId": instId, "last": self.last[instId], "ts": str(self.ts)}])


class FakePublic:
    def __init__(self, latest=NOW_MS - 3_600_000, n=12):
        self.latest, self.n, self.calls = latest, n, []

    def funding_rate_history(self, instId, limit="", after="", before=""):
        self.calls.append((instId, limit))
        rows = [{"fundingTime": str(self.latest - i * H8), "fundingRate": str((self.n - i) * 1e-5)} for i in range(self.n)]
        return ok(rows)                                            # newest first, like OKX


class FakeAccount:
    def __init__(self):
        self.details = [{"ccy": "BTC", "cashBal": "0.002"}, {"ccy": "USDT", "cashBal": "1000"}]
        self.positions = []
        self.mgn = ""
        self.bal_calls, self.pos_calls = 0, 0

    def get_account_balance(self, ccy=""):
        self.bal_calls += 1
        det = [d for d in self.details if not ccy or d["ccy"] == ccy]
        return ok([{"details": det, "mgnRatio": self.mgn}])

    def get_positions(self, instType="", instId="", posId=""):
        self.pos_calls += 1
        return ok([p for p in self.positions if not instId or p["instId"] == instId])


def make(clock=None, **kw):
    clock = clock or Clock()
    parts = dict(trade=None, account=FakeAccount(), public=FakePublic(), market=FakeMarket())
    parts.update({k: kw.pop(k) for k in list(kw) if k in parts})
    ex = OkxExchange(parts["trade"], parts["account"], parts["public"], parts["market"],
                     clock=clock.now, sleep=clock.sleep, **kw)
    return ex, parts, clock


def test_price_returns_spot_last():
    ex, _, _ = make()
    assert ex.price("BTC") == 60000.0


def test_price_rejects_stale_tick():
    ex, _, _ = make(market=FakeMarket(ts=NOW_MS - 120_000))
    with pytest.raises(ExchangeError):
        ex.price("BTC")


def test_price_rejects_spot_swap_divergence():
    ex, _, _ = make(market=FakeMarket({"BTC-USDT": "60000", "BTC-USDT-SWAP": "63000", "ETH-USDT": "1", "ETH-USDT-SWAP": "1"}))
    with pytest.raises(ExchangeError):
        ex.price("BTC")


def test_price_rejects_non_positive_or_nan():
    for bad in ("0", "-5", "nan"):
        ex, _, _ = make(market=FakeMarket({"BTC-USDT": bad, "BTC-USDT-SWAP": "60000", "ETH-USDT": "1", "ETH-USDT-SWAP": "1"}))
        with pytest.raises(ExchangeError):
            ex.price("BTC")


def test_funding_history_oldest_first_and_sliced():
    ex, parts, _ = make()
    rates = ex.funding_history("BTC", 3)
    assert rates == pytest.approx([10e-5, 11e-5, 12e-5])
    assert parts["public"].calls == [("BTC-USDT-SWAP", "3")]


def test_funding_history_is_cached_until_next_settlement():
    ex, parts, clock = make()
    ex.funding_history("BTC", 9)
    clock.t += 600
    ex.funding_history("BTC", 9)
    assert len(parts["public"].calls) == 1
    clock.t += 8 * 3600                                            # past latest + 8h + 60s
    ex.funding_history("BTC", 9)
    assert len(parts["public"].calls) == 2


def test_funding_history_refetches_for_larger_n_and_validates():
    ex, parts, _ = make()
    ex.funding_history("BTC", 3)
    ex.funding_history("BTC", 9)
    assert len(parts["public"].calls) == 2
    assert ex.funding_history("BTC", 0) == []
    with pytest.raises(ExchangeError):
        ex.funding_history("BTC", 101)


def test_funding_history_empty_raises():
    ex, _, _ = make(public=FakePublic(n=0))
    with pytest.raises(ExchangeError):
        ex.funding_history("BTC", 3)


def test_spot_balance():
    ex, parts, _ = make()
    assert ex.spot_balance("BTC") == pytest.approx(0.002)
    assert ex.spot_balance("ETH") == 0.0                           # not in details
    parts["account"].details = [{"ccy": "BTC", "cashBal": ""}]
    ex2, _, _ = make(account=parts["account"])
    assert ex2.spot_balance("BTC") == 0.0


def pos(inst, p, side="net", adl="3"):
    return {"instId": inst, "pos": p, "posSide": side, "adl": adl}


def test_swap_contracts_net_mode_short_is_unsigned():
    acc = FakeAccount()
    acc.positions = [pos("BTC-USDT-SWAP", "-0.2")]
    ex, _, _ = make(account=acc)
    assert ex.swap_contracts("BTC") == pytest.approx(0.2)
    assert ex.swap_contracts("ETH") == 0.0


def test_swap_contracts_long_position_raises():
    acc = FakeAccount()
    acc.positions = [pos("BTC-USDT-SWAP", "0.3")]
    ex, _, _ = make(account=acc)
    with pytest.raises(ExchangeError):
        ex.swap_contracts("BTC")


def test_swap_contracts_long_short_mode():
    acc = FakeAccount()
    acc.positions = [pos("BTC-USDT-SWAP", "0.2", side="short"), pos("BTC-USDT-SWAP", "0", side="long")]
    ex, _, _ = make(account=acc)
    assert ex.swap_contracts("BTC") == pytest.approx(0.2)
    acc.positions = [pos("BTC-USDT-SWAP", "0.5", side="long")]
    ex2, _, _ = make(account=acc)
    with pytest.raises(ExchangeError):
        ex2.swap_contracts("BTC")


def test_adl_rank():
    acc = FakeAccount()
    acc.positions = [pos("BTC-USDT-SWAP", "-0.2", adl="4")]
    ex, _, _ = make(account=acc)
    assert ex.swap_adl_rank("BTC") == 4
    assert ex.swap_adl_rank("ETH") == 0
    acc.positions = [pos("BTC-USDT-SWAP", "-0.2", adl="")]
    ex2, _, _ = make(account=acc)
    assert ex2.swap_adl_rank("BTC") == 0


def test_mgn_ratio_parsing():
    for raw, want in (("", None), ("3.25", 3.25), ("nan", None), ("inf", None), ("abc", None)):
        acc = FakeAccount()
        acc.mgn = raw
        ex, _, _ = make(account=acc)
        assert ex.mgn_ratio() == want


def test_reads_are_cached_briefly_and_position_call_is_shared():
    acc = FakeAccount()
    acc.positions = [pos("BTC-USDT-SWAP", "-0.2")]
    ex, _, clock = make(account=acc)
    ex.spot_balance("BTC")
    ex.spot_balance("BTC")
    assert acc.bal_calls == 1
    ex.swap_contracts("BTC")
    ex.swap_adl_rank("BTC")
    assert acc.pos_calls == 1
    clock.t += 2.0
    ex.spot_balance("BTC")
    assert acc.bal_calls == 2


def test_transient_errors_are_retried_with_backoff():
    market = FakeMarket()
    market.errors = [ConnectionError("t"), ConnectionError("t")]
    ex, _, clock = make(market=market)
    assert ex.price("BTC") == 60000.0
    assert clock.slept[:2] == [0.5, 1.0]


def test_transient_error_code_exhausts_retries_and_others_fail_fast():
    class Bad:
        def __init__(self, code):
            self.code, self.n = code, 0

        def get_ticker(self, instId):
            self.n += 1
            return {"code": self.code, "msg": "x", "data": []}

    for code, calls in (("50011", 3), ("51000", 1)):
        bad = Bad(code)
        ex, _, _ = make(market=bad)
        with pytest.raises(ExchangeError) as e:
            ex.price("BTC")
        assert e.value.code == code and bad.n == calls
```

- [ ] **Step 2: Run to verify it fails**

Run: `$PY -m pytest tests/test_carry_okx_read.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'carry.okx_exchange'`

- [ ] **Step 3: Write the implementation**

```python
"""OKX implementation of the carry Exchange protocol (spot + USDT-swap, net mode).

The four API objects are injected python-okx-shaped clients, so every path is unit-testable
with fakes. `clock()` returns seconds. Reads go through a short cache that writes clear.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable

from carry.types import DEFAULT_INSTRUMENTS, Instrument

TRANSIENT_CODES = frozenset({"50001", "50004", "50011", "50013", "50026"})   # verify on demo
TERMINAL_STATES = frozenset({"filled", "canceled", "mmp_canceled"})
SETTLEMENT_MS = 8 * 3600 * 1000


class ExchangeError(RuntimeError):
    def __init__(self, msg: str, code: str | None = None) -> None:
        super().__init__(msg)
        self.code = code


def _f(x: Any) -> float:
    if x is None or x == "":
        return 0.0
    v = float(x)
    if not math.isfinite(v):
        raise ExchangeError(f"non-finite number {x!r}")
    return v


@dataclass
class _FundingCache:
    rates: list[float]
    limit: int
    expires_ms: float


class OkxExchange:
    def __init__(self, trade: Any, account: Any, public: Any, market: Any,
                 insts: dict[str, Instrument] | None = None, *, spot_td_mode: str = "cross",
                 max_basis: float = 0.01, max_tick_age_sec: float = 30.0, order_timeout_sec: float = 10.0,
                 poll_sec: float = 0.5, retries: int = 3, backoff_sec: float = 0.5, read_ttl_sec: float = 1.0,
                 clock: Callable[[], float] = time.time, sleep: Callable[[float], None] = time.sleep) -> None:
        self.trade, self.account, self.public, self.market = trade, account, public, market
        self.insts = insts or DEFAULT_INSTRUMENTS
        self.spot_td_mode, self.max_basis, self.max_tick_age_sec = spot_td_mode, max_basis, max_tick_age_sec
        self.order_timeout_sec, self.poll_sec, self.retries = order_timeout_sec, poll_sec, retries
        self.backoff_sec, self.read_ttl_sec, self.clock, self._sleep = backoff_sec, read_ttl_sec, clock, sleep
        self._funding: dict[str, _FundingCache] = {}
        self._reads: dict[tuple, tuple[float, Any]] = {}
        self._seq = 0

    # --- plumbing ----------------------------------------------------------
    def _call(self, fn: Callable[..., dict], what: str, **kw: Any) -> dict:
        last: ExchangeError | None = None
        for attempt in range(self.retries):
            try:
                resp = fn(**kw)
            except Exception as e:  # noqa: BLE001 - network/timeouts are retried
                last = ExchangeError(f"{what}: {type(e).__name__}: {e}")
            else:
                code = str(resp.get("code", ""))
                if code == "0":
                    return resp
                last = ExchangeError(f"{what}: code={code} msg={resp.get('msg', '')}", code=code)
                if code not in TRANSIENT_CODES:
                    raise last
            if attempt < self.retries - 1:
                self._sleep(self.backoff_sec * (2 ** attempt))
        assert last is not None
        raise last

    def _cached(self, key: tuple, fn: Callable[[], Any]) -> Any:
        now = self.clock()
        hit = self._reads.get(key)
        if hit is not None and now - hit[0] < self.read_ttl_sec:
            return hit[1]
        val = fn()
        self._reads[key] = (now, val)
        return val

    # --- reads -------------------------------------------------------------
    def _ticker(self, inst_id: str) -> float:
        d = self._call(self.market.get_ticker, f"ticker {inst_id}", instId=inst_id)["data"][0]
        last, ts = _f(d["last"]), int(d["ts"])
        if abs(self.clock() * 1000 - ts) > self.max_tick_age_sec * 1000:
            raise ExchangeError(f"{inst_id}: stale ticker (age {abs(self.clock() * 1000 - ts) / 1000:.0f}s)")
        if not last > 0:
            raise ExchangeError(f"{inst_id}: bad price {last}")
        return last

    def price(self, coin: str) -> float:
        inst = self.insts[coin]
        spot, swap = self._ticker(inst.spot_id), self._ticker(inst.swap_id)
        if abs(spot - swap) / spot > self.max_basis:
            raise ExchangeError(f"{coin}: spot {spot} and swap {swap} diverge by more than {self.max_basis:.1%}")
        return spot

    def funding_history(self, coin: str, n: int) -> list[float]:
        if n <= 0:
            return []
        if n > 100:
            raise ExchangeError("funding_history: n must be <= 100")
        c = self._funding.get(coin)
        if c is None or c.limit < n or self.clock() * 1000 >= c.expires_ms:
            rows = self._call(self.public.funding_rate_history, f"funding history {coin}",
                              instId=self.insts[coin].swap_id, limit=str(n))["data"]
            pts = sorted((int(r["fundingTime"]), _f(r["fundingRate"])) for r in rows)
            if not pts:
                raise ExchangeError(f"{coin}: no funding history returned")
            c = _FundingCache([r for _, r in pts], n, pts[-1][0] + SETTLEMENT_MS + 60_000)
            self._funding[coin] = c
        return c.rates[-n:]

    def spot_balance(self, coin: str) -> float:
        def fetch() -> float:
            data = self._call(self.account.get_account_balance, f"balance {coin}", ccy=coin)["data"]
            for det in (data[0].get("details", []) if data else []):
                if det.get("ccy") == coin:
                    return _f(det.get("cashBal"))
            return 0.0
        return self._cached(("bal", coin), fetch)

    def _positions(self, coin: str) -> list[dict]:
        inst_id = self.insts[coin].swap_id

        def fetch() -> list[dict]:
            rows = self._call(self.account.get_positions, f"positions {coin}",
                              instType="SWAP", instId=inst_id)["data"]
            return [r for r in rows if r.get("instId") == inst_id]
        return self._cached(("pos", coin), fetch)

    def swap_contracts(self, coin: str) -> float:
        total = 0.0
        for r in self._positions(coin):
            p, side = _f(r.get("pos")), r.get("posSide", "net")
            if side == "short":
                total += abs(p)
            elif side == "long":
                if p != 0:
                    raise ExchangeError(f"{coin}: unexpected LONG swap position {p}")
            else:                                                 # net mode: negative = short
                if p > 0:
                    raise ExchangeError(f"{coin}: unexpected LONG swap position {p}")
                total += -p
        return total

    def swap_adl_rank(self, coin: str) -> int:
        best = 0
        for r in self._positions(coin):
            try:
                best = max(best, int(r.get("adl") or 0))
            except ValueError:
                pass
        return best

    def mgn_ratio(self) -> float | None:
        def fetch() -> float | None:
            data = self._call(self.account.get_account_balance, "account balance")["data"]
            raw = data[0].get("mgnRatio") if data else None
            if raw in (None, ""):
                return None
            try:
                v = float(raw)
            except ValueError:
                return None
            return v if math.isfinite(v) else None
        return self._cached(("mgn",), fetch)
```

- [ ] **Step 4: Run to verify it passes**

Run: `$PY -m pytest tests/test_carry_okx_read.py -q`
Expected: 17 passed

- [ ] **Step 5: Commit**

```bash
git add src/carry/okx_exchange.py tests/test_carry_okx_read.py
git commit -m "feat(carry): OKX adapter read side (prices, funding, balances, positions, margin)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 4: OKX adapter — write side (idempotent market orders)

**Files:**
- Modify: `src/carry/okx_exchange.py` (append methods to `OkxExchange`, add `_fmt`)
- Test: `tests/test_carry_okx_write.py`

**Interfaces:**
- Consumes: everything from Task 3; `OrderResult`.
- Produces: `sell_swap(coin, contracts)`, `buy_swap(coin, contracts)` (`reduceOnly="true"`), `buy_spot(coin, qty)`, `sell_spot(coin, qty)` (`tgtCcy="base_ccy"`, `tdMode=self.spot_td_mode`). All are MARKET orders; swaps use `tdMode="cross"`, net mode (no `posSide`). Every order carries a unique `clOrdId` (`"carry"` + 13-digit ms timestamp + 3-digit sequence, <= 32 alphanumerics). `_place` clears the read cache, places the order via `_call` (retries reuse the SAME `clOrdId`, so a duplicate can never be created), and then polls `get_order(instId, clOrdId)` every `poll_sec` until a terminal state (`filled`/`canceled`/`mmp_canceled`) or `order_timeout_sec` (then cancels and re-reads once). Result: `OrderResult(ok=state=="filled", filled=accFillSz, avg_price=avgPx, error="" or f"state={state}")`. Definitive rejections (`sCode != "0"` in the row, or a non-transient top-level code) return `OrderResult(False, error=...)` WITHOUT polling. An ambiguous placement (exception or exhausted transient code) is resolved by polling `clOrdId`; if the order cannot be found, the result is `OrderResult(False, error=<placement error>)`. `_fmt(x)` renders a number without scientific notation or trailing zeros (`0.2 -> "0.2"`, `1e-8 -> "0.00000001"`, `100.0 -> "100"`).

- [ ] **Step 1: Write the failing test**

```python
import re

import pytest

from carry.okx_exchange import OkxExchange, _fmt

NOW_MS = 1_800_000_000_000


class Clock:
    def __init__(self):
        self.t, self.slept = NOW_MS / 1000, []

    def now(self):
        return self.t

    def sleep(self, s):
        self.slept.append(s)
        self.t += s


def ok(data):
    return {"code": "0", "msg": "", "data": data}


FILLED = {"state": "filled", "accFillSz": "0.2", "avgPx": "60000"}


class FakeTrade:
    def __init__(self, states=(FILLED,)):
        self.states = list(states)
        self.placed, self.cancelled, self.get_calls = [], [], 0
        self.errors, self.reject, self.top_code = [], False, None

    def place_order(self, **kw):
        self.placed.append(kw)
        if self.errors:
            raise self.errors.pop(0)
        if self.top_code:
            return {"code": self.top_code, "msg": "bad", "data": []}
        if self.reject:
            return ok([{"sCode": "51008", "sMsg": "insufficient balance", "clOrdId": kw["clOrdId"]}])
        return ok([{"sCode": "0", "ordId": "1", "clOrdId": kw["clOrdId"]}])

    def get_order(self, instId, ordId="", clOrdId=""):
        self.get_calls += 1
        row = self.states[0] if len(self.states) == 1 else self.states.pop(0)
        return ok([dict(row, instId=instId, clOrdId=clOrdId)])

    def cancel_order(self, instId, ordId="", clOrdId=""):
        self.cancelled.append(clOrdId)
        return ok([{"sCode": "0"}])


class FakeAccount:
    def __init__(self):
        self.bal_calls = 0

    def get_account_balance(self, ccy=""):
        self.bal_calls += 1
        return ok([{"details": [{"ccy": "BTC", "cashBal": "0.002"}], "mgnRatio": ""}])


def make(trade, **kw):
    clock = Clock()
    return OkxExchange(trade, FakeAccount(), None, None, clock=clock.now, sleep=clock.sleep, **kw), clock


def test_fmt():
    assert _fmt(0.2) == "0.2" and _fmt(1e-8) == "0.00000001" and _fmt(100.0) == "100" and _fmt(1.5) == "1.5"


def test_sell_swap_open_short_params_and_fill():
    tr = FakeTrade()
    ex, _ = make(tr)
    r = ex.sell_swap("BTC", 0.2)
    kw = tr.placed[0]
    assert (kw["instId"], kw["tdMode"], kw["side"], kw["ordType"], kw["sz"]) == \
        ("BTC-USDT-SWAP", "cross", "sell", "market", "0.2")
    assert kw["clOrdId"].startswith("carry")
    assert "reduceOnly" not in kw and "posSide" not in kw and "tgtCcy" not in kw
    assert r.ok and r.filled == pytest.approx(0.2) and r.avg_price == pytest.approx(60000.0)


def test_buy_swap_is_reduce_only():
    tr = FakeTrade()
    make(tr)[0].buy_swap("BTC", 0.2)
    kw = tr.placed[0]
    assert kw["side"] == "buy" and kw["reduceOnly"] == "true" and kw["tdMode"] == "cross"


def test_spot_orders_use_base_currency_target_and_configured_td_mode():
    tr = FakeTrade()
    ex, _ = make(tr, spot_td_mode="cash")
    ex.buy_spot("BTC", 0.00200201)
    ex.sell_spot("BTC", 0.002)
    buy, sell = tr.placed
    assert (buy["instId"], buy["side"], buy["tgtCcy"], buy["tdMode"], buy["sz"]) == \
        ("BTC-USDT", "buy", "base_ccy", "cash", "0.00200201")
    assert (sell["side"], sell["tgtCcy"], sell["sz"]) == ("sell", "base_ccy", "0.002")
    assert "reduceOnly" not in buy


def test_partial_then_canceled_reports_partial_fill():
    tr = FakeTrade([{"state": "partially_filled", "accFillSz": "0.1", "avgPx": "60000"},
                    {"state": "canceled", "accFillSz": "0.1", "avgPx": "60000"}])
    r = make(tr)[0].sell_swap("BTC", 0.2)
    assert not r.ok and r.filled == pytest.approx(0.1) and "canceled" in r.error


def test_live_order_at_timeout_is_cancelled():
    tr = FakeTrade([{"state": "live", "accFillSz": "0", "avgPx": ""}])
    r = make(tr, order_timeout_sec=2.0)[0].sell_swap("BTC", 0.2)
    assert len(tr.cancelled) == 1 and not r.ok and "live" in r.error


def test_ambiguous_placement_is_resolved_by_client_order_id():
    tr = FakeTrade()
    tr.errors = [ConnectionError("t")] * 3               # all 3 placement attempts time out
    r = make(tr)[0].sell_swap("BTC", 0.2)
    assert len({p["clOrdId"] for p in tr.placed}) == 1 and len(tr.placed) == 3
    assert r.ok and r.filled == pytest.approx(0.2)


def test_ambiguous_placement_with_unknown_order_fails_cleanly():
    tr = FakeTrade([{"state": "live", "accFillSz": "0"}])
    tr.errors = [ConnectionError("t")] * 3
    tr.get_order = lambda **kw: {"code": "51603", "msg": "order not found", "data": []}
    r = make(tr, order_timeout_sec=2.0)[0].sell_swap("BTC", 0.2)
    assert not r.ok and r.filled == 0.0 and "ConnectionError" in r.error


def test_row_rejection_does_not_poll():
    tr = FakeTrade()
    tr.reject = True
    r = make(tr)[0].buy_spot("BTC", 0.002)
    assert not r.ok and "51008" in r.error and tr.get_calls == 0


def test_non_transient_top_level_code_does_not_poll_or_retry():
    tr = FakeTrade()
    tr.top_code = "51000"
    r = make(tr)[0].sell_swap("BTC", 0.2)
    assert not r.ok and tr.get_calls == 0 and len(tr.placed) == 1


def test_client_order_ids_are_unique_and_valid():
    tr = FakeTrade()
    ex, _ = make(tr)
    ex.sell_swap("BTC", 0.2)
    ex.sell_swap("BTC", 0.2)
    a, b = (p["clOrdId"] for p in tr.placed)
    assert a != b and all(re.fullmatch(r"[A-Za-z0-9]{1,32}", x) for x in (a, b))


def test_write_clears_the_read_cache():
    tr = FakeTrade()
    ex, _ = make(tr)
    ex.spot_balance("BTC")
    ex.spot_balance("BTC")
    assert ex.account.bal_calls == 1
    ex.sell_swap("BTC", 0.2)
    ex.spot_balance("BTC")
    assert ex.account.bal_calls == 2
```

- [ ] **Step 2: Run to verify it fails**

Run: `$PY -m pytest tests/test_carry_okx_write.py -q`
Expected: FAIL with `ImportError: cannot import name '_fmt'`

- [ ] **Step 3: Implement** — add `from decimal import Decimal` and `from carry.types import DEFAULT_INSTRUMENTS, Instrument, OrderResult` to the imports of `okx_exchange.py`, then add:

```python
def _fmt(x: float) -> str:
    return format(Decimal(str(round(x, 12))).normalize(), "f")
```

and append to `OkxExchange`:

```python
    # --- writes (market orders, idempotent by clOrdId) ------------------------
    def _next_cl_ord_id(self) -> str:
        self._seq += 1
        return f"carry{int(self.clock() * 1000)}{self._seq % 1000:03d}"

    def _place(self, inst_id: str, side: str, sz: float, td_mode: str, *,
               reduce_only: bool = False, base_target: bool = False) -> OrderResult:
        self._reads.clear()
        cl = self._next_cl_ord_id()
        kw: dict[str, Any] = dict(instId=inst_id, tdMode=td_mode, side=side, ordType="market",
                                  sz=_fmt(sz), clOrdId=cl)
        if reduce_only:
            kw["reduceOnly"] = "true"
        if base_target:
            kw["tgtCcy"] = "base_ccy"
        unknown = ""
        try:
            resp = self._call(self.trade.place_order, f"place {side} {inst_id}", **kw)
            row = (resp.get("data") or [{}])[0]
            if str(row.get("sCode", "0")) != "0":
                return OrderResult(False, error=f"sCode={row.get('sCode')} {row.get('sMsg', '')}".strip())
        except ExchangeError as e:
            if e.code is not None and e.code not in TRANSIENT_CODES:
                return OrderResult(False, error=str(e))          # definitive rejection
            unknown = str(e)                                     # outcome unknown: resolve by clOrdId
        res = self._wait_terminal(inst_id, cl)
        self._reads.clear()
        if unknown and not res.ok and res.filled <= 0:
            return OrderResult(False, error=unknown)
        return res

    def _read_order(self, inst_id: str, cl: str) -> dict | None:
        try:
            data = self._call(self.trade.get_order, "get_order", instId=inst_id, clOrdId=cl)["data"]
            return data[0] if data else None
        except ExchangeError:
            return None                                          # not found yet / transient

    def _wait_terminal(self, inst_id: str, cl: str) -> OrderResult:
        deadline = self.clock() + self.order_timeout_sec
        last: dict | None = None
        while True:
            row = self._read_order(inst_id, cl)
            if row is not None:
                last = row
                if row.get("state") in TERMINAL_STATES:
                    break
            if self.clock() >= deadline:
                break
            self._sleep(self.poll_sec)
        if last is None:
            return OrderResult(False, error="order status unknown")
        if last.get("state") not in TERMINAL_STATES:
            try:
                self._call(self.trade.cancel_order, "cancel_order", instId=inst_id, clOrdId=cl)
            except ExchangeError:
                pass
            last = self._read_order(inst_id, cl) or last
        state = str(last.get("state"))
        good = state == "filled"
        return OrderResult(good, _f(last.get("accFillSz")), _f(last.get("avgPx")),
                           "" if good else f"state={state}")

    def sell_swap(self, coin: str, contracts: float) -> OrderResult:
        return self._place(self.insts[coin].swap_id, "sell", contracts, "cross")

    def buy_swap(self, coin: str, contracts: float) -> OrderResult:
        return self._place(self.insts[coin].swap_id, "buy", contracts, "cross", reduce_only=True)

    def buy_spot(self, coin: str, qty: float) -> OrderResult:
        return self._place(self.insts[coin].spot_id, "buy", qty, self.spot_td_mode, base_target=True)

    def sell_spot(self, coin: str, qty: float) -> OrderResult:
        return self._place(self.insts[coin].spot_id, "sell", qty, self.spot_td_mode, base_target=True)
```

- [ ] **Step 4: Run to verify it passes**

Run: `$PY -m pytest tests/test_carry_okx_read.py tests/test_carry_okx_write.py -q`
Expected: 29 passed (17 read + 12 write). If `test_ambiguous_placement_with_unknown_order_fails_cleanly` loops forever, the deadline arithmetic is wrong: `_wait_terminal` must advance only through the injected `sleep` and stop when `clock() >= deadline`.

- [ ] **Step 5: Commit**

```bash
git add src/carry/okx_exchange.py tests/test_carry_okx_write.py
git commit -m "feat(carry): OKX adapter write side with idempotent market orders

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 5: Bootstrap — state fallback, reconcile, exposure report

**Files:**
- Create: `src/carry/bootstrap.py`
- Test: `tests/test_carry_bootstrap.py`

**Interfaces:**
- Consumes: `load_state`, `CarryState`, `CoinLeg` (state); `check_hedge`, `HedgeStatus` (hedge); `CarryConfig`; `Instrument`; `Exchange`.
- Produces: `load_state_or_halt(path, alert, clock=time.time) -> CarryState` — a missing file gives the default FLAT state; an unreadable/corrupt file is renamed to `<name>.corrupt-<int(clock())>`, a `[CRITICAL]` alert is sent and a `HALTED` state (`halted_reason="state file unreadable: <ExcClass>"`) is returned (never default-and-trade). `reconcile(ex, insts, cfg, state, alert, now_ms) -> list[str]` — for every coin with no recorded leg: exchange FLAT -> nothing; a healthy hedge (`HedgeStatus.OK`) whose perp notional `contracts*ct_val*price` is `<= 1.2*cfg.max_order_notional` is ADOPTED (`state.legs[coin] = CoinLeg(contracts, spot_qty=spot balance, entry_ts=now_ms)`, `state.mode = "HEDGED"`) only when `state.mode != "HALTED"`; every other non-flat case is NOT adopted and produces a `[CRITICAL]` alert naming the coin and the `HedgeStatus` (the engine's holdings guard then halts it). It never trades. Returns human-readable notes. `exposure_report(ex, insts, cfg, state) -> str` — multi-line text with mode, halted reason, recorded legs, per coin `spot`, `perp` contracts, hedge status, ADL rank, and the margin ratio; never raises (returns `"exposure unavailable: <ExcClass>: <msg>"` on any failure).

- [ ] **Step 1: Write the failing test**

```python
import json

import pytest

from carry.bootstrap import exposure_report, load_state_or_halt, reconcile
from carry.config import CarryConfig
from carry.fake_exchange import FakeExchange
from carry.state import CarryState, CoinLeg, save_state
from carry.types import DEFAULT_INSTRUMENTS

CFG = CarryConfig()


def hedged_exchange(contracts=0.2, spot_qty=0.00200201):
    ex = FakeExchange()
    ex.sell_swap("BTC", contracts)
    ex.buy_spot("BTC", spot_qty)
    return ex


def test_missing_state_file_gives_default(tmp_path):
    st = load_state_or_halt(tmp_path / "none.json", [].append)
    assert st.mode == "FLAT" and st.legs == {}


def test_valid_state_file_is_loaded(tmp_path):
    p = tmp_path / "s.json"
    save_state(p, CarryState(mode="HEDGED", legs={"BTC": CoinLeg(0.2, 0.002, 5)}))
    assert load_state_or_halt(p, [].append).legs["BTC"].contracts == 0.2


def test_corrupt_state_starts_halted_and_keeps_evidence(tmp_path):
    p = tmp_path / "s.json"
    p.write_text("{not json")
    alerts = []
    st = load_state_or_halt(p, alerts.append, clock=lambda: 1234.0)
    assert st.mode == "HALTED" and "unreadable" in st.halted_reason
    assert not p.exists() and (tmp_path / "s.json.corrupt-1234").exists()
    assert any("CRITICAL" in a for a in alerts)


def test_unknown_leg_keys_also_start_halted(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"mode": "HEDGED", "legs": {"BTC": {"contracts": 1, "bogus": 2}}}))
    assert load_state_or_halt(p, [].append).mode == "HALTED"


def test_reconcile_flat_exchange_is_a_noop():
    st = CarryState()
    assert reconcile(FakeExchange(), DEFAULT_INSTRUMENTS, CFG, st, [].append, 1000) == []
    assert st.mode == "FLAT" and st.legs == {}


def test_reconcile_adopts_a_healthy_hedge():
    ex, st = hedged_exchange(), CarryState()
    notes = reconcile(ex, DEFAULT_INSTRUMENTS, CFG, st, [].append, 1000)
    assert st.mode == "HEDGED" and st.legs["BTC"].contracts == pytest.approx(0.2)
    assert st.legs["BTC"].spot_qty == pytest.approx(0.002, rel=1e-3) and st.legs["BTC"].entry_ts == 1000
    assert any("adopted" in n and "BTC" in n for n in notes)


def test_reconcile_does_not_adopt_when_halted():
    st = CarryState(mode="HALTED", halted_reason="x")
    alerts = []
    reconcile(hedged_exchange(), DEFAULT_INSTRUMENTS, CFG, st, alerts.append, 1000)
    assert st.legs == {} and st.mode == "HALTED" and any("CRITICAL" in a for a in alerts)


def test_reconcile_never_adopts_a_naked_short():
    ex = FakeExchange()
    ex.sell_swap("BTC", 0.2)
    st, alerts = CarryState(), []
    reconcile(ex, DEFAULT_INSTRUMENTS, CFG, st, alerts.append, 1000)
    assert st.legs == {} and any("PERP_ONLY" in a and "CRITICAL" in a for a in alerts)
    assert ex.swap["BTC"] == 0.2                        # and it did not trade


def test_reconcile_does_not_adopt_an_oversized_hedge():
    ex = hedged_exchange(contracts=20.0, spot_qty=0.2002002)      # ~12000 USDT notional
    st, alerts = CarryState(), []
    reconcile(ex, DEFAULT_INSTRUMENTS, CFG, st, alerts.append, 1000)
    assert st.legs == {} and any("CRITICAL" in a for a in alerts)


def test_exposure_report_contents():
    ex = hedged_exchange()
    ex.adl["BTC"] = 3
    text = exposure_report(ex, DEFAULT_INSTRUMENTS, CFG, CarryState(mode="HEDGED", legs={"BTC": CoinLeg(0.2, 0.002, 1)}))
    for needle in ("mode=HEDGED", "BTC:", "perp=0.2", "hedge=OK", "adl=3", "ETH:", "mgnRatio=10.0"):
        assert needle in text


def test_exposure_report_never_raises():
    ex = FakeExchange()

    def boom():
        raise ConnectionError("down")

    ex.mgn_ratio = boom
    assert exposure_report(ex, DEFAULT_INSTRUMENTS, CFG, CarryState()).startswith("exposure unavailable: ConnectionError")
```

- [ ] **Step 2: Run to verify it fails**

Run: `$PY -m pytest tests/test_carry_bootstrap.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'carry.bootstrap'`

- [ ] **Step 3: Write the implementation**

```python
"""Startup safety: unreadable state starts HALTED, reconcile adopts only healthy hedges, exposure report."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

from carry.config import CarryConfig
from carry.exchange import Exchange
from carry.hedge import HedgeStatus, check_hedge
from carry.state import CarryState, CoinLeg, load_state
from carry.types import Instrument


def load_state_or_halt(path: str | Path, alert: Callable[[str], None],
                       clock: Callable[[], float] = time.time) -> CarryState:
    try:
        return load_state(path)
    except Exception as e:  # noqa: BLE001 - corrupt JSON, unknown keys, bad types
        p = Path(path)
        bad = p.with_name(f"{p.name}.corrupt-{int(clock())}")
        try:
            p.rename(bad)
        except OSError:
            pass
        alert(f"[CRITICAL] state file unreadable ({type(e).__name__}); moved to {bad.name}; starting HALTED")
        return CarryState(mode="HALTED", halted_reason=f"state file unreadable: {type(e).__name__}")


def reconcile(ex: Exchange, insts: dict[str, Instrument], cfg: CarryConfig, state: CarryState,
              alert: Callable[[str], None], now_ms: int) -> list[str]:
    """Compare the exchange with the state at startup. Adopts healthy hedges only; never trades."""
    notes: list[str] = []
    for coin in cfg.coins:
        if coin in state.legs:
            continue
        inst = insts[coin]
        spot, contracts = ex.spot_balance(coin), ex.swap_contracts(coin)
        status = check_hedge(spot, contracts, inst, cfg.drift_tol)
        if status == HedgeStatus.FLAT:
            continue
        notional = contracts * inst.ct_val * ex.price(coin)
        if (status == HedgeStatus.OK and state.mode != "HALTED"
                and notional <= 1.2 * cfg.max_order_notional):
            state.legs[coin] = CoinLeg(contracts=contracts, spot_qty=spot, entry_ts=now_ms)
            state.mode = "HEDGED"
            notes.append(f"adopted existing {coin} hedge: {contracts} contracts, spot {spot}")
            continue
        msg = (f"[CRITICAL] startup: {coin} exchange holdings not adopted "
               f"(status={status.value}, spot={spot}, perp={contracts} contracts, mode={state.mode})")
        alert(msg)
        notes.append(msg)
    return notes


def exposure_report(ex: Exchange, insts: dict[str, Instrument], cfg: CarryConfig, state: CarryState) -> str:
    try:
        lines = [f"mode={state.mode} reason={state.halted_reason or '-'} legs={sorted(state.legs)}"]
        for coin in cfg.coins:
            inst = insts[coin]
            spot, contracts = ex.spot_balance(coin), ex.swap_contracts(coin)
            status = check_hedge(spot, contracts, inst, cfg.drift_tol)
            lines.append(f"{coin}: spot={spot:.8g} perp={contracts:.8g} contracts "
                         f"hedge={status.value} adl={ex.swap_adl_rank(coin)}")
        lines.append(f"mgnRatio={ex.mgn_ratio()}")
        return "\n".join(lines)
    except Exception as e:  # noqa: BLE001
        return f"exposure unavailable: {type(e).__name__}: {e}"
```

- [ ] **Step 4: Run to verify it passes**

Run: `$PY -m pytest tests/test_carry_bootstrap.py -q`
Expected: 11 passed. (For `test_reconcile_does_not_adopt_when_halted` the healthy hedge must alert CRITICAL; check that the message in the code path for `HALTED` is the not-adopted message.)

- [ ] **Step 5: Commit**

```bash
git add src/carry/bootstrap.py tests/test_carry_bootstrap.py
git commit -m "feat(carry): startup state fallback, reconcile and exposure report

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 6: Runner and instance lock

**Files:**
- Create: `src/carry/runner.py`
- Test: `tests/test_carry_runner.py`

**Interfaces:**
- Produces: `InstanceLock(path)` with `acquire()` (exclusive non-blocking `fcntl.flock`; raises `RuntimeError("another carry instance holds the lock")` when held), `release()`, and context-manager support. `Runner(engine, report_fn, alert, cfg, *, heartbeat_sec=3600.0, clock=time.time, sleep=time.sleep)` with `run(should_stop: Callable[[], bool]) -> None`: loop `while not should_stop()`: `actions = engine.step(int(clock()*1000))`, alert each action as `"[INFO] <action>"`, send `"[HEARTBEAT]\n" + report_fn()` when `heartbeat_sec` has elapsed since the last one (the first iteration always heartbeats), any `Exception` from the step is alerted as `"[CRITICAL] runner: step raised <repr>"` and the loop continues, then `sleep(cfg.poll_sec)`. `report_fn` failures must not stop the loop (wrapped in try/except).

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `$PY -m pytest tests/test_carry_runner.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'carry.runner'`

- [ ] **Step 3: Write the implementation**

```python
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
```

- [ ] **Step 4: Run to verify it passes**

Run: `$PY -m pytest tests/test_carry_runner.py -q`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/carry/runner.py tests/test_carry_runner.py
git commit -m "feat(carry): run loop with heartbeat and single-instance lock

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 7: Preflight and CLI

> **Amendments from the Batch B/A reviews (binding, apply while implementing):** (1) `make_clients` is called as `make_clients(env, cfg, alerter)` and `default_clients(env, cfg, alert=None)` passes `alert=alerter` to `OkxExchange` so the adapter can raise CRITICAL alerts when an order cannot be confirmed; the code and test lambdas below already use the 3-argument form. (2) The default spot trade mode is `"cash"`. (3) The CLI must call `alerter.flush()` (if the alerter has it) before returning from EVERY command path — `alert-test`, `preflight`, `status`, the manual commands and the end of `run` including SIGTERM/SIGINT shutdown — implement it once by wrapping the command execution in `try/finally`; add a test that a fake alerter's `flush` was called after `alert-test` and after `run --max-steps 1`. (4) `rearm` must also call `exchange.clear_unconfirmed()` when the exchange object has that method (the adapter blocks new orders while an order is unconfirmed) — add a test with a fake exchange exposing `clear_unconfirmed`. (5) The read-only guard already permits trade verbs in `src/carry/cli.py` and `src/carry/okx_exchange.py` only.

**Files:**
- Create: `src/carry/preflight.py`, `src/carry/cli.py`, `src/carry/__main__.py`
- Test: `tests/test_carry_preflight.py`, `tests/test_carry_cli.py`

**Interfaces:**
- Consumes: `OkxExchange`/`ExchangeError` (only in `default_clients`), `build_alerter`, `load_state_or_halt`, `reconcile`, `exposure_report`, `Runner`, `InstanceLock`, `Engine`, `save_state`, `CarryConfig`, `DEFAULT_INSTRUMENTS`, `FakeExchange` (tests).
- Produces: `PreflightResult(problems: list[str], warnings: list[str])` with `ok` property; `preflight(account, public, cfg, insts) -> PreflightResult` flags as PROBLEMS: `acctLv != "3"`, `autoLoan` on, `posMode != "net_mode"`, any balance of a `cfg.coins` currency `>= spot_min`, USDT available `< 1.2 * cfg.deploy_usdt`, any non-zero SWAP position, instrument specs (ctVal, lotSz, minSz for swap; lotSz, minSz for spot) differing from `insts`, and any API read failure ("could not read ...: <ExcClass>"); non-BTC/ETH non-USDT balances are WARNINGS. `Clients(exchange, account, public)` dataclass; `load_env_file(path) -> dict[str, str]` (`KEY=VALUE`, optional `export`, `#` comments, matching quotes stripped); `load_config(path) -> CarryConfig` (JSON; unknown keys -> `ValueError`; `coins` list -> tuple); `default_clients(env, cfg, alert=None) -> Clients` (needs `OKX_API_KEY/SECRET/PASSPHRASE`, else `SystemExit` with a clear message; `OKX_DEMO` in `1/true/yes` -> flag `"1"`; builds python-okx `TradeAPI/AccountAPI/PublicAPI/MarketAPI` with `flag=` and `debug=False`, imports `okx` lazily; `spot_td_mode` from `OKX_SPOT_TD_MODE`, default `"cash"` (the adapter's default); the `OkxExchange` also gets `alert=alerter`; it gets `order_timeout_sec`, `read_ttl_sec` defaults). `main(argv=None, env=None, *, make_clients=None, make_alerter=None, out=print, clock=time.time, sleep=time.sleep) -> int`. Global options: `--config PATH`, `--env-file PATH`, `--log PATH` (default `live_state/carry.log`). Commands: `alert-test`, `preflight` (exit 0/1), `status` (no lock; prints `exposure_report`), `run [--max-steps N] [--heartbeat-sec S]`, `force-open COIN NOTIONAL`, `force-close COIN`, `rearm`. `run`, `force-*`, `rearm` first acquire `InstanceLock(cfg.state_path + ".lock")` (held -> print message, exit 3), build the engine (`load_state_or_halt`, `reconcile`, `Engine(..., save=save_state)`), and return 1 with the message on `ValueError`/`RuntimeError`. `run` installs SIGTERM/SIGINT handlers that stop the loop after the current step and restores the old handlers afterwards; `--max-steps` bounds the loop (for demos/tests). Config/env errors -> exit 2. `src/carry/__main__.py` is `from carry.cli import main; raise SystemExit(main())`.

- [ ] **Step 1: Write the failing tests**

`tests/test_carry_preflight.py`:

```python
import pytest

from carry.config import CarryConfig
from carry.preflight import preflight
from carry.types import DEFAULT_INSTRUMENTS

CFG = CarryConfig()


def ok(data):
    return {"code": "0", "msg": "", "data": data}


class FakeAccount:
    def __init__(self, acct_lv="3", auto_loan=False, pos_mode="net_mode", details=None, positions=None):
        self.acct_lv, self.auto_loan, self.pos_mode = acct_lv, auto_loan, pos_mode
        self.details = details if details is not None else [{"ccy": "USDT", "cashBal": "1000", "availBal": "1000"}]
        self.positions, self.fail = positions or [], False

    def get_account_config(self):
        if self.fail:
            raise ConnectionError("down")
        return ok([{"acctLv": self.acct_lv, "autoLoan": self.auto_loan, "posMode": self.pos_mode}])

    def get_account_balance(self, ccy=""):
        return ok([{"details": self.details, "mgnRatio": ""}])

    def get_positions(self, instType="", instId="", posId=""):
        return ok(self.positions)


class FakePublic:
    def __init__(self, overrides=None):
        self.overrides = overrides or {}

    def get_instruments(self, instType, instId=""):
        coin = instId.split("-")[0]
        i = DEFAULT_INSTRUMENTS[coin]
        if instType == "SWAP":
            row = {"ctVal": str(i.ct_val), "lotSz": str(i.swap_lot), "minSz": str(i.swap_min)}
        else:
            row = {"lotSz": str(i.spot_lot), "minSz": str(i.spot_min)}
        row.update(self.overrides.get(instId, {}))
        return ok([row])


def run(**kw):
    public = kw.pop("public", FakePublic())
    return preflight(FakeAccount(**kw), public, CFG, DEFAULT_INSTRUMENTS)


def test_healthy_account_passes():
    r = run()
    assert r.ok and r.problems == []


def test_account_level_autoloan_and_position_mode_are_checked():
    assert any("acctLv" in p for p in run(acct_lv="2").problems)
    assert any("autoLoan" in p for p in run(auto_loan=True).problems)
    assert any("posMode" in p for p in run(pos_mode="long_short_mode").problems)


def test_stray_coin_balance_is_a_problem_but_dust_below_min_is_not():
    usdt = {"ccy": "USDT", "cashBal": "1000", "availBal": "1000"}
    assert any("BTC" in p for p in run(details=[usdt, {"ccy": "BTC", "cashBal": "0.5"}]).problems)
    assert run(details=[usdt, {"ccy": "BTC", "cashBal": "0.000001"}]).ok


def test_other_coins_are_only_warnings():
    r = run(details=[{"ccy": "USDT", "cashBal": "1000", "availBal": "1000"}, {"ccy": "OKB", "cashBal": "5"}])
    assert r.ok and any("OKB" in w for w in r.warnings)


def test_not_enough_usdt_is_a_problem():
    assert any("USDT" in p for p in run(details=[{"ccy": "USDT", "cashBal": "100", "availBal": "100"}]).problems)


def test_open_swap_position_is_a_problem():
    assert any("position" in p for p in run(positions=[{"instId": "BTC-USDT-SWAP", "pos": "-0.2"}]).problems)


def test_changed_contract_spec_is_a_problem():
    assert any("ETH-USDT-SWAP" in p and "ctVal" in p
               for p in run(public=FakePublic({"ETH-USDT-SWAP": {"ctVal": "0.05"}})).problems)


def test_read_failure_is_reported_not_raised():
    acc = FakeAccount()
    acc.fail = True
    r = preflight(acc, FakePublic(), CFG, DEFAULT_INSTRUMENTS)
    assert any("could not read" in p and "ConnectionError" in p for p in r.problems)
```

`tests/test_carry_cli.py`:

```python
import json

import pytest

from carry.cli import Clients, load_config, load_env_file, main
from carry.fake_exchange import FakeExchange
from carry.runner import InstanceLock
from carry.state import load_state
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `$PY -m pytest tests/test_carry_preflight.py tests/test_carry_cli.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'carry.preflight'`

- [ ] **Step 3: Write the implementations**

`src/carry/preflight.py`:

```python
"""Pre-trade account checks. Read-only."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from carry.config import CarryConfig
from carry.types import Instrument


@dataclass
class PreflightResult:
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def _num(x: Any) -> float:
    return 0.0 if x in (None, "") else float(x)


def _rows(resp: dict, what: str) -> list[dict]:
    if str(resp.get("code")) != "0":
        raise RuntimeError(f"{what}: code={resp.get('code')} msg={resp.get('msg')}")
    return resp["data"]


def preflight(account: Any, public: Any, cfg: CarryConfig, insts: dict[str, Instrument]) -> PreflightResult:
    r = PreflightResult()

    def guarded(label: str, fn) -> None:
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            r.problems.append(f"could not read {label}: {type(e).__name__}: {e}")

    def check_config() -> None:
        c = _rows(account.get_account_config(), "account config")[0]
        if str(c.get("acctLv")) != "3":
            r.problems.append(f"acctLv is {c.get('acctLv')}, need 3 (multi-currency margin)")
        if c.get("autoLoan") not in (False, "false"):
            r.problems.append("autoLoan must be off (no borrowing in v1)")
        if c.get("posMode") != "net_mode":
            r.problems.append(f"posMode is {c.get('posMode')}, need net_mode")

    def check_balances() -> None:
        details = _rows(account.get_account_balance(), "balance")[0].get("details", [])
        usdt = 0.0
        for d in details:
            ccy, cash = d.get("ccy"), _num(d.get("cashBal"))
            if ccy in cfg.coins:
                if abs(cash) >= insts[ccy].spot_min:
                    r.problems.append(f"account holds {cash} {ccy}; the account must hold no {ccy} before the bot starts")
            elif ccy == "USDT":
                usdt = _num(d.get("availBal") if d.get("availBal") not in (None, "") else cash)
            elif abs(cash) > 0:
                r.warnings.append(f"other balance: {cash} {ccy}")
        if usdt < 1.2 * cfg.deploy_usdt:
            r.problems.append(f"USDT available {usdt} < 1.2 x deploy_usdt ({cfg.deploy_usdt})")

    def check_positions() -> None:
        for p in _rows(account.get_positions(instType="SWAP"), "positions"):
            if _num(p.get("pos")) != 0:
                r.problems.append(f"open swap position {p.get('instId')}: {p.get('pos')}")

    def check_specs() -> None:
        for coin in cfg.coins:
            i = insts[coin]
            swap = _rows(public.get_instruments(instType="SWAP", instId=i.swap_id), "swap spec")[0]
            spot = _rows(public.get_instruments(instType="SPOT", instId=i.spot_id), "spot spec")[0]
            for inst_id, row, want in (
                (i.swap_id, swap, {"ctVal": i.ct_val, "lotSz": i.swap_lot, "minSz": i.swap_min}),
                (i.spot_id, spot, {"lotSz": i.spot_lot, "minSz": i.spot_min}),
            ):
                for key, expected in want.items():
                    if not math.isclose(_num(row.get(key)), expected, rel_tol=1e-9):
                        r.problems.append(f"{inst_id} {key} is {row.get(key)}, expected {expected}")

    guarded("account config", check_config)
    guarded("balances", check_balances)
    guarded("positions", check_positions)
    guarded("instrument specs", check_specs)
    return r
```

`src/carry/cli.py`:

```python
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
from carry.preflight import preflight
from carry.runner import InstanceLock, Runner
from carry.state import save_state
from carry.types import DEFAULT_INSTRUMENTS


@dataclass
class Clients:
    exchange: Exchange
    account: Any
    public: Any


def load_env_file(path: str | Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in Path(path).read_text().splitlines():
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
    if "coins" in data:
        data["coins"] = tuple(data["coins"])
    return CarryConfig(**data)


def default_clients(env: Mapping[str, str], cfg: CarryConfig, alert: Callable[[str], None] | None = None) -> Clients:
    key, secret, phrase = (env.get(k) for k in ("OKX_API_KEY", "OKX_API_SECRET", "OKX_API_PASSPHRASE"))
    if not (key and secret and phrase):
        raise SystemExit("OKX_API_KEY / OKX_API_SECRET / OKX_API_PASSPHRASE must be set")
    flag = "1" if str(env.get("OKX_DEMO", "")).lower() in ("1", "true", "yes") else "0"
    import okx.Account as Account
    import okx.MarketData as MarketData
    import okx.PublicData as PublicData
    import okx.Trade as Trade
    from carry.okx_exchange import OkxExchange
    trade = Trade.TradeAPI(key, secret, phrase, False, flag=flag, debug=False)
    account = Account.AccountAPI(key, secret, phrase, False, flag=flag, debug=False)
    public = PublicData.PublicAPI(flag=flag, debug=False)
    market = MarketData.MarketAPI(flag=flag, debug=False)
    ex = OkxExchange(trade, account, public, market, DEFAULT_INSTRUMENTS,
                     spot_td_mode=env.get("OKX_SPOT_TD_MODE", "cash"), alert=alert)
    return Clients(ex, account, public)


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


def _engine(clients: Clients, cfg: CarryConfig, alerter: Callable[[str], None],
            clock: Callable[[], float], out: Callable[[str], None]) -> Engine:
    state = load_state_or_halt(cfg.state_path, alerter, clock)
    for note in reconcile(clients.exchange, DEFAULT_INSTRUMENTS, cfg, state, alerter, int(clock() * 1000)):
        out(f"reconcile: {note}")
    return Engine(clients.exchange, DEFAULT_INSTRUMENTS, cfg, alerter, state,
                  save=lambda s: save_state(cfg.state_path, s))


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

    if args.cmd == "alert-test":
        alerter("[INFO] carry alert test")
        out(f"channels: {getattr(alerter, 'channels', [])}")
        return 0

    clients = (make_clients or default_clients)(env, cfg, alerter)

    if args.cmd == "preflight":
        res = preflight(clients.account, clients.public, cfg, DEFAULT_INSTRUMENTS)
        for p in res.problems:
            out(f"PROBLEM: {p}")
        for w in res.warnings:
            out(f"warning: {w}")
        out("preflight OK" if res.ok else "preflight FAILED")
        return 0 if res.ok else 1

    if args.cmd == "status":
        state = load_state_or_halt(cfg.state_path, alerter, clock)
        out(exposure_report(clients.exchange, DEFAULT_INSTRUMENTS, cfg, state))
        return 0

    lock = InstanceLock(cfg.state_path + ".lock")
    try:
        lock.acquire()
    except RuntimeError as e:
        out(str(e))
        return 3
    try:
        eng = _engine(clients, cfg, alerter, clock, out)
        if args.cmd == "rearm":
            eng.rearm()
            out("rearmed")
            return 0
        if args.cmd == "force-open":
            ok = eng.force_open(args.coin, args.notional, int(clock() * 1000))
            out(f"force-open {args.coin}: {'ok' if ok else 'FAILED'}")
            return 0 if ok else 1
        if args.cmd == "force-close":
            ok = eng.force_close(args.coin)
            out(f"force-close {args.coin}: {'ok' if ok else 'FAILED'}")
            return 0 if ok else 1
        stop = _StopFlag(args.max_steps)                       # cmd == "run"
        old = {s: signal.signal(s, lambda *_: stop.trigger()) for s in (signal.SIGTERM, signal.SIGINT)}
        try:
            Runner(eng, lambda: exposure_report(clients.exchange, DEFAULT_INSTRUMENTS, cfg, eng.state), alerter,
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
```

`src/carry/__main__.py`:

```python
from carry.cli import main

raise SystemExit(main())
```

- [ ] **Step 4: Run to verify it passes**

Run: `$PY -m pytest tests/test_carry_preflight.py tests/test_carry_cli.py -q`
Expected: 8 + 12 = 20 passed. Notes: the CLI tests import `tests.test_carry_preflight`; if the import fails, add `tests/__init__.py` ONLY if it does not already exist, or copy the two small fakes into `tests/test_carry_cli.py` instead — do not restructure other tests. `test_force_open_close_and_rearm_persist_state`: after `force-close` the engine mode is FLAT, so `rearm()` passes the "no legs / exchange flat" checks.

- [ ] **Step 5: Commit**

```bash
git add src/carry/preflight.py src/carry/cli.py src/carry/__main__.py tests/test_carry_preflight.py tests/test_carry_cli.py
git commit -m "feat(carry): preflight checks and command-line interface

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 8: Deployment files and runbook

**Files:**
- Create: `config/carry.example.json`, `deploy/carry.service`, `deploy/carry.env.example`, `deploy/README-carry.md`
- Test: `tests/test_carry_deploy_files.py`

**Interfaces:**
- Consumes: `load_config`, `load_env_file` (cli).
- Produces: an example config that loads through `load_config` (with `settle_sec` 1.0 for live use, `deploy_usdt` 250, `poll_sec` 2.0, `state_path` `live_state/carry_state.json`); a systemd unit following the repo's existing convention (`User=ubuntu`, `WorkingDirectory=/home/ubuntu/crypto_backtest`, `EnvironmentFile=/home/ubuntu/.secrets/carry.env`, `Restart=on-failure`, `RestartSec=30`, `KillSignal=SIGTERM`, `TimeoutStopSec=60`, `Environment=PYTHONPATH=...src`); an env template with EMPTY values and comments only; a runbook.

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

from carry.cli import load_config, load_env_file

ROOT = Path(__file__).resolve().parents[1]


def test_example_config_loads_and_has_live_settle_delay():
    c = load_config(ROOT / "config" / "carry.example.json")
    assert c.settle_sec >= 1.0 and c.coins == ("BTC", "ETH") and c.deploy_usdt == 250.0


def test_env_example_lists_all_variables_with_empty_values():
    env = load_env_file(ROOT / "deploy" / "carry.env.example")
    for k in ("OKX_API_KEY", "OKX_API_SECRET", "OKX_API_PASSPHRASE", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID",
              "SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASSWORD", "ALERT_EMAIL_TO"):
        assert k in env
    assert env["OKX_API_KEY"] == "" and env["OKX_API_SECRET"] == "" and env["TELEGRAM_BOT_TOKEN"] == ""


def test_service_unit_follows_repo_conventions():
    text = (ROOT / "deploy" / "carry.service").read_text()
    for needle in ("User=ubuntu", "EnvironmentFile=/home/ubuntu/.secrets/carry.env", "Restart=on-failure",
                   "KillSignal=SIGTERM", "-m carry", "run"):
        assert needle in text


def test_runbook_mentions_the_hard_preconditions():
    text = (ROOT / "deploy" / "README-carry.md").read_text()
    for needle in ("preflight", "alert-test", "systemctl stop", "OKX_DEMO", "no BTC/ETH", "force-open"):
        assert needle in text
```

- [ ] **Step 2: Run to verify it fails**

Run: `$PY -m pytest tests/test_carry_deploy_files.py -q`
Expected: FAIL (`FileNotFoundError`)

- [ ] **Step 3: Create the files**

`config/carry.example.json`:

```json
{
  "coins": ["BTC", "ETH"],
  "deploy_usdt": 250.0,
  "max_order_notional": 300.0,
  "enter_apr": 0.06,
  "exit_apr": 0.02,
  "window": 9,
  "min_hold": 9,
  "poll_sec": 2.0,
  "settle_sec": 1.0,
  "state_path": "live_state/carry_state.json"
}
```

`deploy/carry.env.example`:

```
# Copy to /home/ubuntu/.secrets/carry.env and chmod 600. NEVER commit real values.
# OKX (use a key with Trade permission, NO Withdraw; bind it to this server's public IP)
OKX_API_KEY=
OKX_API_SECRET=
OKX_API_PASSPHRASE=
# Set OKX_DEMO=1 to use OKX demo trading (needs a demo-trading API key)
OKX_DEMO=
# Spot trade mode used for spot orders (verify on demo; default cash)
OKX_SPOT_TD_MODE=
# Telegram notifications
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
# Email notifications (SMTP over SSL on 465, or STARTTLS on 587)
SMTP_HOST=
SMTP_PORT=
SMTP_USER=
SMTP_PASSWORD=
ALERT_EMAIL_FROM=
ALERT_EMAIL_TO=
```

`deploy/carry.service`:

```
[Unit]
Description=Funding carry engine (OKX spot long + USDT-perp short, 1.0x)
After=network-online.target
Wants=network-online.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/crypto_backtest
EnvironmentFile=/home/ubuntu/.secrets/carry.env
Environment=PYTHONPATH=/home/ubuntu/crypto_backtest/src
ExecStart=/home/ubuntu/crypto_backtest/.venv/bin/python -m carry \
    --config /home/ubuntu/crypto_backtest/config/carry.json \
    --log /home/ubuntu/crypto_backtest/live_state/carry.log run
Restart=on-failure
RestartSec=30
KillSignal=SIGTERM
TimeoutStopSec=60

[Install]
WantedBy=multi-user.target
```

`deploy/README-carry.md`:

```markdown
# Funding carry — runbook

The engine keeps BTC and ETH hedged (spot long + USDT-perp short, 1.0x, market orders) when trailing funding is high, and flattens the spot leg immediately if the perp leg is ADL'd/liquidated. After any watchdog/risk close it is HALTED until a human runs `rearm`.

## One-time setup
1. **Account**: a dedicated OKX (sub-)account in multi-currency margin mode (acctLv 3), `autoLoan` OFF, net position mode, holding ONLY USDT — **no BTC/ETH** (any BTC/ETH makes the bot halt; move dust away). Stop any other strategy that uses the account: `mgnRatio` is account-wide.
2. **Keys**: create the API key with Trade permission and NO Withdraw, bound to this server's public IP. Put it in `/home/ubuntu/.secrets/carry.env` (template: `deploy/carry.env.example`), `chmod 600`. Add Telegram (`TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`) and SMTP variables.
3. **Config**: `cp config/carry.example.json config/carry.json` and edit (`deploy_usdt` about 50-60% of equity for the first run).

## Before the service
```
python -m carry --config config/carry.json --env-file ~/.secrets/carry.env alert-test   # log + Telegram + email arrive
python -m carry --config config/carry.json --env-file ~/.secrets/carry.env preflight     # must print "preflight OK"
python -m carry --config config/carry.json --env-file ~/.secrets/carry.env status        # exposure report
```

## Demo cycle (do this BEFORE live money; needs a demo-trading API key, set `OKX_DEMO=1`)
1. `preflight`, then `force-open BTC 100`: check `status` shows a hedge (`hedge=OK`), spot and perp quantities match.
2. Record `mgnRatio` from `status` with the position open. It sets the units and thresholds: if the value is a percentage (e.g. 350) set `mgn_unit` to 100 in the config; calibrate `mgn_warn` / `mgn_close` / `mgn_emergency` from the observed value and a price-shock estimate. Do not run live until this is done.
3. Check spot orders work with `OKX_SPOT_TD_MODE` (`cash` by default because the spot leg is unlevered; try `cross` only if `cash` orders are rejected) and that spot buys deduct the fee in kind (the hedge stays within `drift_tol`).
4. Simulate ADL by closing the perp leg by hand: within a few polls the bot must sell the spot, alert CRITICAL, and go HALTED. Then `force-close`, `rearm`.
5. Restart mid-position: the bot must adopt the healthy hedge or halt and alert (see `[CRITICAL] startup`).
6. Kill the network briefly: expect a halt with alerts, no unhedged exposure.

## API assumptions the demo run must confirm (from the adapter review; record each result in the demo findings doc)
1. `clOrdId` uniqueness: re-POST a filled order's clOrdId — rejected (which code) or accepted as a new order? Confirms `DUPLICATE_CLORDID_CODES`.
2. Rejection envelope: an insufficient-balance order — top-level `code` and row `sCode`? Confirms the adapter's row-level verdict.
3. `get_order` by `clOrdId` right after placement (visibility delay) and the shape/code for a non-existent order (`NOT_FOUND_CODES`, empty data?).
4. Spot market buy with `tgtCcy=base_ccy` under `cash` AND `cross`: fill size in coins; fee taken in the base coin (`cashBal` rises by `sz*(1-fee)`); `cross` must not borrow with autoLoan off.
5. Swap orders: `sz` in contracts, net mode with empty `posSide`, `reduceOnly="true"` on a market buy, behaviour when the size exceeds the position, `accFillSz` in contracts.
6. Cancel path: cancelling an already-filled order (code) and `cancel_order` by `clOrdId` alone.
7. Read fields: `details[].cashBal` is the total incl. coin frozen by live orders; `mgnRatio` scale (ratio vs percent -> `mgn_unit`); `pos` sign in net mode; `adl` range.
8. Funding history: does it include the unsettled period? `fundingRate` vs `realizedRate` — which was actually paid?
9. `TRANSIENT_CODES` (50001/50004/50011/50013/50026) and that rate limiting surfaces as 50011, not an HTTP error with a non-JSON body.
10. Account settings: swap leverage for the instruments, position mode net, `autoLoan` off; API clients are built with `debug=False` (never log signed headers).

## Live (small)
1. Stop other users of the account, run `preflight`, `alert-test`.
2. `force-open BTC 100` then `force-close BTC` once with real money as a smoke test (cost about 0.3%).
3. Install the unit: `sudo cp deploy/carry.service /etc/systemd/system/ && sudo systemctl daemon-reload && sudo systemctl enable --now carry`. Watch `journalctl -u carry -f`, the log file and Telegram; the heartbeat arrives hourly.

## Operations
- Manual commands need the lock: `sudo systemctl stop carry`, then `force-open|force-close|rearm|status`, then `sudo systemctl start carry`.
- HALTED: read the alert and `status`; fix the cause; flatten any leg with `force-close COIN`; when the exchange is flat run `rearm`.
- The bot never sells coins it did not buy; if it reports "unexpected holdings", move them out of the account.
```

- [ ] **Step 4: Run to verify it passes**

Run: `$PY -m pytest tests/test_carry_*.py -q`
Expected: all pass (145 + 3 (Task 1) + 9 + 17 + 12 + 11 + 7 + 20 + 4 = 228 tests).

- [ ] **Step 5: Commit**

```bash
git add config/carry.example.json deploy/carry.service deploy/carry.env.example deploy/README-carry.md tests/test_carry_deploy_files.py
git commit -m "feat(carry): deployment files and runbook

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Phase 5 (manual, done LAST, needs the user's keys)

Not implementation tasks — follow `deploy/README-carry.md`:
1. User creates: a Telegram bot + chat id, SMTP credentials, and finally an OKX demo-trading API key (last, as requested).
2. Demo cycle (open, `mgnRatio` calibration and `mgn_unit`, spot `tdMode`, simulated ADL, restart, network loss).
3. Update `config/carry.json` from the demo findings (thresholds, `mgn_unit`, `OKX_SPOT_TD_MODE`); record the findings in a short `docs/superpowers/specs/2026-XX-XX-funding-carry-demo-findings.md`.
4. Live smoke test (`force-open` + `force-close`, 100 USDT), then the service with `deploy_usdt` about 250.
