"""Alerting: a durable log plus throttled Telegram/email notifications. Never raises."""
from __future__ import annotations

import logging
import queue
import smtplib
import ssl
import threading
import time
from email.message import EmailMessage
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Callable, Mapping, Protocol, Sequence

LOGGER_NAME = "carry"


def setup_logger(path: str | Path, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False
    for noisy in ("httpx", "httpcore"):      # their INFO lines contain the full request URL (Telegram token)
        logging.getLogger(noisy).setLevel(logging.WARNING)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    if not any(isinstance(h, RotatingFileHandler) and Path(h.baseFilename).resolve() == p.resolve()
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

    def __init__(self, token: str, chat_id: str, post: Callable | None = None, timeout: float = 5.0) -> None:
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
                 recipients: Sequence[str], smtp_factory: Callable | None = None, timeout: float = 10.0) -> None:
        self.host, self.port, self.user, self.password = host, port, user, password
        self.sender, self.recipients, self.timeout, self._factory = sender, list(recipients), timeout, smtp_factory

    def send(self, text: str) -> None:
        msg = EmailMessage()
        msg["Subject"] = "[carry] " + (text.splitlines() or [""])[0][:100]
        msg["From"] = self.sender
        msg["To"] = ", ".join(self.recipients)
        msg.set_content(text)
        ctx = ssl.create_default_context()
        if self.port == 465:
            conn = (self._factory or smtplib.SMTP_SSL)(self.host, self.port, timeout=self.timeout, context=ctx)
        else:
            conn = (self._factory or smtplib.SMTP)(self.host, self.port, timeout=self.timeout)
        with conn as s:
            if self.port != 465:
                s.starttls(context=ctx)
            s.login(self.user, self.password)
            s.send_message(msg)


class Alerter:
    QUEUE_MAX = 100

    def __init__(self, log: logging.Logger, sinks: Sequence[Sink] = (), throttle_sec: float = 300.0,
                 clock: Callable[[], float] = time.monotonic,
                 dispatch: Callable[[Callable[[], None]], None] | None = None) -> None:
        self.log, self.sinks, self.throttle_sec, self.clock = log, list(sinks), throttle_sec, clock
        self._last: dict[tuple[str, str], float] = {}
        self._lock = threading.Lock()
        self._dispatch = dispatch or self._enqueue
        self._queue: queue.Queue[Callable[[], None]] = queue.Queue(maxsize=self.QUEUE_MAX)
        self._cond = threading.Condition()
        self._pending = 0
        self._worker: threading.Thread | None = None

    @property
    def channels(self) -> list[str]:
        return [s.name for s in self.sinks]

    def __call__(self, msg: str) -> None:
        try:
            msg = str(msg)
            (self.log.critical if msg.startswith("[CRITICAL]") else self.log.info)(msg)
        except Exception:  # noqa: BLE001
            pass
        if not self.sinks:
            return
        try:
            try:
                now = self.clock()
            except Exception:  # noqa: BLE001
                now = time.monotonic()
            self._dispatch(lambda: self._deliver(msg, now))
        except Exception:  # noqa: BLE001
            pass

    def flush(self, timeout: float = 20.0) -> bool:
        """Wait until queued deliveries are done; True if drained, False on timeout."""
        with self._cond:
            return self._cond.wait_for(lambda: self._pending == 0, timeout)

    # --- delivery --------------------------------------------------------
    def _deliver(self, msg: str, now: float) -> None:
        with self._lock:
            self._last = {k: t for k, t in self._last.items() if now - t < self.throttle_sec}
        for s in self.sinks:
            key = (s.name, msg)
            with self._lock:
                if key in self._last:
                    continue
            try:
                s.send(msg)
            except Exception as e:  # noqa: BLE001 - never log str(e): it can contain a token/URL
                try:
                    self.log.error("alert sink %s failed: %s", s.name, type(e).__name__)
                except Exception:  # noqa: BLE001
                    pass
            else:
                with self._lock:
                    self._last[key] = now

    def _enqueue(self, fn: Callable[[], None]) -> None:
        with self._cond:
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(target=self._run, name="carry-alerts", daemon=True)
                self._worker.start()
            try:
                self._queue.put_nowait(fn)
            except queue.Full:
                try:
                    self.log.error("alert queue full; notification dropped")
                except Exception:  # noqa: BLE001
                    pass
                return
            self._pending += 1

    def _run(self) -> None:
        while True:
            fn = self._queue.get()
            try:
                fn()
            except Exception:  # noqa: BLE001
                pass
            finally:
                with self._cond:
                    self._pending -= 1
                    self._cond.notify_all()


def build_alerter(env: Mapping[str, str], log_path: str, *, post: Callable | None = None,
                  smtp_factory: Callable | None = None, clock: Callable[[], float] = time.monotonic,
                  throttle_sec: float = 300.0,
                  dispatch: Callable[[Callable[[], None]], None] | None = None) -> Alerter:
    logger = setup_logger(log_path)
    sinks: list[Sink] = []

    def missing(names: Sequence[str]) -> list[str]:
        return [n for n in names if not env.get(n)]

    tg_keys = ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID")
    if len(missing(tg_keys)) == 0:
        sinks.append(TelegramSink(env["TELEGRAM_BOT_TOKEN"], env["TELEGRAM_CHAT_ID"], post=post))
    elif len(missing(tg_keys)) < len(tg_keys):
        logger.warning("telegram not configured: missing %s", ", ".join(missing(tg_keys)))

    mail_keys = ("SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD", "ALERT_EMAIL_TO")
    if len(missing(mail_keys)) == 0:
        recipients = [a.strip() for a in env["ALERT_EMAIL_TO"].split(",") if a.strip()]
        try:
            port = int(env.get("SMTP_PORT") or 465)
        except ValueError:
            port = None
            logger.warning("email disabled: SMTP_PORT is not a number")
        if not recipients:
            logger.warning("email disabled: ALERT_EMAIL_TO contains no recipients")
        elif port is not None:
            sinks.append(EmailSink(env["SMTP_HOST"], port, env["SMTP_USER"], env["SMTP_PASSWORD"],
                                   env.get("ALERT_EMAIL_FROM") or env["SMTP_USER"], recipients,
                                   smtp_factory=smtp_factory))
    elif len(missing(mail_keys)) < len(mail_keys):
        logger.warning("email not configured: missing %s", ", ".join(missing(mail_keys)))

    if not sinks:
        logger.warning("no notification channel configured (log only)")
    return Alerter(logger, sinks, throttle_sec, clock, dispatch=dispatch)
