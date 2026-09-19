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
