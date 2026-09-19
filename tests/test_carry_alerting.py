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
