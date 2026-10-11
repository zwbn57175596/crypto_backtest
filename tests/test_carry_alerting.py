import logging
import ssl
import threading

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


def sync(fn):
    fn()


def make(sinks, clock=None, throttle=300.0):
    return Alerter(logging.getLogger("carry.test"), sinks, throttle_sec=throttle,
                   clock=clock or Clock(), dispatch=sync)


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
    assert "secret-token-123" not in caplog.text


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
    assert payload["chat_id"] == "42" and len(payload["text"]) == 4000 and timeout == 5.0
    assert calls[1] == "checked"


class FakeSMTP:
    instances = []

    def __init__(self, host, port, timeout, context=None):
        self.host, self.port, self.log = host, port, []
        self.init_context, self.tls_context = context, None
        FakeSMTP.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def starttls(self, context=None):
        self.log.append("starttls")
        self.tls_context = context

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
    lg.propagate = True


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


# --- fix round 1 ------------------------------------------------------------

def test_failed_delivery_is_not_throttled_and_is_retried():
    class Flaky(Sink):
        def send(self, text):
            if not self.sent and not getattr(self, "tried", False):
                self.tried = True
                raise RuntimeError("boom")
            self.sent.append(text)

    s, clock = Flaky("flaky"), Clock()
    a = make([s], clock)
    a("m")
    assert s.sent == []
    clock.t += 1
    a("m")                                   # identical, within throttle window, but the first never delivered
    assert s.sent == ["m"]
    clock.t += 1
    a("m")                                   # now delivered and throttled
    assert s.sent == ["m"]


def test_throttle_is_per_sink():
    bad_then_ok, good, clock = Sink("a", fail=True), Sink("b"), Clock()
    a = make([bad_then_ok, good], clock)
    a("m")
    bad_then_ok.fail = False
    clock.t += 1
    a("m")
    assert good.sent == ["m"]                # b throttled on the repeat
    assert bad_then_ok.sent == ["m"]         # a retried because its first attempt failed


def test_non_str_message_and_broken_clock_never_raise():
    s = Sink()

    def bad_clock():
        raise RuntimeError("clock")

    Alerter(logging.getLogger("carry.test"), [s], clock=bad_clock, dispatch=sync)(123)
    assert s.sent == ["123"]


def test_default_dispatcher_delivers_in_background():
    s = Sink()
    a = Alerter(logging.getLogger("carry.test"), [s], clock=Clock())
    a("hi")
    assert a.flush(5.0) is True
    assert s.sent == ["hi"]


def test_flush_true_immediately_with_injected_dispatcher():
    assert make([Sink()]).flush(0.0) is True


class BlockingSink:
    name = "slow"

    def __init__(self):
        self.started, self.release, self.sent = threading.Event(), threading.Event(), []

    def send(self, text):
        self.started.set()
        assert self.release.wait(5.0)
        self.sent.append(text)


def test_slow_sink_does_not_block_the_caller():
    s = BlockingSink()
    a = Alerter(logging.getLogger("carry.test"), [s], clock=Clock())
    a("x")                                   # returns although the sink is blocked
    assert s.started.wait(5.0)
    assert s.sent == []
    s.release.set()
    assert a.flush(5.0) is True
    assert s.sent == ["x"]


def test_flush_times_out_while_sink_is_blocked():
    s = BlockingSink()
    a = Alerter(logging.getLogger("carry.test"), [s], clock=Clock())
    a("x")
    assert s.started.wait(5.0)
    assert a.flush(0.05) is False
    s.release.set()
    assert a.flush(5.0) is True


def test_full_queue_drops_with_log_line_and_never_blocks(caplog):
    s = BlockingSink()
    a = Alerter(logging.getLogger("carry.test"), [s], clock=Clock())
    with caplog.at_level(logging.INFO, logger="carry.test"):
        for i in range(150):
            a(f"m{i}")
    assert "dropped" in caplog.text
    s.release.set()
    assert a.flush(5.0) is True


def test_email_sink_passes_verifying_tls_context_on_starttls_and_ssl():
    FakeSMTP.instances.clear()
    EmailSink("h", 587, "u", "p", "u@x", ["a@x"], smtp_factory=FakeSMTP).send("m")
    ctx = FakeSMTP.instances[0].tls_context
    assert isinstance(ctx, ssl.SSLContext) and ctx.verify_mode == ssl.CERT_REQUIRED
    assert FakeSMTP.instances[0].init_context is None
    FakeSMTP.instances.clear()
    EmailSink("h", 465, "u", "p", "u@x", ["a@x"], smtp_factory=FakeSMTP).send("m")
    ctx = FakeSMTP.instances[0].init_context
    assert isinstance(ctx, ssl.SSLContext) and ctx.verify_mode == ssl.CERT_REQUIRED


def test_email_sink_empty_message_does_not_raise():
    FakeSMTP.instances.clear()
    EmailSink("h", 465, "u", "p", "u@x", ["a@x"], smtp_factory=FakeSMTP).send("")
    assert FakeSMTP.instances[0].log[-1] == ("send", "[carry] ", "a@x")


def test_email_sink_defaults_to_only_critical():
    assert EmailSink("h", 465, "u", "p", "u@x", ["a@x"]).only_critical is True


def test_alerter_routes_routine_messages_away_from_only_critical_sinks():
    email, telegram, clock = Sink("email"), Sink("telegram"), Clock()
    email.only_critical = True
    a = make([email, telegram], clock)
    for msg in ("[HEARTBEAT]\nall good", "[INFO] opened BTC", "[WARNING] mgn_unit not calibrated", "risk tier NORMAL -> NORMAL"):
        a(msg)
        clock.t += 400  # clear throttling between messages
    assert email.sent == []
    assert len(telegram.sent) == 4


def test_alerter_still_sends_critical_messages_to_only_critical_sinks():
    email = Sink("email")
    email.only_critical = True
    make([email])("[CRITICAL] hedge broken")
    assert email.sent == ["[CRITICAL] hedge broken"]


def test_build_alerter_email_sink_is_only_critical(tmp_path, clean_carry_logger):
    log = tmp_path / "carry.log"
    mail = {"SMTP_HOST": "h", "SMTP_USER": "u", "SMTP_PASSWORD": "p", "ALERT_EMAIL_TO": "a@x.com"}
    alerter = build_alerter(mail, str(log))
    assert alerter.sinks[0].only_critical is True


def test_setup_logger_isolates_and_quiets_http_loggers(tmp_path, clean_carry_logger):
    setup_logger(tmp_path / "carry.log")
    assert logging.getLogger("httpx").level == logging.WARNING
    assert logging.getLogger("httpcore").level == logging.WARNING
    assert logging.getLogger("carry").propagate is False


def test_setup_logger_dedupes_equivalent_paths(tmp_path, clean_carry_logger):
    (tmp_path / "sub").mkdir()
    lg = setup_logger(tmp_path / "sub" / "carry.log")
    n = len(lg.handlers)
    assert len(setup_logger(tmp_path / "sub" / ".." / "sub" / "carry.log").handlers) == n


def _logtext(path):
    for h in logging.getLogger("carry").handlers:
        h.flush()
    return path.read_text()


def test_build_alerter_warns_about_partial_telegram_by_name_only(tmp_path, clean_carry_logger):
    log = tmp_path / "carry.log"
    assert build_alerter({"TELEGRAM_BOT_TOKEN": "SECRETTOK"}, str(log)).channels == []
    text = _logtext(log)
    assert "TELEGRAM_CHAT_ID" in text and "SECRETTOK" not in text


def test_build_alerter_warns_about_partial_smtp(tmp_path, clean_carry_logger):
    log = tmp_path / "carry.log"
    assert build_alerter({"SMTP_HOST": "hostval"}, str(log)).channels == []
    text = _logtext(log)
    assert all(n in text for n in ("SMTP_USER", "SMTP_PASSWORD", "ALERT_EMAIL_TO"))
    assert "hostval" not in text


def test_build_alerter_skips_email_without_recipients(tmp_path, clean_carry_logger):
    log = tmp_path / "carry.log"
    env = {"SMTP_HOST": "h", "SMTP_USER": "u", "SMTP_PASSWORD": "p", "ALERT_EMAIL_TO": " , "}
    assert build_alerter(env, str(log)).channels == []
    assert "ALERT_EMAIL_TO" in _logtext(log)


def test_build_alerter_bad_smtp_port_skips_only_email(tmp_path, clean_carry_logger):
    log = tmp_path / "carry.log"
    env = {"TELEGRAM_BOT_TOKEN": "t", "TELEGRAM_CHAT_ID": "1", "SMTP_HOST": "h", "SMTP_USER": "u",
           "SMTP_PASSWORD": "p", "ALERT_EMAIL_TO": "a@x.com", "SMTP_PORT": "abc"}
    assert build_alerter(env, str(log)).channels == ["telegram"]
    assert "SMTP_PORT" in _logtext(log)


def test_build_alerter_passes_dispatch_through(tmp_path, clean_carry_logger):
    calls = []

    class Resp:
        def raise_for_status(self):
            pass

    a = build_alerter({"TELEGRAM_BOT_TOKEN": "t", "TELEGRAM_CHAT_ID": "1"}, str(tmp_path / "c.log"),
                      post=lambda url, json, timeout: (calls.append(json), Resp())[1], dispatch=sync)
    a("hello")
    assert calls and calls[0]["text"] == "hello"
