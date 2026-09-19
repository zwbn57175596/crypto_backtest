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
                   "KillSignal=SIGTERM", "-m carry", "run", "TimeoutStopSec=300",
                   "RestartPreventExitStatus=2 3", "Environment=PYTHONUNBUFFERED=1"):
        assert needle in text


def test_runbook_mentions_the_hard_preconditions():
    text = (ROOT / "deploy" / "README-carry.md").read_text()
    for needle in ("preflight", "alert-test", "systemctl stop", "OKX_DEMO", "no BTC/ETH", "force-open"):
        assert needle in text


def test_runbook_covers_the_batch_review_operational_notes():
    text = (ROOT / "deploy" / "README-carry.md").read_text()
    for needle in ("dead-man", "timeout", "force-close", "corrupt", "local filesystem", "clOrdId", "cash"):
        assert needle in text


def test_env_example_has_no_values_at_all():
    env = load_env_file(ROOT / "deploy" / "carry.env.example")
    assert all(v == "" for v in env.values())


def test_example_config_has_absolute_state_path_and_explicit_margin_thresholds():
    c = load_config(ROOT / "config" / "carry.example.json")
    assert c.state_path == "/home/ubuntu/crypto_backtest/live_state/carry_state.json"
    assert (c.mgn_unit, c.mgn_warn, c.mgn_close, c.mgn_emergency) == (1.0, 4.0, 2.5, 1.5)


def test_env_example_has_no_export_lines():
    assert not any(l.strip().startswith("export ") for l in (ROOT / "deploy" / "carry.env.example").read_text().splitlines())


def test_runbook_covers_fix_round_1_operator_notes():
    text = (ROOT / "deploy" / "README-carry.md").read_text()
    for needle in ("okx mode", "DEMO", "LIVE", "cd /home/ubuntu/crypto_backtest", "availEq", "EnvironmentFile",
                   "1.2", "leverage", "OKX UI", "delete/rename", "coins", "calibrated"):
        assert needle in text


def test_gitignore_covers_carry_runtime_files_but_not_the_whole_directory():
    lines = (ROOT / ".gitignore").read_text().splitlines()
    for pat in ("live_state/carry_state.json*", "live_state/carry.log*", "live_state/*.lock", "live_state/*.corrupt-*"):
        assert pat in lines
    assert "live_state/" not in lines and "live_state" not in lines
