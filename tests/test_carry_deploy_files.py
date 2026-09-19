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


def test_runbook_covers_the_batch_review_operational_notes():
    text = (ROOT / "deploy" / "README-carry.md").read_text()
    for needle in ("dead-man", "timeout", "force-close", "corrupt", "local filesystem", "clOrdId", "cash"):
        assert needle in text


def test_env_example_has_no_values_at_all():
    env = load_env_file(ROOT / "deploy" / "carry.env.example")
    assert all(v == "" for v in env.values())
