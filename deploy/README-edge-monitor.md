# Edge Health Monitor — deployment (LXC 203 / appserver)

A **measurement-only** weekly job. It fetches fresh OKX 1h klines, measures the
volatility-filtered, cost-adjusted rolling edge of the mean-reversion portfolio
strategy, and emits **GO / HOLD**. It **never trades, never holds positions, and
needs no API keys** (klines are public). Purpose: the strategy's edge was real
through 2021–2024 but has eroded to ~zero/negative in 2025–2026; this watches for
it to revive before any real capital is ever considered.

## What it decides

- Metric: `edge_bps` = net-of-fee profit per dollar of notional traded (compounding-invariant).
- Trades are gated to **high-volatility regime** (BTC 7d realized vol above its ~60d EMA).
- **GO** when the trailing-90d edge ≥ `--go-threshold-bps` (default **+2.0 bps**) **and** the
  60d edge > 0. Otherwise **HOLD**. Default slippage assumption: **5 bps/side** (realistic for alts).
- Output: `data/edge_monitor_status.json` (machine-readable) + a stdout summary; process exit
  code `0`=HOLD, `10`=GO, `2`=error.

## One-time setup on LXC 203

```bash
# as root (or a sudoer)
apt-get update && apt-get install -y python3-venv git

# clone to /opt (matches the unit files; change both if you use another path)
git clone <your-remote-or-copy> /opt/crypto_backtest
cd /opt/crypto_backtest
git checkout <the-branch-with-this-monitor>

python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# smoke test (skips network fetch, uses whatever data is present):
python scripts/edge_monitor.py --no-collect || true
# full run (fetches fresh klines first):
scripts/run_edge_monitor.sh
```

Create/adjust the service user (the unit files assume a user named `appserver`
and `WorkingDirectory=/opt/crypto_backtest`):

```bash
# if the user doesn't exist:
useradd -r -s /usr/sbin/nologin appserver 2>/dev/null || true
chown -R appserver:appserver /opt/crypto_backtest
```

## Install the weekly timer

```bash
cp deploy/edge-monitor.service /etc/systemd/system/
cp deploy/edge-monitor.timer   /etc/systemd/system/
# edit the two marked lines in edge-monitor.service if your user/path differ:
#   User=appserver
#   WorkingDirectory=/opt/crypto_backtest
systemctl daemon-reload
systemctl enable --now edge-monitor.timer
```

## Operate

```bash
systemctl list-timers edge-monitor.timer      # next run
systemctl start edge-monitor.service          # run once now
journalctl -u edge-monitor.service -n 50      # last run's output
cat /opt/crypto_backtest/data/edge_monitor_status.json   # latest signal + windows
```

## Getting alerted when the edge revives (GO)

The runner calls `$EDGE_ALERT_CMD` on a GO. Set it in the service unit, e.g.:

```ini
# in /etc/systemd/system/edge-monitor.service, under [Service]
Environment=EDGE_ALERT_CMD=curl -fsS -X POST https://your-webhook -d "MR edge revived (GO)"
```

(Telegram/Slack/email — any one-shot command works.) After editing:
`systemctl daemon-reload`.

## Tuning (optional)

Pass extra args via `EDGE_MONITOR_ARGS` in the unit, or on the CLI:

```bash
# stricter cost assumption, tighter GO bar, majors-inclusive universe:
scripts/run_edge_monitor.sh --slippage-bps 4 --go-threshold-bps 3 \
  --symbols ETHUSDT,DOGEUSDT,WLDUSDT,PEPEUSDT,AAVEUSDT,UNIUSDT,SOLUSDT,XRPUSDT
```

## Important

A **GO is not an instruction to trade** — it means the measured edge has come back
and it is worth a human re-evaluation (and, separately, building the not-yet-existent
multi-symbol live execution path). This job only measures.
