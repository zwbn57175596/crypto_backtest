# Funding carry — runbook

The engine keeps BTC and ETH hedged (spot long + USDT-perp short, 1.0x, market orders) when trailing funding is high, and flattens the spot leg immediately if the perp leg is ADL'd/liquidated. After any watchdog/risk close it is HALTED until a human runs `rearm`.

## One-time setup
1. **Account**: a dedicated OKX (sub-)account in multi-currency margin mode (acctLv 3), `autoLoan` OFF, net position mode, holding ONLY USDT — **no BTC/ETH** (any BTC/ETH makes the bot halt; move dust away). Stop any other strategy that uses the account: `mgnRatio` is account-wide.
2. **Keys**: create the API key with Trade permission and NO Withdraw, bound to this server's public IP. Put it in `/home/ubuntu/.secrets/carry.env` (template: `deploy/carry.env.example`), `chmod 600`. Add Telegram (`TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`) and SMTP variables.
3. **Config**: `cp config/carry.example.json config/carry.json` and edit (`deploy_usdt` about 50-60% of equity for the first run).
4. **State and lock location**: `state_path` (and its `.lock` file, `<state_path>.lock`) must be on a local filesystem (ext4 on the server's own disk). `flock` is not reliable on NFS/SMB/synced folders, and a broken lock allows two instances to trade the same account.

## Before the service
```
python -m carry --config config/carry.json --env-file ~/.secrets/carry.env alert-test   # log + Telegram + email arrive
python -m carry --config config/carry.json --env-file ~/.secrets/carry.env preflight     # must print "preflight OK"
python -m carry --config config/carry.json --env-file ~/.secrets/carry.env status        # exposure report
```
`preflight` also fails (`PROBLEM:` line) for every carry order (`clOrdId` starting `carry`) still live on the exchange: resolve or cancel them by hand first.

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
- HALTED: read the alert and `status`; fix the cause; flatten any leg with `force-close COIN`; when the exchange is flat run `rearm`. `rearm` also clears the adapter's "unconfirmed order" block, so only run it after you have checked by hand that no carry order is still live.
- The bot never sells coins it did not buy; if it reports "unexpected holdings", move them out of the account.
- **Pending orders after a restart**: on start (and before every manual command) the bot registers carry orders that are still live on the exchange as "unconfirmed" and sends a CRITICAL alert; new orders on those instruments stay blocked until the orders resolve (or you `rearm` after verifying them by hand).

## Failure modes to know about
- **HALTED start with a healthy pre-existing hedge is NOT adopted.** If the bot starts HALTED (unreadable state file, or the startup check could not complete — alert `[CRITICAL] startup check failed (...); starting HALTED`), it does not adopt an existing hedge. Close it by hand with `force-close COIN` (the hedge is otherwise unmanaged by the risk logic that only acts on recorded legs), then `rearm` once flat.
- **Startup check failure**: a transient exchange error while reconciling starts the bot HALTED and persists that; it does not exit and it does not trade. The watchdog and heartbeat keep running. Fix the cause, then `force-close`/`rearm`.
- **Hung exchange call**: the watchdog, the risk checks and the heartbeat run in one loop, so one hung HTTP call stalls all of them together. python-okx clients are `httpx` clients with httpx's default 5 s timeout (the CLI wiring does not change it; set `timeout` on the clients if you ever tune it). Do not rely on the in-process heartbeat alone: run an external dead-man's switch that alerts when the heartbeat message is ABSENT for more than about 2 hours (for example a Healthchecks-style ping or a monitor on the Telegram chat/`carry.log`).
- **Corrupt state file**: an unreadable state file is renamed to `<state>.corrupt-<unix time>` and replaced by a HALTED state. If the rename fails (alert says "could not be moved aside"), the first save overwrites the bad file: copy it somewhere first (`cp live_state/carry_state.json ~/carry_state.bak`) if you want to inspect it.
- **Lock**: the single-instance lock is `<state_path>.lock`; a second instance or a manual command while the service runs exits with code 3 and "another carry instance holds the lock". Keep it on a local filesystem (see setup).
