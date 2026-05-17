# Live History Query Reference

历史数据库路径：`data/live_history.db`

## 近期状态快照

```bash
sqlite3 data/live_history.db \
  "SELECT account_id, exchange, strategy, symbol, datetime(ts/1000,'unixepoch') FROM state_snapshots ORDER BY ts DESC LIMIT 20"
```

## 账户成交历史

```bash
sqlite3 data/live_history.db \
  "SELECT datetime(ts/1000,'unixepoch'), side, price, notional, realized_pnl FROM trades WHERE account_id='<account_id>' ORDER BY ts"
```

## 仓位快照时间线

```bash
sqlite3 data/live_history.db \
  "SELECT datetime(ts/1000,'unixepoch'), side, quantity, entry_price, unrealized_pnl, equity FROM positions WHERE symbol='BTCUSDT' ORDER BY ts"
```
