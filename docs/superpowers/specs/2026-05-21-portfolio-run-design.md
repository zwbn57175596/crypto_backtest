# Portfolio Run — 设计规范

**日期**: 2026-05-21  
**状态**: 已确认，待实现

---

## 背景

现有框架只支持单标的回测（`backtest run`）。实盘策略 BET（`bet_live.py`）在多个 USDT-M 永续合约上同时运行 consecutive_reverse 策略，共享一个账户余额池。本规范描述如何在回测框架中复现这一模式。

---

## 方案选择

**A — 新 CLI 子命令 `portfolio-run`**（已选）

独立的 `PortfolioBacktestEngine`，不改现有 `BacktestEngine`、`DataFeed`、`BaseStrategy`。复用 `SimExchange`（已支持多标的持仓）和 `Reporter`（计算合并指标）。

放弃原因：
- B（BasePortfolioStrategy）：改动面覆盖引擎、数据流、策略接口，风险高
- C（独立脚本）：已有 `strategies/portfolio_backtest.py` 验证了该路线的局限（数据源写死、不入库）

未来加速路线：`PortfolioBacktestEngine` → `numba_portfolio_simulate.py` → `cuda_portfolio_runner.py`，和现有加速层模式一致。

---

## 架构

### 新增文件

```
src/backtest/
├── portfolio_engine.py      # PortfolioBacktestEngine
└── multi_data_feed.py       # MultiSymbolDataFeed
```

`src/backtest/__main__.py` 新增 `portfolio-run` 子命令（唯一改动的现有文件）。

### 复用现有组件（零改动）

| 组件 | 复用方式 |
|---|---|
| `SimExchange` | 持仓已是 `Dict[str, Position]`，天然多标的 |
| `Reporter` | 计算合并的 14 个指标，存入 `reports.db` |
| `models.py` | Bar、Order、Position、Trade 不变 |
| `DataFeed` | 不改，`MultiSymbolDataFeed` 独立实现 |

### 数据流

```
SQLite klines（多标的）
    ↓
MultiSymbolDataFeed
  → 时间戳并集对齐，每步 yield (ts, {sym: Bar})
    ↓
PortfolioBacktestEngine.run()
  → 每个时间步：
      1. SimExchange.on_new_bar(bar) × N 标的
      2. 每标的运行 consecutive_reverse 决策逻辑
      3. 账户风控检查（新品种开仓前）
      4. 向 SimExchange 提交订单
    ↓
PortfolioReporter
  → 合并 14 指标 → reports.db（一条记录）
  → 各标的交易统计 → 终端打印
```

---

## 策略逻辑

### 每标的独立状态

```python
states[sym] = {
    "consecutive_count": 0,
    "streak_direction": 0,   # +1 / -1 / 0
    "profit_candle_count": 0,
}
```

### 每根 K 线决策（与 consecutive_reverse.py 一致）

```
无仓位：
  consecutive >= threshold → 反向开仓

有仓位 + 盈利K线：
  profit_count += 1
  profit_count >= profit_candle_threshold → 平仓，重新开仓

有仓位 + 亏损K线：
  立即平仓，重新开仓
```

### 仓位计算（对齐 BET 实盘）

```
size = wallet_balance × initial_position_pct × position_multiplier^(n-1)
n = consecutive_count - threshold + 1
```

`wallet_balance` = 总余额（含占用保证金），不用 available，避免多仓占用导致档位缩水。

---

## 账户风控

在**新品种开仓**前检查（已有仓位的平仓/调仓不受限制）：

| 检查项 | 条件 | 动作 |
|---|---|---|
| 品种数上限 | 当前持仓品种数 >= `MAX_POSITION_SYMBOLS` | 跳过该标的开仓 |
| 保证金占用率 | `sum(margins) / equity > MARGIN_RATIO_LIMIT` | 跳过该标的开仓 |

---

## 多标的数据对齐（MultiSymbolDataFeed）

- 取所有标的时间戳的**并集**（不用交集，避免因晚上市标的截断早期数据）
- 每步只推送该时间戳有数据的标的
- 某标的某时间步无数据 → 跳过，状态保持不变，下一根有数据时继续

```python
all_timestamps = sorted(union of timestamps across all symbols)

for ts in all_timestamps:
    bars = {sym: Bar(...) for sym in symbols if ts in sym_data[sym]}
    yield ts, bars
```

---

## CLI 接口

```bash
python -m backtest portfolio-run \
    --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT \
    --interval 1h \
    --start 2024-01-01 --end 2024-12-31 \
    --balance 10000 \
    --leverage 10 \
    --consecutive-threshold 7 \
    --position-multiplier 2.9 \
    --initial-position-pct 1.19 \
    --profit-candle-threshold 3 \
    --max-position-symbols 10 \
    --margin-ratio-limit 0.8
```

### 标的列表（静态，预留动态扩展口）

通过 `--symbols` 传入逗号分隔列表。`MultiSymbolDataFeed` 预留 `universe_fn` 钩子供后续动态 Top N 扩展。

---

## 输出格式

```
[组合回测] 标的=4 区间=2024-01-01~2024-12-31 余额=10000 杠杆=10x

=== 组合指标 ===
net_return        +38.4%
annual_return     +38.4%
max_drawdown      -22.1%
sharpe_ratio      1.43
...（14 项）

=== 各标的统计 ===
BTCUSDT  交易=124  胜率=52.4%  净PnL=+1823 USDT
ETHUSDT  交易=138  胜率=51.1%  净PnL=+1204 USDT
...

报告已存入 reports.db (id=42)
```

reports.db 存**一条记录**，`report_json` 包含合并指标 + 各标的明细（嵌套）。

---

## 不在本次范围内

- `portfolio-optimize`（参数网格搜索）— 后续独立实现
- 动态标的池（24h 成交额 Top N）— 预留接口，后续实现
- Web 界面各标的独立页面 — 现有 web 不改动
- Numba / CUDA 加速 — 后续基于本引擎扩展
