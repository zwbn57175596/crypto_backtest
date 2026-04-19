# Crypto Backtest — 系统设计文档

U本位永续合约历史数据回测系统。基于事件驱动架构，支持从币安/OKX/HTX拉取K线数据，运行自定义策略回测，通过Web页面展示回测报告。

## 架构概览

事件驱动回测引擎，核心组件：

```
CLI / Web API
     │
     ▼
BacktestEngine ─── 事件循环协调器
  ├── DataFeed       从SQLite按时间逐条推送K线
  ├── BaseStrategy   策略基类，用户继承实现 on_bar()
  ├── SimExchange    模拟交易所：撮合、仓位、保证金、手续费、资金费率、强平
  └── Reporter       收集交易记录，计算指标，生成报告

DataCollector (独立模块)
  └── 币安/OKX/HTX REST API → 拉取历史K线 → SQLite
```

## 项目目录结构

```
crypto_backtest/
├── pyproject.toml
├── config/
│   └── default.yaml          # 默认回测配置
├── src/
│   └── backtest/
│       ├── __init__.py
│       ├── __main__.py        # CLI入口
│       ├── engine.py          # BacktestEngine 事件循环
│       ├── data_feed.py       # DataFeed 数据推送
│       ├── exchange.py        # SimExchange 模拟交易所
│       ├── strategy.py        # BaseStrategy 基类
│       ├── reporter.py        # 指标计算 + 报告生成
│       ├── models.py          # 数据模型 (Bar, Order, Position, Trade)
│       ├── collector/
│       │   ├── __init__.py
│       │   ├── base.py        # BaseCollector
│       │   ├── binance.py     # 币安K线采集
│       │   ├── okx.py         # OKX K线采集
│       │   └── htx.py         # HTX K线采集
│       └── web/
│           ├── __init__.py
│           ├── app.py          # FastAPI应用
│           ├── routes.py       # API路由
│           └── static/
│               └── index.html  # 单页报告页面 (ECharts)
├── strategies/
│   └── example_ma_cross.py    # 示例: 均线交叉策略
├── data/
│   └── klines.db              # SQLite数据库
└── tests/
```

## 数据模型

四个核心 dataclass:

### Bar (K线)

| 字段 | 类型 | 说明 |
|------|------|------|
| symbol | str | 交易对，如 BTCUSDT |
| timestamp | int | 毫秒时间戳 |
| open | float | 开盘价 |
| high | float | 最高价 |
| low | float | 最低价 |
| close | float | 收盘价 |
| volume | float | 成交量 |
| interval | str | K线周期: 1m/5m/15m/1h/4h/1d |

### Order (订单)

| 字段 | 类型 | 说明 |
|------|------|------|
| id | str | 订单ID |
| symbol | str | 交易对 |
| side | str | "buy" 或 "sell" |
| type | str | "market" 或 "limit" |
| quantity | float | 下单量 (USDT价值) |
| price | float \| None | limit单价格，market为None |
| status | str | "pending" → "filled" / "canceled" |
| filled_price | float | 成交价 |
| filled_at | int | 成交时间戳 |
| commission | float | 手续费 |

### Position (仓位)

| 字段 | 类型 | 说明 |
|------|------|------|
| symbol | str | 交易对 |
| side | str | "long" 或 "short" |
| quantity | float | 持仓量 |
| entry_price | float | 持仓均价 |
| leverage | int | 杠杆倍数 |
| unrealized_pnl | float | 未实现盈亏 |
| margin | float | 占用保证金 |

### Trade (成交记录)

| 字段 | 类型 | 说明 |
|------|------|------|
| id | str | 成交ID |
| order_id | str | 关联订单ID |
| symbol | str | 交易对 |
| side | str | "buy" 或 "sell" |
| price | float | 成交价 |
| quantity | float | 成交量 |
| pnl | float | 平仓盈亏 (开仓时为0) |
| commission | float | 手续费 |
| timestamp | int | 成交时间戳 |

## SimExchange 撮合逻辑

每根K线的处理顺序：

1. **资金费率结算** — 每8小时 (00:00, 08:00, 16:00 UTC)，`funding_payment = position.quantity × funding_rate`，多头支付/收取，空头反向
2. **撮合挂单** — 遍历 pending_orders：
   - market 单：以 bar.open 成交
   - limit buy：bar.low <= order.price 时成交
   - limit sell：bar.high >= order.price 时成交
3. **更新未实现盈亏** — long: `(close - entry) × qty`，short: `(entry - close) × qty`
4. **强平检测** — 当 `margin / (margin + unrealized_pnl)` 触及维持保证金率时，强制平仓
5. **推送bar给策略** — 调用 `strategy.on_bar(bar)`

## 策略基类 API

```python
class BaseStrategy:
    # 生命周期回调 (用户实现)
    def on_init(self): ...
    def on_bar(self, bar: Bar): ...

    # 交易操作
    def buy(self, quantity, price=None): ...    # 开多 / 平空
    def sell(self, quantity, price=None): ...   # 开空 / 平多
    def close(self): ...                        # 平掉当前仓位

    # 查询接口
    @property
    def position(self) -> Position | None
    @property
    def balance(self) -> float
    @property
    def equity(self) -> float
    def history(self, n: int) -> list[Bar]      # 最近N根K线
```

## 数据采集

三个交易所各自实现 Collector：

| 交易所 | API端点 | 每次上限 |
|--------|---------|---------|
| Binance | GET /fapi/v1/klines | 1500条 |
| OKX | GET /api/v5/market/history-candles | 100条 |
| HTX | GET /linear-swap-ex/market/history/kline | 2000条 |

- 无需API Key（公开行情接口）
- 自动分页拉取
- 支持增量更新：从SQLite中最新时间戳继续拉取

### SQLite 表结构

```sql
CREATE TABLE klines (
    symbol    TEXT,
    interval  TEXT,
    timestamp INTEGER,
    open      REAL,
    high      REAL,
    low       REAL,
    close     REAL,
    volume    REAL,
    exchange  TEXT,
    PRIMARY KEY (exchange, symbol, interval, timestamp)
);
```

## 回测报告指标

| 指标 | 字段名 | 计算方式 |
|------|--------|---------|
| 总收益率 | net_return | (equity_final - initial) / initial |
| 年化收益率 | annual_return | 按交易天数折算 |
| 最大回撤 | max_drawdown | 权益曲线峰值到谷值的最大跌幅 |
| 最大回撤持续时间 | max_dd_duration | 回撤恢复耗时 |
| 夏普比率 | sharpe_ratio | (年化收益 - 无风险利率) / 年化波动率 |
| 索提诺比率 | sortino_ratio | 仅考虑下行波动 |
| 胜率 | win_rate | 盈利交易 / 总交易次数 |
| 盈亏比 | profit_factor | 总盈利 / 总亏损 |
| 总交易次数 | total_trades | |
| 多头交易次数 | long_trades | |
| 空头交易次数 | short_trades | |
| 平均持仓时间 | avg_hold_time | |
| 总手续费 | total_commission | |
| 总资金费用 | total_funding | |

## Web 报告页面

技术栈：FastAPI + 静态 HTML + ECharts

### 页面布局

- **顶部栏** — 策略名称、交易对、时间范围
- **KPI 卡片行** — 4个核心指标：总收益率、最大回撤、夏普比率、胜率
- **权益曲线** — 折线图，X轴时间，Y轴账户权益
- **回撤曲线** — 面积图，展示回撤深度
- **每笔交易盈亏** — 柱状图，绿色盈利/红色亏损
- **交易明细表** — 时间、方向、价格、数量、盈亏、手续费

### API 路由

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | / | 返回 index.html |
| GET | /api/reports | 获取所有回测报告列表 |
| GET | /api/reports/{id} | 获取单个报告详情（含指标+交易记录） |
| POST | /api/backtest/run | 触发回测运行 |

## CLI 命令

```bash
# 拉取数据
python -m backtest collect --exchange binance --symbol BTCUSDT --interval 1h \
    --start 2024-01-01 --end 2024-12-31

# 运行回测
python -m backtest run --strategy strategies/example_ma_cross.py \
    --symbol BTCUSDT --interval 1h \
    --start 2024-01-01 --end 2024-12-31 \
    --balance 10000 --leverage 10

# 启动Web查看报告
python -m backtest web --port 8000
```

## 默认配置

```yaml
# config/default.yaml
backtest:
  initial_balance: 10000
  leverage: 10
  commission_rate: 0.0004      # taker 0.04%
  funding_rate: 0.0001         # 0.01% / 8h
  maintenance_margin: 0.005    # 维持保证金率 0.5%
```

## 依赖

- Python >= 3.11
- fastapi + uvicorn — Web服务
- httpx — 异步HTTP客户端（拉取K线）
- pyyaml — 配置文件解析
- ECharts (CDN) — 前端图表渲染
