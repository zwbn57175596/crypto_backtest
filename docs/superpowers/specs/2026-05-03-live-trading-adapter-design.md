# Live Trading Adapter 设计文档

**日期**: 2026-05-03  
**范围**: 为现有回测框架新增实盘交易能力，支持任意 `BaseStrategy` 子类对接 Binance U本位合约 API

---

## 目标

在不修改现有回测代码的前提下，实现一个通用实盘适配层，让 `consecutive_reverse.py` 等任意策略可以在真实交易所运行，并具备 B 级可靠性（断线重连、崩溃恢复、订单确认）。

---

## 整体架构

新增三个文件，**现有代码零修改**（`BaseStrategy` 除外，增加两个可选钩子）：

```
src/backtest/
├── live_exchange.py   ← LiveExchange：Binance API 适配层
├── live_feed.py       ← LiveFeed：K 线定时轮询
└── live_engine.py     ← LiveEngine：实盘事件循环
```

对应关系：

```
DataFeed       ←→  LiveFeed
SimExchange    ←→  LiveExchange
BacktestEngine ←→  LiveEngine
```

运行时数据流：

```
LiveFeed（REST 轮询，等待 bar 收盘 + 5s buffer）
    ↓ Bar
LiveEngine.run()
    ├── live_exchange.sync()         ← 从 Binance 刷新余额/持仓缓存
    ├── strategy._push_bar(bar)      ← 调用 on_bar()
    │       └── live_exchange.submit_order()  ← 发真实订单
    ├── live_exchange.wait_fills()   ← 轮询直到 FILLED/FAILED
    └── strategy.save_state() → live_state/{symbol}_{interval}.json
```

---

## LiveExchange

`BaseStrategy` 通过四个接口与 exchange 交互：`submit_order`、`get_position`、`balance`、`equity`。`LiveExchange` 实现相同签名，内部调用 Binance `UMFutures` 客户端（`binance-futures-connector`）。

### 接口

```python
class LiveExchange:
    def __init__(self, client: UMFutures, symbol: str, leverage: int,
                 commission_rate: float, dry_run: bool = False): ...

    # LiveEngine 调用
    def sync(self) -> None: ...
    def wait_fills(self, timeout: float = 30.0) -> None: ...

    # BaseStrategy 调用（与 SimExchange 签名完全一致）
    def submit_order(self, symbol, side, type_, quantity, price=None) -> Order: ...
    def get_position(self, symbol: str) -> Position | None: ...

    @property
    def balance(self) -> float: ...
    @property
    def equity(self) -> float: ...
```

### 关键细节

**数量换算**：框架使用 USDT 名义价值，Binance 使用合约张数。
```
BTC 张数 = USDT 名义价值 / 当前价格
按 LOT_SIZE 步进取整（通过 GET /fapi/v1/exchangeInfo 获取）
```

**sync() 数据源**：
- `GET /fapi/v2/balance` → `balance`
- `GET /fapi/v2/positionRisk` → 转换为框架 `Position` dataclass
- `GET /fapi/v1/premiumIndex` → 当前价格（用于 equity 计算）

**重试策略**：所有 API 调用包在 `_retry(fn, attempts=3, backoff=2s)` 里，三次失败后抛异常。

**dry_run 模式**：`submit_order()` 只打印日志不发单；`sync()` 正常拉取（可观察真实账户状态）。

---

## LiveFeed

```python
class LiveFeed:
    def __init__(self, client: UMFutures, symbol: str, interval: str,
                 close_buffer_sec: float = 5.0): ...

    def __iter__(self) -> Iterator[Bar]: ...
```

**时序**：在每根 bar 收盘时刻 + `close_buffer_sec`（默认 5s）后，拉取 `GET /fapi/v1/klines?limit=2`，取倒数第二根（已完全收盘）构造 `Bar`。

**漏 bar 补全**：每次 yield 前对比上一根 bar 的 timestamp。若检测到跳空（宕机期间缺失 N 根），从 Binance 拉取历史 K 线依次补发，确保策略内部状态正常推进。

**间隔计算**：根据 `interval` 字符串计算下一根收盘的绝对 UTC 时刻，精确 `sleep` 到秒，避免忙轮询。

---

## LiveEngine

### 启动序列

```
1. 创建 UMFutures client（testnet 或 mainnet）
2. 创建 LiveExchange，调用 sync() 获取真实余额/持仓
3. 创建 strategy，调用 on_init()
4. 若 live_state/{symbol}_{interval}.json 存在，调用 strategy.load_state()
5. 启动时调用 change_leverage() 确保与参数一致
6. 打印启动摘要（余额、持仓、杠杆、模式）
7. 创建 LiveFeed，进入主循环
```

### 主循环

```python
for bar in live_feed:
    try:
        live_exchange.sync()
        strategy._push_bar(bar)
        live_exchange.wait_fills(timeout=30)
        save_state(strategy, bar)
        log_bar_summary(bar, live_exchange)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        alert(f"bar {bar.timestamp} 处理异常: {e}")
        log_exception(e)
        # 跳过当根 bar，下根 bar 继续
```

### 可靠性矩阵

| 场景 | 处理方式 |
|------|---------|
| 网络抖动 | API 调用内置 3 次重试 + 指数退避 |
| 程序崩溃重启 | `sync()` 从 Binance 拉真实状态，`load_state()` 恢复策略计数 |
| 订单未成交 | `wait_fills()` 超时后打印告警，不强制撤单（市价单 <1s 成交） |
| bar 处理异常 | 捕获记录 + 告警，跳过当根 bar，下根继续 |
| 漏 bar | LiveFeed 补发历史 bar，策略状态正常推进 |
| 杠杆不一致 | 启动时调用 `change_leverage()` 校正 |

### 告警

`_alert()` 当前输出到 stderr + 日志文件，预留接口供后续接入 Telegram / 企业微信。

### 干净退出

捕获 `KeyboardInterrupt` / SIGTERM → 打印当前持仓摘要 → 询问是否平仓 → 退出。

---

## BaseStrategy 状态持久化

在 `BaseStrategy` 新增两个可选钩子（默认空实现，回测完全不受影响）：

```python
def save_state(self) -> dict:
    return {}

def load_state(self, state: dict) -> None:
    pass
```

`ConsecutiveReverseStrategy` 覆盖实现：

```python
def save_state(self) -> dict:
    return {
        "consecutive_count": self._consecutive_count,
        "streak_direction": self._streak_direction,
        "profit_candle_count": self._profit_candle_count,
    }

def load_state(self, state: dict) -> None:
    self._consecutive_count = state.get("consecutive_count", 0)
    self._streak_direction = state.get("streak_direction", 0)
    self._profit_candle_count = state.get("profit_candle_count", 0)
```

状态文件路径：`live_state/{symbol}_{interval}.json`

---

## CLI 集成

在 `__main__.py` 新增 `live` 子命令：

```bash
python -m backtest live \
    --strategy strategies/consecutive_reverse.py \
    --symbol BTCUSDT --interval 4h \
    --leverage 17 \
    --commission-rate 0.0004 \
    [--testnet]          # 默认 True，实盘需显式 --no-testnet
    [--dry-run]          # 只打印，不发单
    [--state-dir live_state]

# 密钥通过环境变量注入（避免命令历史泄漏）
export BINANCE_API_KEY=xxx
export BINANCE_SECRET=yyy
```

`--testnet` 默认为 True，实盘需显式传 `--no-testnet`，防止手误。

---

## 依赖

新增运行时依赖：`binance-futures-connector`（已在 `shadow_power_live.py` 中使用）

---

## 未来扩展

当前只实现 Binance。后续 OKX / HTX 支持只需再实现一个 `LiveExchange` 子类，`LiveFeed` 和 `LiveEngine` 不需要改动。可在届时引入 `ExchangeProtocol` ABC 做正式接口约束。
