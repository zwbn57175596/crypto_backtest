# Crypto Backtest

U本位永续合约历史数据回测系统，基于事件驱动架构。支持从币安、OKX、HTX 拉取 K 线数据，运行自定义策略回测，通过 Web 页面查看可视化报告。

## 功能特性

- **事件驱动回测引擎** — 逐根 K 线推送，精确模拟交易执行
- **多交易所支持** — 币安、OKX、HTX 一键拉取历史行情
- **完整交易模拟** — 撮合机制、保证金、杠杆、手续费、资金费率、强平检测
- **灵活策略编写** — 继承 BaseStrategy，实现 on_bar() 方法即可
- **14 个回测指标** — 收益率、回撤、夏普比率、胜率、盈亏比等
- **Web 可视化报告** — 权益曲线、回撤分析、交易明细、实时查看
- **支持多种周期** — 1m、5m、15m、1h、4h、1d K 线时间间隔
- **参数优化** — 网格搜索 / Optuna 贝叶斯 / Numba CPU 加速 / CUDA GPU 加速
- **CUDA GPU 加速** — 基于 Numba CUDA，百万级参数组合秒级完成（需 NVIDIA GPU）

## 项目结构

```
crypto_backtest/
├── src/backtest/                  # 核心代码
│   ├── __main__.py               # CLI 入口
│   ├── engine.py                 # BacktestEngine 事件循环协调器
│   ├── data_feed.py              # DataFeed 数据推送模块
│   ├── exchange.py               # SimExchange 模拟交易所
│   ├── strategy.py               # BaseStrategy 基类
│   ├── reporter.py               # 指标计算及报告生成
│   ├── models.py                 # 数据模型 (Bar, Order, Position, Trade)
│   ├── collector/                # 数据采集模块
│   │   ├── base.py               # BaseCollector 基类
│   │   ├── binance.py            # 币安 K 线采集
│   │   ├── okx.py                # OKX K 线采集
│   │   └── htx.py                # HTX K 线采集
│   ├── numba_simulate.py         # Numba JIT CPU 加速模拟
│   ├── cuda_exchange.py          # CUDA GPU 交易所 device 函数
│   ├── cuda_runner.py            # CUDA GPU 网格搜索优化器
│   ├── cuda_strategies/          # CUDA 策略 kernel
│   │   ├── __init__.py           # 策略注册表
│   │   └── consecutive_reverse.py # 连续反转策略 CUDA kernel
│   └── web/                      # Web 报告服务
│       ├── app.py                # FastAPI 应用
│       ├── routes.py             # API 路由
│       └── static/
│           └── index.html        # 单页报告页面 (ECharts)
├── strategies/
│   └── example_ma_cross.py       # 示例：均线交叉策略
├── config/
│   └── default.yaml              # 默认回测配置
├── data/
│   └── klines.db                 # SQLite 历史 K 线数据库
└── tests/                        # 单元测试
```

### 核心模块说明

| 文件 | 功能说明 |
|------|--------|
| engine.py | 事件循环协调器，逐根K线推送数据、撮合、强平、计算盈亏 |
| data_feed.py | 从SQLite按时间逐条推送K线给策略 |
| exchange.py | 模拟交易所：订单撮合、仓位管理、保证金计算、强平逻辑 |
| strategy.py | BaseStrategy 基类，用户通过继承实现交易逻辑 |
| reporter.py | 收集交易记录，计算14个回测指标 |
| models.py | 数据模型定义（K线、订单、仓位、成交） |
| collector/\*.py | 三个交易所的数据采集器，支持增量更新 |
| web/app.py | FastAPI Web服务，提供API和静态页面 |

## 快速开始

### 1. 安装依赖

```bash
pip install -e .
```

### 2. 拉取历史 K 线数据

选择一个交易所（binance、okx、htx），拉取指定交易对和周期的历史行情：

```bash
# 从币安拉取 BTCUSDT 1小时 K线（2024年全年）
python -m backtest collect \
    --exchange binance \
    --symbol BTCUSDT \
    --interval 1h \
    --start 2024-01-01 \
    --end 2024-12-31
```

数据默认存储在 `data/klines.db`。支持以下参数：

| 参数 | 说明 | 示例 |
|------|------|------|
| --exchange | 交易所（必需） | binance / okx / htx |
| --symbol | 交易对（必需） | BTCUSDT, ETHUSDT 等 |
| --interval | K线周期（必需） | 1m / 5m / 15m / 1h / 4h / 1d |
| --start | 开始日期（必需） | 2024-01-01 |
| --end | 结束日期（必需） | 2024-12-31 |
| --db | 数据库路径（可选） | 默认 data/klines.db |

### 3. 运行回测

使用策略文件运行回测：

```bash
# 运行均线交叉策略
python -m backtest run \
    --strategy strategies/example_ma_cross.py \
    --symbol BTCUSDT \
    --interval 1h \
    --start 2024-01-01 \
    --end 2024-12-31 \
    --balance 10000 \
    --leverage 10
```

主要参数说明：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| --strategy | 策略文件路径（必需） | - |
| --symbol | 交易对（必需） | - |
| --interval | K线周期（必需） | - |
| --exchange | 数据来源交易所 | binance |
| --start | 回测开始日期 | - |
| --end | 回测结束日期 | - |
| --balance | 初始资金（USDT） | 由 config/default.yaml 决定 |
| --leverage | 杠杆倍数 | 由 config/default.yaml 决定 |
| --db | 数据库路径 | data/klines.db |

回测完成后，在命令行输出关键指标：

```
Backtest Complete: MaCrossStrategy
  Net Return:     45.32%
  Max Drawdown:   -12.50%
  Sharpe Ratio:   1.85
  Win Rate:       58.50%
  Total Trades:   245
  Total Commission: 98.50 USDT

Report saved. View with: python -m backtest web
```

### 4. 启动 Web 查看报告

```bash
# 启动Web服务，访问 http://localhost:8000
python -m backtest web --port 8000
```

Web 报告展示：

- **KPI 卡片** — 总收益率、最大回撤、夏普比率、胜率
- **权益曲线** — 账户权益变化趋势
- **回撤曲线** — 回撤深度和恢复过程
- **交易明细** — 每笔交易的方向、价格、数量、盈亏
- **P&L 分析** — 按交易统计盈利/亏损分布

## 参数优化

支持四种优化方式，适用于不同规模的参数搜索：

| 方式 | CLI 参数 | 适用场景 | 速度 |
|------|----------|---------|------|
| Python 网格搜索 | `--method grid` | 通用策略，少量参数组合 | 基准 |
| Optuna 贝叶斯优化 | `--method optuna` | 通用策略，智能搜索 | 取决于 n_trials |
| Numba CPU 加速 | `--method numba-grid` | ConsecutiveReverse 策略，CPU 多核并行 | ~50-200x |
| **CUDA GPU 加速** | `--method cuda-grid` | ConsecutiveReverse 策略，GPU 大规模并行 | ~1000x+ |
| **CUDA 自动寻优** | `--method cuda-auto` | ConsecutiveReverse 策略，策略内置搜索空间 + 两阶段粗到细搜索 | ~1000x+ |

### 基本用法

```bash
# 网格搜索（通用，任意策略）
python -m backtest optimize \
    --strategy strategies/example_ma_cross.py \
    --symbol BTCUSDT --interval 1h \
    --start 2024-01-01 --end 2024-12-31 \
    --balance 10000 --leverage 10 \
    --params "short_period=5:15:1,long_period=20:40:5" \
    --method grid --objective sharpe_ratio

# CUDA GPU 加速（需要 NVIDIA GPU + CUDA 环境）
python -m backtest optimize \
    --strategy strategies/consecutive_reverse.py \
    --symbol BTCUSDT --interval 1h \
    --start 2024-01-01 --end 2024-12-31 \
    --balance 1000 --leverage 50 \
    --params "CONSECUTIVE_THRESHOLD=3:8:1,POSITION_MULTIPLIER=1.0:1.5:0.1,INITIAL_POSITION_PCT=0.005:0.03:0.005,PROFIT_CANDLE_THRESHOLD=1:5:1" \
    --method cuda-grid --objective sharpe_ratio

# CUDA 自动寻优（使用策略内置 OPTIMIZE_SPACE + 两阶段细化）
python -m backtest optimize \
    --strategy strategies/consecutive_reverse.py \
    --symbol BTCUSDT --interval 1h \
    --start 2024-01-01 --end 2024-12-31 \
    --balance 1000 --leverage 50 \
    --method cuda-auto --objective sharpe_ratio
```

### 优化参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| --params | 参数搜索空间，格式 `名称=最小:最大:步长` 或 `名称=值1\|值2\|值3`。若策略定义了 `OPTIMIZE_SPACE`，可省略 | 可选 |
| --method | 优化方式：`grid` / `optuna` / `numba-grid` / `cuda-grid` / `cuda-auto` | grid |
| --objective | 优化目标指标 | sharpe_ratio |
| --n-jobs | CPU 并行进程数（grid/numba-grid） | 自动检测 |
| --n-trials | Optuna 搜索次数 | 100 |
| --top | 控制台显示前 N 名 | 10 |
| --save-top | 保存前 N 名到数据库 | 1000 |
| --report-top | 保存前 N 名完整报告（含权益曲线） | 3 |

### CUDA GPU 加速

#### 环境要求

| 组件 | 要求 |
|------|------|
| 操作系统 | Windows 10/11 或 Linux（macOS 不支持 NVIDIA CUDA） |
| GPU | NVIDIA GPU，计算能力 >= 3.5（推荐 RTX 3060 以上） |
| 驱动 | NVIDIA 驱动 >= 520（支持 CUDA 11.8+） |
| CUDA Toolkit | 11.8 或 12.x |
| Python | 3.9 - 3.12 |

#### 安装步骤（Windows 11 + NVIDIA GPU）

**第一步：确认 GPU 和驱动**

```bash
# 查看 GPU 信息和驱动版本
nvidia-smi
```

输出应包含 GPU 型号和 CUDA Version（如 `CUDA Version: 12.x`）。如果命令不存在，需先安装 [NVIDIA 驱动](https://www.nvidia.com/drivers)。

**第二步：安装 CUDA Toolkit**

推荐通过 conda 安装（自动管理版本兼容）：

```bash
# 方式一：conda 安装（推荐）
conda install -c conda-forge cudatoolkit=11.8

# 方式二：pip 安装 CUDA 运行时
pip install nvidia-cuda-runtime-cu11
```

或从 [NVIDIA 官网](https://developer.nvidia.com/cuda-downloads) 下载安装完整 CUDA Toolkit。

**第三步：安装 Numba（含 CUDA 支持）**

```bash
# conda 安装（推荐，自动处理依赖）
conda install numba

# 或 pip 安装
pip install numba
```

**第四步：验证安装**

```python
# 运行验证脚本
python -c "
from numba import cuda
print(f'CUDA available: {cuda.is_available()}')
if cuda.is_available():
    gpu = cuda.get_current_device()
    print(f'GPU: {gpu.name}')
    free, total = cuda.current_context().get_memory_info()
    print(f'VRAM: {free/1024**3:.1f} GB free / {total/1024**3:.1f} GB total')
    print(f'Compute Capability: {gpu.compute_capability}')
"
```

期望输出：

```
CUDA available: True
GPU: NVIDIA GeForce RTX 3080
VRAM: 8.5 GB free / 10.0 GB total
Compute Capability: (8, 6)
```

**第五步：安装项目依赖**

```bash
pip install -e .
```

#### 安装步骤（Linux）

```bash
# Ubuntu / Debian
sudo apt install nvidia-driver-535 nvidia-cuda-toolkit
pip install numba
pip install -e .

# 验证
python -c "from numba import cuda; print(cuda.is_available())"
```

#### 运行 CUDA 优化

```bash
python -m backtest optimize \
    --strategy strategies/consecutive_reverse.py \
    --symbol BTCUSDT --interval 1h \
    --start 2020-01-01 --end 2025-12-31 \
    --balance 1000 --leverage 50 \
    --params "CONSECUTIVE_THRESHOLD=2:8:1,POSITION_MULTIPLIER=1.0:1.5:0.1,INITIAL_POSITION_PCT=0.005:0.03:0.005,PROFIT_CANDLE_THRESHOLD=1:5:1" \
    --method cuda-grid --objective sharpe_ratio
```

输出示例：

```
Loaded 8760 bars, preparing GPU...
Running 1800 combos in 1 batch(es) (batch_size=500000)
[########################################] 1800/1800 (100.0%)

Optimization Complete: 1800 trials, best sharpe_ratio = 2.3456 (3.2s)

 Rank | CONSECUTIVE_THRESHOLD | POSITION_MULTIPLIER | ...
------+----------------------+--------------------+-----
    1 |                    5 |                1.1 | ...
    2 |                    4 |                1.2 | ...
    ...
```

#### 显存与分批

CUDA 优化器会自动检测 GPU 可用显存并分批处理：

- 每个参数组合约占 320 字节 GPU 显存
- RTX 3080 (10GB) 一批可处理约 ~50 万组合
- 100 万组合 → 自动分为 2 批
- 如需手动控制，可在代码中设置 `batch_size` 参数

#### 常见问题

**Q: `numba.cuda.cudadrv.error.CudaSupportError: Error at driver init`**

A: CUDA 驱动未正确安装。运行 `nvidia-smi` 确认驱动正常。

**Q: `No CUDA kernel registered for 'XxxStrategy'`**

A: 当前仅 ConsecutiveReverseStrategy 有 CUDA 实现。其他策略请使用 `--method numba-grid` 或 `--method grid`。

**Q: 显存不足 (Out of Memory)**

A: 优化器会自动分批。如果仍报错，可能是其他程序占用了显存，关闭后重试。

**Q: CUDA 结果和 CPU 结果不一致**

A: 正常情况下 CUDA 和 CPU 结果应在 1e-6 相对误差内一致。如有较大差异，请提交 issue。

**Q: macOS 能否使用 CUDA 加速？**

A: 不能。macOS 不支持 NVIDIA CUDA。请使用 `--method numba-grid`（Numba CPU 加速）作为替代。

## 策略编写指南

### BaseStrategy API

所有策略需继承 `BaseStrategy` 基类，实现以下方法：

```python
from backtest.strategy import BaseStrategy
from backtest.models import Bar

class MyStrategy(BaseStrategy):
    # 策略初始化（可选）
    def on_init(self) -> None:
        pass

    # K线处理（必需）
    # 每根新K线到来时调用一次
    def on_bar(self, bar: Bar) -> None:
        pass
```

### 交易操作 API

在 `on_bar()` 中调用以下方法执行交易：

| 方法 | 说明 | 示例 |
|------|------|------|
| `self.buy(quantity, price=None)` | 买入/平空 (USDT金额) | `self.buy(1000)` 买入1000USDT的多头 |
| `self.sell(quantity, price=None)` | 卖出/平多 (USDT金额) | `self.sell(1000)` 卖出1000USDT的空头 |
| `self.close()` | 平掉当前仓位 | `self.close()` 无论多空都平仓 |

- `price=None` 时为市价单（以 K 线 open 成交）
- `price` 非空时为限价单（需要等待触发条件满足）

### 查询接口

| 属性/方法 | 返回值 | 说明 |
|----------|--------|------|
| `self.position` | Position \| None | 当前仓位（None表示无仓位） |
| `self.balance` | float | 账户余额（USDT） |
| `self.equity` | float | 账户权益 = 余额 + 未实现盈亏 |
| `self.history(n)` | list[Bar] | 最近n根K线，按时间升序 |

### 数据模型

```python
# Bar 数据模型
@dataclass
class Bar:
    symbol: str          # 交易对，如 BTCUSDT
    timestamp: int       # 毫秒时间戳
    open: float          # 开盘价
    high: float          # 最高价
    low: float           # 最低价
    close: float         # 收盘价
    volume: float        # 成交量
    interval: str        # K线周期: 1m/5m/15m/1h/4h/1d

# Position 仓位模型
@dataclass
class Position:
    symbol: str          # 交易对
    side: str            # "long" 或 "short"
    quantity: float      # 持仓量（USDT）
    entry_price: float   # 开仓均价
    leverage: int        # 杠杆倍数
    unrealized_pnl: float # 未实现盈亏
    margin: float        # 占用保证金
```

### 示例：均线交叉策略

```python
from backtest.strategy import BaseStrategy
from backtest.models import Bar

class MaCrossStrategy(BaseStrategy):
    """
    均线交叉策略
    - 短期均线 > 长期均线，做多
    - 短期均线 < 长期均线，做空
    """
    short_period = 7      # 短期均线周期
    long_period = 25      # 长期均线周期
    trade_quantity = 1000.0  # 每次交易金额 (USDT)

    def on_bar(self, bar: Bar) -> None:
        # 获取历史K线
        bars = self.history(self.long_period)
        if len(bars) < self.long_period:
            return  # 数据不足，等待

        # 计算当前均线
        short_ma = sum(b.close for b in bars[-self.short_period:]) / self.short_period
        long_ma = sum(b.close for b in bars) / self.long_period

        # 获取前一根K线的均线（用于交叉判断）
        prev_bars = self.history(self.long_period + 1)
        if len(prev_bars) < self.long_period + 1:
            return
        prev_short = sum(b.close for b in prev_bars[-self.short_period - 1:-1]) / self.short_period
        prev_long = sum(b.close for b in prev_bars[:-1]) / self.long_period

        pos = self.position

        # 金叉信号：短期均线从下穿过长期均线
        if prev_short <= prev_long and short_ma > long_ma:
            if pos is None:
                # 无仓位，做多
                self.buy(self.trade_quantity)
            elif pos.side == "short":
                # 持空头，先平仓再做多
                self.close()
                self.buy(self.trade_quantity)

        # 死叉信号：短期均线从上穿过长期均线
        elif prev_short >= prev_long and short_ma < long_ma:
            if pos is None:
                # 无仓位，做空
                self.sell(self.trade_quantity)
            elif pos.side == "long":
                # 持多头，先平仓再做空
                self.close()
                self.sell(self.trade_quantity)
```

## 回测报告指标

回测完成后生成 14 个性能指标，用于评估策略质量：

| 指标 | 字段名 | 计算方式 | 说明 |
|------|--------|--------|------|
| 总收益率 | net_return | (最终权益 - 初始资金) / 初始资金 | 整个回测周期的盈亏百分比 |
| 年化收益率 | annual_return | 按交易天数换算年收益 | 假设全年365天计算的年化收益 |
| 最大回撤 | max_drawdown | 权益从峰值下跌到谷值的最大幅度 | 评估策略风险的关键指标 |
| 最大回撤持续时间 | max_dd_duration | 回撤恢复所需的交易日数 | 反映策略恢复能力 |
| 夏普比率 | sharpe_ratio | (年化收益 - 无风险利率) / 年化波动率 | 每单位风险获得的超额收益 |
| 索提诺比率 | sortino_ratio | (年化收益 - 无风险利率) / 下行波动率 | 仅惩罚负收益的风险调整收益 |
| 胜率 | win_rate | 盈利交易数 / 总交易数 | 交易获利概率 |
| 盈亏比 | profit_factor | 总盈利 / 总亏损 | 平均盈利 vs 平均亏损的比例 |
| 总交易次数 | total_trades | 完成的往返交易次数 | 包括多头和空头 |
| 多头交易次数 | long_trades | 多头完成交易数 | 仅统计已平仓的多头 |
| 空头交易次数 | short_trades | 空头完成交易数 | 仅统计已平仓的空头 |
| 平均持仓时间 | avg_hold_time | 总持仓时间 / 交易次数 | 以交易日数计 |
| 总手续费 | total_commission | 所有交易的累计手续费 | USDT 单位 |
| 总资金费用 | total_funding | 所有资金费率结算的累计支付 | USDT 单位（多头为正支出，空头为正收入） |

## 支持的交易所和周期

### 支持的交易所

| 交易所 | 标识 | API 说明 |
|--------|------|--------|
| 币安 (Binance) | binance | 永续合约 FAPI，无需API Key |
| OKX | okx | 交割/永续合约，无需API Key |
| 火币 (HTX) | htx | U本位永续合约，无需API Key |

### 支持的 K 线周期

| 周期 | 标识 | 说明 |
|------|------|------|
| 1 分钟 | 1m | 适合高频策略 |
| 5 分钟 | 5m | 日内短线 |
| 15 分钟 | 15m | 日内短线 |
| 1 小时 | 1h | 中期趋势 |
| 4 小时 | 4h | 中期趋势 |
| 1 天 | 1d | 长期趋势 |

## 配置说明

编辑 `config/default.yaml` 修改默认参数，在 `run` 命令中未指定参数时使用：

```yaml
backtest:
  initial_balance: 10000        # 初始资金（USDT）
  leverage: 10                  # 杠杆倍数
  commission_rate: 0.0004       # 交易手续费率（taker 0.04%）
  funding_rate: 0.0001          # 每8小时资金费率（0.01%）
  maintenance_margin: 0.005     # 维持保证金率（0.5%）
```

### 参数详解

| 参数 | 默认值 | 范围 | 说明 |
|------|--------|------|------|
| initial_balance | 10000 | > 0 | 回测账户起始资金 |
| leverage | 10 | 1-125 | 杠杆倍数，越高风险越大 |
| commission_rate | 0.0004 | > 0 | 每次交易的手续费比例（币安taker为0.04%） |
| funding_rate | 0.0001 | > 0 | 每8小时的资金费率（永续合约独有） |
| maintenance_margin | 0.005 | > 0 | 触发强平的保证金率（0.5%时强平） |

## 工作流程

### 回测引擎执行流程

每根 K 线的处理顺序：

1. **资金费率结算** — 每 8 小时（00:00/08:00/16:00 UTC）
   - 多头支付资金费率（或收取负值费率）
   - 空头反向结算

2. **订单撮合** — 遍历所有待处理订单
   - 市价单：以 K 线 open 价成交
   - 限价多单：当 K 线 low <= 订单价格时成交
   - 限价空单：当 K 线 high >= 订单价格时成交

3. **仓位更新** — 计算未实现盈亏
   - 多头: `unrealized_pnl = (close - entry_price) × quantity`
   - 空头: `unrealized_pnl = (entry_price - close) × quantity`

4. **强平检测** — 维持保证金率 < maintenance_margin 时
   - 自动平掉亏损仓位
   - 记录强平事件

5. **策略推送** — 调用 `strategy.on_bar(bar)`
   - 策略可访问当前仓位、余额、历史K线
   - 发出买卖信号

### 数据流

```
交易所API
   ↓
拉取K线 (collect)
   ↓
SQLite 数据库
   ↓
DataFeed (顺序读取)
   ↓
BacktestEngine (事件循环)
   ├─ SimExchange (撮合)
   ├─ Strategy (决策)
   └─ Reporter (统计)
   ↓
回测报告 (JSON)
   ↓
Web 可视化
```

## 常见问题

**Q: 回测数据不足怎么办？**

A: 使用 `collect` 命令增量拉取更多数据。查询 `data/klines.db` 验证数据完整性：

```bash
sqlite3 data/klines.db "SELECT COUNT(*) FROM klines WHERE symbol='BTCUSDT'"
```

**Q: 如何自定义交易费用？**

A: 编辑 `config/default.yaml` 修改 `commission_rate` 和 `funding_rate`，或在 CLI 中使用参数覆盖。

**Q: 回测结果如何导出？**

A: 回测报告自动保存到 `data/reports.db`，可通过 Web 界面查看或直接查询数据库。

**Q: 支持多币种回测吗？**

A: 支持，只需准备不同的K线数据并指定不同的 `--symbol` 参数即可。

**Q: 如何添加新的交易所？**

A: 在 `src/backtest/collector/` 中新建文件，继承 `BaseCollector` 并实现 `fetch()` 方法。

## 技术栈

- **Python 3.11+** — 核心语言
- **FastAPI** — Web 框架
- **SQLite** — 数据存储
- **ECharts** — 前端图表
- **httpx** — 异步 HTTP 客户端
- **PyYAML** — 配置解析
- **Numba** — JIT 编译加速（CPU `@njit` + GPU `@cuda.jit`）
- **Numba CUDA** — GPU 并行计算（可选，需 NVIDIA GPU）

## API Reference

The web server (`python -m backtest web`) exposes the following endpoints:

### Pages

| Endpoint | Description |
|----------|-------------|
| `GET /` | 回测报告查看器 |
| `GET /optimize` | 参数优化结果页面 |

### Backtest Reports

| Endpoint | Description |
|----------|-------------|
| `GET /api/reports` | 列出所有回测报告 (id, strategy, symbol, interval, created_at) |
| `GET /api/reports/{id}` | 获取单个报告详情 (含 equity_curve, trades, 所有指标) |

### Optimization Results

| Endpoint | Parameters | Description |
|----------|-----------|-------------|
| `GET /api/optimize_results` | `?strategy=X&symbol=Y` | 列出优化结果，按 score 降序。支持策略名和交易对筛选 |
| `GET /api/optimize_results/strategies` | — | 列出所有策略/交易对组合 (含 count, best_score, latest_date) |

### Response Examples

**GET /api/optimize_results**

```json
[
  {
    "id": 1,
    "strategy": "ShadowPowerStrategy",
    "symbol": "BTCUSDT",
    "interval": "15m",
    "start_date": "2024-01-01",
    "end_date": "2024-06-30",
    "objective": "sharpe_ratio",
    "score": 2.31,
    "params_json": "{\"DECISION_LEN\": 40, \"SHADOW_FACTOR\": 2.5}",
    "report_json": "{\"net_return\": 1.85, \"max_drawdown\": 0.12, ...}",
    "created_at": "2026-04-20T10:00:00+00:00"
  }
]
```

**GET /api/optimize_results/strategies**

```json
[
  {
    "strategy": "ShadowPowerStrategy",
    "symbol": "BTCUSDT",
    "count": 120,
    "best_score": 2.31,
    "latest_date": "2026-04-20T10:00:00+00:00"
  }
]
```

## 许可证

MIT
