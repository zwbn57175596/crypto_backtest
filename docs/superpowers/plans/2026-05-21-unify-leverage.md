# Unify Leverage Parameter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 删除策略类中的 `SIZE_LEVERAGE` 参数，将其合并进 `INITIAL_POSITION_PCT`，使 Numba/CUDA 两条计算链路共享唯一的 `leverage`（交易所杠杆），消除 optimizer fallback bug。

**Architecture:** `SIZE_LEVERAGE` 本质是仓位大小放大系数，不是交易所杠杆。将 `INITIAL_POSITION_PCT_new = INITIAL_POSITION_PCT_old × SIZE_LEVERAGE` 后，`_calc_quantity` 和两个 kernel（numba/CUDA）均只剩一个 `exchange_leverage` 参数。optimizer 不再需要读取 `SIZE_LEVERAGE`，fallback 路径消失。

**Tech Stack:** Python 3.11, Numba JIT, CUDA (numba.cuda), pytest

---

## 数值换算速查

| 策略文件 | 旧值 | 新 INITIAL_POSITION_PCT |
|---|---|---|
| `consecutive_reverse.py` | PCT=0.07, SIZE_LEVERAGE=17 | **1.19** |
| `consecutive_reverse_martingale.py` | PCT=0.01, SIZE_LEVERAGE=50 | **0.50** |
| `deploy/trading-consecutive.service` | PCT=0.205, SIZE_LEVERAGE=26 | **5.33** |

---

## 文件变更总览

| 文件 | 操作 |
|---|---|
| `strategies/consecutive_reverse.py` | 删 SIZE_LEVERAGE，更新 INITIAL_POSITION_PCT=1.19，更新 _calc_quantity |
| `strategies/consecutive_reverse_martingale.py` | 删 SIZE_LEVERAGE，更新 PCT=0.50，更新 OPTIMIZE_SPACE/AUTO_OPTIMIZE_CONFIG |
| `src/backtest/numba_simulate.py` | _calc_quantity 删 sizing_leverage 参数，simulate/simulate_close_reopen 签名简化 |
| `src/backtest/cuda_exchange.py` | device_calc_quantity 删 leverage 参数 |
| `src/backtest/cuda_strategies/consecutive_reverse.py` | 删 params[idx,4] 读取，更新 device_calc_quantity 调用 |
| `src/backtest/cuda_strategies/__init__.py` | param_order 删 SIZE_LEVERAGE |
| `src/backtest/optimizer.py` | 删 sizing_lev 变量及 sizing_leverage= 调用 |
| `src/backtest/__main__.py` | 删 --sizing-leverage CLI 参数及处理逻辑 |
| `deploy/trading-consecutive.service` | 删 SIZE_LEVERAGE/LEVERAGE，更新 INITIAL_POSITION_PCT=5.33 |
| `tests/test_consecutive_reverse.py` | 更新 SIZE_LEVERAGE 相关测试 |
| `tests/test_numba_simulate.py` | 删 size_leverage/sizing_leverage 参数传递 |
| `tests/test_cuda_runner.py` | 删 SIZE_LEVERAGE 测试参数 |

---

## Task 1: 更新策略类

**Files:**
- Modify: `strategies/consecutive_reverse.py`
- Modify: `strategies/consecutive_reverse_martingale.py`
- Modify: `tests/test_consecutive_reverse.py`

- [ ] **Step 1: 写失败测试**

在 `tests/test_consecutive_reverse.py` 中替换 SIZE_LEVERAGE 相关测试：

```python
def test_no_size_leverage_attr():
    from strategies.consecutive_reverse import ConsecutiveReverseStrategy
    assert not hasattr(ConsecutiveReverseStrategy, "SIZE_LEVERAGE")
    assert not hasattr(ConsecutiveReverseStrategy, "LEVERAGE")


def test_calc_quantity_uses_initial_pct_only():
    from strategies.consecutive_reverse import ConsecutiveReverseStrategy
    from backtest.exchange import SimExchange
    exchange = SimExchange(balance=1000.0, leverage=50,
                           commission_rate=0.0004, funding_rate=0.0001,
                           maintenance_margin=0.005)
    strategy = ConsecutiveReverseStrategy(exchange=exchange, symbol="BTCUSDT")
    strategy.on_init()
    strategy.INITIAL_POSITION_PCT = 0.10   # effective pct, no SIZE_LEVERAGE
    strategy.CONSECUTIVE_THRESHOLD = 3
    strategy.POSITION_MULTIPLIER = 1.0
    strategy._consecutive_count = 3
    # base = 1000 * 0.10 = 100, n=1, mult=1.0^0=1, qty = 100
    assert strategy._calc_quantity() == pytest.approx(100.0)
```

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/test_consecutive_reverse.py::test_no_size_leverage_attr tests/test_consecutive_reverse.py::test_calc_quantity_uses_initial_pct_only -v
```

预期：`FAILED` — `AssertionError: SIZE_LEVERAGE attr exists`

- [ ] **Step 3: 更新 consecutive_reverse.py**

`strategies/consecutive_reverse.py` 中：

```python
# 删除这行
SIZE_LEVERAGE = 17

# 更新（0.07 × 17 = 1.19）
INITIAL_POSITION_PCT = 1.19     # 初始仓位比例（有效值，已含原 SIZE_LEVERAGE=17）
```

`_calc_quantity` 方法：

```python
def _calc_quantity(self) -> float:
    """计算开仓名义价值（USDT）"""
    if self._consecutive_count < self.CONSECUTIVE_THRESHOLD:
        return 0
    base = self.balance * self.INITIAL_POSITION_PCT
    n = self._consecutive_count - self.CONSECUTIVE_THRESHOLD + 1
    multiplier = self.POSITION_MULTIPLIER ** (n - 1)
    return base * multiplier
```

- [ ] **Step 4: 更新 consecutive_reverse_martingale.py**

```python
# 删除这行
SIZE_LEVERAGE = 50

# 更新（0.01 × 50 = 0.50）
INITIAL_POSITION_PCT = 0.50     # 初始仓位比例（有效值，已含原 SIZE_LEVERAGE=50）
```

`OPTIMIZE_SPACE` 和 `AUTO_OPTIMIZE_CONFIG` 中删除 SIZE_LEVERAGE，并扩大 INITIAL_POSITION_PCT 的搜索范围：

```python
OPTIMIZE_SPACE = {
    "CONSECUTIVE_THRESHOLD": (2, 8, 1),
    "POSITION_MULTIPLIER": (1.0, 1.5, 0.1),
    "INITIAL_POSITION_PCT": (0.1, 1.5, 0.05),
    "PROFIT_CANDLE_THRESHOLD": (1, 10, 1),
}
AUTO_OPTIMIZE_CONFIG = {
    "refine_top_k": 8,
    "refine_space": {
        "CONSECUTIVE_THRESHOLD": {"radius": 1, "step": 1},
        "POSITION_MULTIPLIER": {"radius": 0.1, "step": 0.05},
        "INITIAL_POSITION_PCT": {"radius": 0.1, "step": 0.025, "min": 0.05, "max": 2.0},
        "PROFIT_CANDLE_THRESHOLD": {"radius": 1, "step": 1},
    },
}
```

`_calc_quantity` 方法同上（删掉 `* self.SIZE_LEVERAGE`）：

```python
def _calc_quantity(self) -> float:
    if self._consecutive_count < self.CONSECUTIVE_THRESHOLD:
        return 0
    base = self.balance * self.INITIAL_POSITION_PCT
    n = self._consecutive_count - self.CONSECUTIVE_THRESHOLD + 1
    multiplier = self.POSITION_MULTIPLIER ** (n - 1)
    return base * multiplier
```

- [ ] **Step 5: 运行测试确认通过**

```bash
pytest tests/test_consecutive_reverse.py -v
```

预期：全部 PASS

- [ ] **Step 6: Commit**

```bash
git add strategies/consecutive_reverse.py strategies/consecutive_reverse_martingale.py tests/test_consecutive_reverse.py
git commit -m "refactor: remove SIZE_LEVERAGE, absorb into INITIAL_POSITION_PCT"
```

---

## Task 2: 更新 Numba 模拟内核

**Files:**
- Modify: `src/backtest/numba_simulate.py`
- Modify: `tests/test_numba_simulate.py`

- [ ] **Step 1: 写失败测试**

在 `tests/test_numba_simulate.py` 中，`_run_numba` 函数的调用不再传 `sizing_leverage`：

找到 `_run_numba` 函数，将其改为：

```python
def _run_numba(start: str, end: str, **params):
    start_ts = int(datetime.strptime(start, "%Y-%m-%d %H:%M:%S").replace(
        tzinfo=timezone.utc).timestamp() * 1000)
    end_ts = int(datetime.strptime(end, "%Y-%m-%d %H:%M:%S").replace(
        tzinfo=timezone.utc).timestamp() * 1000)

    bars = load_bars(DB_PATH, "BTCUSDT", "1h", "binance", start_ts, end_ts)
    result = simulate(
        bars,
        threshold=params.get("threshold", 5),
        multiplier=params.get("multiplier", 1.1),
        initial_pct=params.get("initial_pct", 0.01),
        profit_threshold=params.get("profit_threshold", 1),
        leverage=50,
        commission_rate=0.0004,
        funding_rate=0.0001,
        maintenance_margin=0.005,
        initial_balance=1000.0,
    )
    return {
        "net_return": result[0],
        "annual_return": result[1],
        "max_drawdown": result[2],
        "sharpe_ratio": result[3],
        "sortino_ratio": result[4],
        "win_rate": result[5],
        "profit_factor": result[6],
        "total_trades": result[7],
    }
```

`_run_engine` 函数中也删掉 `SIZE_LEVERAGE`：

```python
attrs = {
    "CONSECUTIVE_THRESHOLD": params.get("threshold", 5),
    "POSITION_MULTIPLIER": params.get("multiplier", 1.1),
    "INITIAL_POSITION_PCT": params.get("initial_pct", 0.01),
    "PROFIT_CANDLE_THRESHOLD": params.get("profit_threshold", 1),
}
```

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/test_numba_simulate.py -v -k "not slow" 2>&1 | head -30
```

预期：`TypeError: simulate() got an unexpected keyword argument 'sizing_leverage'` 或类似错误

- [ ] **Step 3: 更新 numba_simulate.py 的 _calc_quantity**

找到文件末尾的 `_calc_quantity` 函数，删除 `sizing_leverage` 参数：

```python
@nb.njit(cache=True)
def _calc_quantity(
    consecutive_count,
    threshold,
    balance,
    initial_pct,
    multiplier,
):
    if consecutive_count < threshold:
        return 0.0
    base = balance * initial_pct
    n = consecutive_count - threshold + 1
    mult = multiplier ** (n - 1)
    return base * mult
```

- [ ] **Step 4: 更新 simulate 函数签名**

找到 `simulate` 函数定义（约第 140 行），将签名从：

```python
sizing_leverage: int,
exchange_leverage: int,
```

改为：

```python
leverage: int,
```

函数体内所有 `exchange_leverage` 替换为 `leverage`，所有 `sizing_leverage` 引用删除。

所有 `_calc_quantity(...)` 调用删掉最后的 `sizing_leverage` 参数，例如：

```python
# 之前
qty = _calc_quantity(consecutive_count, threshold, balance, initial_pct, multiplier, sizing_leverage)
# 之后
qty = _calc_quantity(consecutive_count, threshold, balance, initial_pct, multiplier)
```

对 `simulate_close_reopen`（约第 530 行）做同样处理。

- [ ] **Step 5: 运行测试确认通过**

```bash
pytest tests/test_numba_simulate.py -v -k "not slow"
```

预期：全部 PASS

- [ ] **Step 6: Commit**

```bash
git add src/backtest/numba_simulate.py tests/test_numba_simulate.py
git commit -m "refactor: remove sizing_leverage from numba _calc_quantity and simulate"
```

---

## Task 3: 更新 CUDA 内核

**Files:**
- Modify: `src/backtest/cuda_exchange.py`
- Modify: `src/backtest/cuda_strategies/consecutive_reverse.py`
- Modify: `src/backtest/cuda_strategies/__init__.py`
- Modify: `tests/test_cuda_runner.py`

- [ ] **Step 1: 写失败测试**

在 `tests/test_cuda_runner.py` 中找到所有包含 `"SIZE_LEVERAGE"` 的 params dict，删除该 key：

```python
# 之前
"SIZE_LEVERAGE": [10],
# 之后（删除这行）
```

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/test_cuda_runner.py -v 2>&1 | head -30
```

预期：与 param_order 长度不匹配的错误

- [ ] **Step 3: 更新 cuda_exchange.py 的 device_calc_quantity**

```python
def device_calc_quantity(
    consecutive_count,
    threshold,
    balance,
    initial_pct,
    multiplier,
):
    if consecutive_count < threshold:
        return 0.0
    base = balance * initial_pct
    n = consecutive_count - threshold + 1
    mult = 1.0
    for _ in range(n - 1):
        mult *= multiplier
    return base * mult
```

- [ ] **Step 4: 更新 cuda_strategies/__init__.py**

```python
param_order = [
    "CONSECUTIVE_THRESHOLD",
    "POSITION_MULTIPLIER",
    "INITIAL_POSITION_PCT",
    "PROFIT_CANDLE_THRESHOLD",
]
```

- [ ] **Step 5: 更新 cuda_strategies/consecutive_reverse.py**

找到两处读取 `sizing_leverage = int(params[idx, 4])` 的代码，删除。

所有 `device_calc_quantity(...)` 调用删掉 `sizing_leverage` 参数：

```python
# 之前
qty = device_calc_quantity(
    consecutive_count, threshold, balance, initial_pct, multiplier, sizing_leverage
)
# 之后
qty = device_calc_quantity(
    consecutive_count, threshold, balance, initial_pct, multiplier
)
```

- [ ] **Step 6: 运行测试确认通过**

```bash
pytest tests/test_cuda_runner.py -v
```

预期：全部 PASS（如无 GPU 则 skip）

- [ ] **Step 7: Commit**

```bash
git add src/backtest/cuda_exchange.py src/backtest/cuda_strategies/ tests/test_cuda_runner.py
git commit -m "refactor: remove sizing_leverage from CUDA device_calc_quantity and kernels"
```

---

## Task 4: 更新 Optimizer

**Files:**
- Modify: `src/backtest/optimizer.py`

- [ ] **Step 1: 删除 sizing_lev 变量和 sizing_leverage= 参数**

找到 `NumbaGridOptimizer` 中两处 `sizing_lev` 赋值（约第 622 行和第 651 行），删除：

```python
# 删除这行
sizing_lev = int(params.get("SIZE_LEVERAGE", default_params.get("SIZE_LEVERAGE", self.leverage)))
```

对应的 `simulate_fn(...)` 调用中删除 `sizing_leverage=sizing_lev,` 这一行：

```python
result = simulate_fn(
    bars,
    threshold=int(params.get("CONSECUTIVE_THRESHOLD", default_params.get("CONSECUTIVE_THRESHOLD", 5))),
    multiplier=float(params.get("POSITION_MULTIPLIER", default_params.get("POSITION_MULTIPLIER", 1.1))),
    initial_pct=float(params.get("INITIAL_POSITION_PCT", default_params.get("INITIAL_POSITION_PCT", 0.01))),
    profit_threshold=int(params.get("PROFIT_CANDLE_THRESHOLD", default_params.get("PROFIT_CANDLE_THRESHOLD", 1))),
    leverage=self.leverage,
    commission_rate=self.commission_rate,
    funding_rate=self.funding_rate,
    maintenance_margin=self.maintenance_margin,
    initial_balance=self.balance,
)
```

找到 `_trial_worker` 函数（约第 441 行）的签名和调用，删除 `sizing_leverage` 参数。

- [ ] **Step 2: 运行全部测试**

```bash
pytest tests/ -v --ignore=tests/test_numba_simulate.py -k "optimizer or cuda" 2>&1 | tail -20
```

预期：PASS（或 skip if no GPU/data）

- [ ] **Step 3: Commit**

```bash
git add src/backtest/optimizer.py
git commit -m "refactor: remove sizing_leverage from NumbaGridOptimizer"
```

---

## Task 5: 更新 CLI

**Files:**
- Modify: `src/backtest/__main__.py`

- [ ] **Step 1: 删除 --sizing-leverage 参数**

找到（约第 536 行）并删除：

```python
p_run.add_argument("--sizing-leverage", type=int, default=None, dest="sizing_leverage",
                   help="Strategy sizing leverage (sets LEVERAGE class attr, used in _calc_quantity)")
```

- [ ] **Step 2: 删除 sizing_leverage 处理逻辑**

找到（约第 180 行）并删除：

```python
if args.sizing_leverage is not None:
    attr = "SIZE_LEVERAGE" if hasattr(strategy_class, "SIZE_LEVERAGE") else "LEVERAGE"
    setattr(strategy_class, attr, args.sizing_leverage)
    print(f"  [param] {attr} = {args.sizing_leverage}")
```

- [ ] **Step 3: 清理 _apply_optimize_params 向后兼容代码**

找到（约第 70 行）并删除旧 DB 记录的兼容代码：

```python
# 删除这段
if key == "LEVERAGE" and not hasattr(strategy_class, "LEVERAGE") \
        and hasattr(strategy_class, "SIZE_LEVERAGE"):
    attr = "SIZE_LEVERAGE"
```

改为：

```python
def _apply_optimize_params(strategy_class: type, params: dict) -> None:
    for key, value in params.items():
        if not hasattr(strategy_class, key):
            continue
        try:
            setattr(strategy_class, key, type(getattr(strategy_class, key))(value))
            print(f"  [param] {key} = {value}")
        except (TypeError, ValueError):
            setattr(strategy_class, key, value)
            print(f"  [param] {key} = {value}")
```

- [ ] **Step 4: 运行测试**

```bash
pytest tests/ -v -k "not slow" 2>&1 | tail -20
```

预期：全部 PASS

- [ ] **Step 5: Commit**

```bash
git add src/backtest/__main__.py
git commit -m "refactor: remove --sizing-leverage CLI arg and SIZE_LEVERAGE compat in _apply_optimize_params"
```

---

## Task 6: 更新实盘 Service 文件并部署

**Files:**
- Modify: `deploy/trading-consecutive.service`
- Modify: `.github/workflows/deploy.yml`

- [ ] **Step 1: 更新 service 文件**

`deploy/trading-consecutive.service` 中，将：

```
--INITIAL_POSITION_PCT 0.205 --PROFIT_CANDLE_THRESHOLD 2 \
--SIZE_LEVERAGE 26 --LEVERAGE 50
```

改为：

```
--INITIAL_POSITION_PCT 5.33 --PROFIT_CANDLE_THRESHOLD 2
```

完整的 ExecStart 变为：

```ini
ExecStart=/home/ubuntu/crypto_backtest/.venv/bin/python -m backtest live \
    --strategy strategies/consecutive_reverse.py \
    --symbol BTCUSDT --interval 1h --leverage 50 \
    --no-testnet \
    --env-file /home/ubuntu/.secrets/account1.env \
    --max-order-notional 500 --max-session-loss 300 --order-rate 10/60 \
    --log-dir /home/ubuntu/crypto_backtest/data/logs \
    --CONSECUTIVE_THRESHOLD 8 --POSITION_MULTIPLIER 3.0 \
    --INITIAL_POSITION_PCT 5.33 --PROFIT_CANDLE_THRESHOLD 2
```

- [ ] **Step 2: 更新 deploy.yml tag**

`.github/workflows/deploy.yml` 中将 `v0.1.6` 改为 `v0.1.7`。

- [ ] **Step 3: Commit、打 tag、推送**

```bash
git add deploy/trading-consecutive.service .github/workflows/deploy.yml
git commit -m "ci: deploy v0.1.7"
git tag v0.1.7
git push origin refactor/unify-leverage --tags
```

- [ ] **Step 4: 合并到 master 并触发部署**

```bash
git checkout master
git merge refactor/unify-leverage
git push origin master --tags
gh workflow run deploy.yml -f confirm=deploy
gh run watch --exit-status
```

---

## 自查清单

- [x] **SIZE_LEVERAGE 完全消失**：strategies、numba、cuda、optimizer、CLI 全部删除
- [x] **数值等价**：INITIAL_POSITION_PCT_new = old × SIZE_LEVERAGE_old（0.07×17=1.19，0.01×50=0.50，0.205×26=5.33）
- [x] **Numba 和 CUDA kernel 签名一致**：都是 `_calc_quantity(count, threshold, balance, initial_pct, multiplier)`
- [x] **optimizer fallback 路径消失**：不再有 `params.get("SIZE_LEVERAGE", self.leverage)` 这行
- [x] **实盘和回测使用相同参数语义**：只有一个 `leverage`（交易所杠杆），仓位大小完全由 `INITIAL_POSITION_PCT` 决定
