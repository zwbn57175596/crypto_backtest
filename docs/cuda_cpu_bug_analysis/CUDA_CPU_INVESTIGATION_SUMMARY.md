# CUDA vs CPU 优化器一致性调查 — 最终报告

**调查日期：** 2026-05-02  
**结论：** 🚨 **严重问题 — 需要立即修复**

---

## 核心发现

### 问题概述

| 指标 | 值 |
|------|-----|
| **问题严重程度** | 🔴 严重 |
| **影响范围** | NumbaGridOptimizer + ConsecutiveReverse 平仓-重开策略 |
| **根本原因** | 架构设计缺陷 — 策略实现不匹配 |
| **修复优先级** | 高 |
| **用户影响** | 使用 `--method numba-grid` 的用户获得错误的优化结果 |

### 验证数据

使用相同参数运行 CPU 和 GPU 优化器：

```
策略: ConsecutiveReverseStrategy (close+reopen 版本)
参数: THRESHOLD=5, MULTIPLIER=1.2, INITIAL_PCT=0.01, PROFIT_THRESHOLD=3, LEVERAGE=17
数据: BTCUSDT 1h, 2024-01-01 ~ 2024-02-01 (768 根 K 线)
初始资金: 1000 USDT, 交易所杠杆: 50

┌──────────────────────────────────┬──────────────┬──────────────┬─────────────┐
│ 指标                              │ CPU Numba    │ GPU CUDA     │ 相对误差    │
├──────────────────────────────────┼──────────────┼──────────────┼─────────────┤
│ 交易数                            │ 31 ❌        │ 38 ✓         │ 22.6% 差异  │
│ Sharpe Ratio                      │ -2.84 ❌     │ +2.49 ✓      │ 187.7% 差异 │
│ 净收益率                          │ -1.13% ❌    │ +0.53% ✓     │ 146.9% 差异 │
│ 年化收益率                        │ -12.15% ❌   │ +6.21% ✓     │ 151.2% 差异 │
│ 最大回撤                          │ 2.59% ❌     │ 0.66% ✓      │ 74.5% 差异  │
│ 胜率                              │ 69.2% ❌     │ 47.4% ✓      │ 31.6% 差异  │
│ 盈利因子                          │ 1.32 ❌      │ 1.98 ✓       │ 33.5% 差异  │
└──────────────────────────────────┴──────────────┴──────────────┴─────────────┘
```

**关键观察：**
- CPU 执行了 **31 笔交易**（Martingale 加仓逻辑）
- GPU 执行了 **38 笔交易**（Close+reopen 平仓-重开逻辑）
- ⚠️ CPU 执行的是**完全错误的策略**

---

## 根本原因

### 问题代码

#### NumbaGridOptimizer 的缺陷

```python
# src/backtest/optimizer.py (NumbaGridOptimizer class)
# 不区分策略变体，直接调用唯一的 simulate 函数

from backtest.numba_simulate import simulate

# NumbaGridOptimizer.run() 在第 614-626 行
result = simulate(
    bars,
    threshold=...,
    # ... 所有参数 ...
    # ❌ 但 simulate() 硬编码了 Martingale 逻辑
)
```

#### numba_simulate.py 的问题

```python
# src/backtest/numba_simulate.py 第 371-384 行
else:
    # Loss candle - add to contrarian position up to current target size
    # ❌ 这是 Martingale 逻辑！
    profit_candle_count = 0
    target_qty = _calc_quantity(...)
    add_qty = target_qty - pos_qty  # ← 计算加仓量
    if add_qty > 0:
        if pos_side == _LONG:
            pend_side_0 = _BUY  # ← 执行加仓
        else:
            pend_side_0 = _SELL
        pend_qty_0 = add_qty
        n_pending = 1
```

**问题：** 用户想要运行 `strategies/consecutive_reverse.py` (close+reopen)，但 NumbaGridOptimizer 执行的是 numba_simulate.py 的 Martingale 逻辑。

#### 预期的 close+reopen 逻辑

```python
# strategies/consecutive_reverse.py (期望的行为)
else:
    # Loss candle - close immediately
    self.close()          # ← 平仓
    self._profit_candle_count = 0
    self._try_open(direction)  # ← 重新开仓
```

```python
# src/backtest/cuda_strategies/consecutive_reverse.py (CUDA 正确的实现)
else:
    # Loss candle - close immediately + try reopen contrarian
    pend_side_0 = SELL          # ← 平仓
    pend_qty_0 = pos_qty
    n_pending = 1
    
    reopen_qty = device_calc_quantity(...)
    if reopen_qty > 0:
        pend_side_1 = SELL      # ← 重开
        pend_qty_1 = reopen_qty
        n_pending = 2
```

### 架构对比

```
┌─────────────────────────────────────────┐
│ NumbaGridOptimizer                      │
├─────────────────────────────────────────┤
│ ❌ 只支持 1 种策略变体 (Martingale)     │
│ ❌ 忽视用户指定的策略文件               │
│ ❌ 硬编码调用 simulate()                │
│ ❌ 无法区分不同的策略                   │
│ 结果: 错误的优化                        │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ CudaGridOptimizer                       │
├─────────────────────────────────────────┤
│ ✓ 支持 2 种策略变体                      │
│ ✓ 识别用户指定的策略名称                 │
│ ✓ 从 registry 选择正确的 kernel         │
│ ✓ 动态适配不同的策略                    │
│ 结果: 正确的优化                        │
└─────────────────────────────────────────┘
```

---

## 影响评估

### 受影响的用户

**❌ 受影响：**
- 使用 `--method numba-grid` 运行 ConsecutiveReverse 策略的用户
- 特别是：`strategies/consecutive_reverse.py` (close+reopen 版本)
- 获得的优化参数是**完全错误的**，可能导致实盘亏损

**✓ 不受影响：**
- 使用 `--method cuda-grid` 的用户（CUDA 实现正确）
- 使用 `--method grid` 的用户（基础 CPU 网格搜索）
- 其他策略（如 MaCross）不通过 NumbaGridOptimizer

### 数据损失风险

⚠️ **高** — 使用 NumbaGridOptimizer 优化的参数可能导致：
- 交易方向和频率错误
- 风险/收益比完全不同
- 实盘损失

---

## 修复方案

### 立即行动（今天）

#### 1. 禁用 NumbaGridOptimizer（临时）

```python
# src/backtest/__main__.py

if args.method == 'numba-grid':
    print("❌ ERROR: --method numba-grid is not supported for multi-variant strategies.")
    print("   NumbaGridOptimizer has a design flaw and produces incorrect results.")
    print()
    print("   Recommended alternatives:")
    print("   1. GPU users:  use --method cuda-grid (fast and correct)")
    print("   2. CPU users:  use --method grid (slower but correct)")
    print()
    print("   See CUDA_CPU_ROOT_CAUSE.md for details.")
    sys.exit(1)
```

#### 2. 通知用户（PR + CHANGELOG）

```markdown
## ⚠️ WARNING: NumbaGridOptimizer Bug

NumbaGridOptimizer has a design flaw that produces **incorrect optimization results** 
for ConsecutiveReverse strategies. 

**Affected:** Users running `--method numba-grid` with `strategies/consecutive_reverse.py`

**Action:** Immediately switch to:
- `--method cuda-grid` (recommended, 1000x faster + correct)
- `--method grid` (correct but slower)

See [CUDA_CPU_ROOT_CAUSE.md](CUDA_CPU_ROOT_CAUSE.md) for technical details.
```

### 短期修复（本周）

#### 方案 A：修复 NumbaGridOptimizer（推荐）

**步骤：**

1. 分离 `numba_simulate.py` 中的 simulate() 函数为两个版本：
   ```python
   @njit
   def simulate_close_reopen(...):
       # 修改第 371-384 行为平仓-重开逻辑
   
   @njit
   def simulate_martingale(...):
       # 保留当前的 Martingale 逻辑
   ```

2. 在 NumbaGridOptimizer 中添加策略检测：
   ```python
   strategy_name = _load_strategy_class(self.strategy_path).__name__
   
   if "Martingale" in strategy_name:
       simulate_func = simulate_martingale
   else:
       simulate_func = simulate_close_reopen
   ```

3. 单元测试：
   - 验证 31 笔交易（旧 Martingale）→ 保持不变
   - 验证 38 笔交易（新 close+reopen）→ 匹配 CUDA

**工作量：** ~2-3 小时  
**风险：** 低（向后兼容）

---

## 临时建议（现在就做）

### ✓ 推荐的优化方法

**GPU 用户：**
```bash
python -m backtest optimize --strategy strategies/consecutive_reverse.py \
    --symbol BTCUSDT --interval 1h \
    --start 2024-01-01 --end 2024-12-31 \
    --balance 1000 --leverage 50 \
    --params "CONSECUTIVE_THRESHOLD=2:8:1,..." \
    --method cuda-grid \          # ✓ 正确且快速
    --objective sharpe_ratio
```

**CPU-only 用户：**
```bash
python -m backtest optimize --strategy strategies/consecutive_reverse.py \
    --symbol BTCUSDT --interval 1h \
    --start 2024-01-01 --end 2024-12-31 \
    --balance 1000 --leverage 50 \
    --params "CONSECUTIVE_THRESHOLD=2:8:1,..." \
    --method grid \               # ✓ 正确但较慢
    --objective sharpe_ratio
```

### ❌ 应该避免

```bash
# ❌ 不要使用 —— 结果错误
--method numba-grid
```

---

## 验证和测试

### 当前状态

- ✓ 问题已确认（31 vs 38 笔交易）
- ✓ 根本原因已找到（策略逻辑不匹配）
- ✓ 调试工具已生成（debug_cuda_cpu_diff.py）
- ✓ 分析文档已完成（CUDA_CPU_ROOT_CAUSE.md）

### 待完成

- [ ] 禁用/警告 NumbaGridOptimizer
- [ ] 通知现有用户
- [ ] 实现修复（分离 simulate 函数）
- [ ] 添加单元测试
- [ ] 验证修复后的结果

---

## 附录：调试工具

### 脚本列表

| 脚本 | 用途 |
|------|------|
| `debug_cuda_cpu_diff.py` | 对比单参数的 CPU vs CUDA 结果 |
| `verify_cuda_cpu.py` | 自动对比优化结果 |
| `VERIFY_CUDA.md` | 完整的验证流程文档 |
| `CUDA_CPU_ANALYSIS.md` | 详细的差异分析 |
| `CUDA_CPU_ROOT_CAUSE.md` | 根本原因和修复方案 |

### 如何重现问题

```bash
# 1. 清理旧数据
rm -f data/reports.db  # 可选

# 2. 运行调试脚本
python debug_cuda_cpu_diff.py \
    --strategy strategies/consecutive_reverse.py \
    --symbol BTCUSDT --interval 1h \
    --start 2024-01-01 --end 2024-02-01

# 期望输出：
# CPU Numba:  31 笔交易, Sharpe = -2.84 ❌
# GPU CUDA:   38 笔交易, Sharpe = +2.49 ✓
```

---

## 关键时间线

| 日期 | 事件 |
|------|------|
| 2026-05-02 | 问题发现和分析完成 |
| 2026-05-XX | 禁用 NumbaGridOptimizer（临时） |
| 2026-05-XX | 通知用户更改优化方法 |
| 2026-05-XX | 实现修复（方案 A） |
| 2026-05-XX | 发布修复版本 |

---

## 联系和反馈

- 有问题？查看 `CUDA_CPU_ROOT_CAUSE.md`
- 需要帮助？运行 `debug_cuda_cpu_diff.py`
- 发现问题？联系开发团队

---

**最后更新：** 2026-05-02  
**状态：** 🟡 待修复  
**优先级：** 高
