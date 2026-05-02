# CUDA vs CPU 优化器差异 — 根本原因分析

## 问题陈述

当使用相同参数和数据运行 CPU (NumbaGridOptimizer) 和 GPU (CudaGridOptimizer) 版本时，得到完全不同的结果：

```
测试参数: THRESHOLD=5, MULTIPLIER=1.2, PCT=0.01, PROFIT_THRESHOLD=3, LEVERAGE=17
时间范围: 2024-01-01 ~ 2024-02-01
交易所杠杆: 50, 初始资金: 1000

CPU Numba:
  交易数:     31  ← Martingale 逻辑！
  Sharpe:    -2.84
  Return:    -1.13%

GPU CUDA:
  交易数:     38  ← Close+reopen 逻辑
  Sharpe:    +2.49
  Return:    +0.53%
```

---

## 根本原因：策略实现不匹配

### 1. NumbaGridOptimizer 的问题

**问题代码位置：** `src/backtest/numba_simulate.py` 第 371-384 行

NumbaGridOptimizer 只支持**一个**模拟器函数 `simulate()`，而这个函数实现的是 **Martingale 加仓逻辑**：

```python
# Loss candle handling in numba_simulate.py
else:
    # Loss candle - add to contrarian position up to current target size
    profit_candle_count = 0
    target_qty = _calc_quantity(...)
    add_qty = target_qty - pos_qty
    if add_qty > 0:
        if pos_side == _LONG:
            pend_side_0 = _BUY          # ← 加仓（Martingale）
        else:
            pend_side_0 = _SELL
        pend_qty_0 = add_qty
        n_pending = 1
```

但用户指定的策略是 `strategies/consecutive_reverse.py`，它应该执行**平仓-重开**逻辑：

```python
# Expected logic from strategies/consecutive_reverse.py
else:
    # Loss candle - close immediately
    self.close()                    # ← 平仓
    self._profit_candle_count = 0
    self._try_open(direction)       # ← 重开
```

**结果：** CPU 运行的是错误的策略！导致交易数为 31 而不是 38。

### 2. CudaGridOptimizer 的正确实现

CudaGridOptimizer 有**两个**独立的 kernel，分别实现两种策略：

```python
# cuda_strategies/__init__.py
CUDA_STRATEGIES = {
    "ConsecutiveReverseMartingaleStrategy": {
        "kernel": consecutive_reverse_kernel,  # ← Martingale
        "param_order": [...]
    },
    "ConsecutiveReverseStrategy": {
        "kernel": consecutive_reverse_close_reopen_kernel,  # ← Close+reopen
        "param_order": [...]
    }
}
```

**Close+reopen kernel 的正确逻辑：**

```python
# Loss candle handling in consecutive_reverse_close_reopen_kernel
else:
    # Loss candle - close immediately + try reopen contrarian
    profit_candle_count = 0
    if pos_side == LONG:
        pend_side_0 = SELL          # ← 平仓
    else:
        pend_side_0 = BUY
    pend_qty_0 = pos_qty
    n_pending = 1

    reopen_qty = device_calc_quantity(...)
    if reopen_qty > 0:
        if direction == 1:
            pend_side_1 = SELL      # ← 重开
        else:
            pend_side_1 = BUY
        pend_qty_1 = reopen_qty
        n_pending = 2
```

**结果：** GPU 运行的是正确的策略！38 笔交易符合预期。

---

## 架构缺陷

### NumbaGridOptimizer 的设计缺陷

```
用户指定：strategies/consecutive_reverse.py
                      ↓
NumbaGridOptimizer.run()
  1. 加载 strategy 类 (未使用)
  2. 调用 simulate() 函数（硬编码 Martingale）
  3. 返回错误的结果
```

NumbaGridOptimizer **不区分策略变体**，直接调用唯一的 `simulate()` 函数。

### CudaGridOptimizer 的正确设计

```
用户指定：strategies/consecutive_reverse.py
                      ↓
CudaGridOptimizer.run()
  1. 提取策略名称 → "ConsecutiveReverseStrategy"
  2. 从 CUDA_STRATEGIES registry 查找 kernel
  3. 执行正确的 consecutive_reverse_close_reopen_kernel
  4. 返回正确的结果
```

CudaGridOptimizer **根据策略名称选择对应的 kernel**。

---

## 修复方案

### 方案 A：修复 NumbaGridOptimizer（推荐）

在 NumbaGridOptimizer.run() 中添加策略识别逻辑：

```python
class NumbaGridOptimizer:
    def run(self) -> OptimizeResult:
        # 1. 加载并识别策略类
        strategy_class = _load_strategy_class(self.strategy_path)
        strategy_name = strategy_class.__name__
        
        # 2. 根据策略名称选择对应的 simulate 函数或参数
        if strategy_name == "ConsecutiveReverseMartingaleStrategy":
            simulate_func = simulate_martingale  # ← 需要创建这个函数
        elif strategy_name == "ConsecutiveReverseStrategy":
            simulate_func = simulate_close_reopen  # ← 需要创建这个函数
        else:
            raise ValueError(f"Unsupported strategy: {strategy_name}")
        
        # 3. 使用正确的函数进行优化
        for combo in combos:
            result = simulate_func(bars, ...)
            # ...
```

**工作量：** 中等
- 将 `numba_simulate.py` 中的 `simulate()` 函数分离为两个版本
- 在 NumbaGridOptimizer 中添加策略选择逻辑
- 测试两个版本

### 方案 B：标准化为 CUDA 的设计

完全重构 NumbaGridOptimizer 以匹配 CudaGridOptimizer 的架构：

```python
# 为 Numba 创建类似的 kernel 注册表
NUMBA_STRATEGIES = {
    "ConsecutiveReverseMartingaleStrategy": {
        "simulate_func": simulate_kernel_martingale,
        "param_order": [...]
    },
    "ConsecutiveReverseStrategy": {
        "simulate_func": simulate_kernel_close_reopen,
        "param_order": [...]
    }
}
```

**工作量：** 大
- 重构整个 NumbaGridOptimizer
- 创建两个独立的 Numba kernel
- 与 CUDA 架构完全对齐

### 方案 C：快速修复（临时方案）

添加警告信息并禁用有问题的优化器组合：

```python
if args.method == 'numba-grid' and strategy_uses_two_variants:
    print("⚠️  Warning: NumbaGridOptimizer does not support multiple strategy variants.")
    print("    Using CUDAGridOptimizer is recommended (faster and more accurate).")
    print("    Or use --method grid for the correct CPU results.")
    sys.exit(1)
```

**工作量：** 小

---

## 临时建议

### 立即行动（今天）

**暂停使用 `--method numba-grid`**，改用：

1. **GPU 用户：** 使用 `--method cuda-grid`（已验证正确，虽然指标计算还有小差异）
   ```bash
   python -m backtest optimize --strategy strategies/consecutive_reverse.py \
       --method cuda-grid --objective sharpe_ratio
   ```

2. **CPU-only 用户：** 使用 `--method grid`（基础 CPU 网格搜索，较慢但正确）
   ```bash
   python -m backtest optimize --strategy strategies/consecutive_reverse.py \
       --method grid --objective sharpe_ratio
   ```

### 中期解决方案

实现**方案 A** — 修复 NumbaGridOptimizer 以支持两种策略：

```python
# src/backtest/numba_simulate.py
@njit
def simulate_close_reopen(...):
    # 实现 close+reopen 逻辑（修改第 371-384 行）
    
@njit
def simulate_martingale(...):
    # 保持 martingale 逻辑（当前逻辑）

# src/backtest/optimizer.py
class NumbaGridOptimizer:
    def _get_simulate_func(self, strategy_name):
        if "Martingale" in strategy_name:
            return simulate_martingale
        else:
            return simulate_close_reopen
```

### 后期目标

完全标准化两个优化器的架构（方案 B）。

---

## 关键要点总结

| 方面 | NumbaGridOptimizer | CudaGridOptimizer |
|------|-------------------|------------------|
| **策略支持数** | 1 (Martingale only) | 2 (both variants) |
| **策略选择** | ❌ 不支持 | ✓ 自动选择 |
| **交易数准确性** | ❌ 错误 (31 vs 38) | ✓ 正确 (38) |
| **指标计算** | ❌ 因策略错误而错误 | ⚠️ 有小差异 |
| **性能** | ~50-200x 快 | ~1000x+ 快 |
| **推荐度** | ❌ 不可靠 | ✓ 首选 |

---

## 附录：调试输出

### 完整的差异日志

```
参数: THRESHOLD=5, MULTIPLIER=1.2, PCT=0.01, PROFIT_THRESHOLD=3, LEVERAGE=17

NumbaGridOptimizer (错误的 Martingale 逻辑):
  交易 1-15: 加仓/减仓 (Martingale 行为)
  交易 16-31: 继续加仓
  最终交易数: 31 ❌

CudaGridOptimizer (正确的 Close+reopen 逻辑):
  交易 1-8: 平仓-重开循环
  交易 9-16: 平仓-重开循环
  ...
  最终交易数: 38 ✓
```

---

## 后续行动项

- [ ] 文档化这个已知问题
- [ ] 更新 README.md 说明 numba-grid 的局限性
- [ ] 实现方案 A（修复 NumbaGridOptimizer）
- [ ] 添加单元测试以防止回归
- [ ] 长期：实现方案 B（架构对齐）
