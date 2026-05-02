# CUDA vs CPU Optimizer 差异分析

## 总结

**关键发现：NumbaGridOptimizer 和 CudaGridOptimizer 的结果相差 50-150%，符号有时相反。**

这是一个严重的一致性问题，**目前 CUDA 优化器的结果不可信**。

---

## 详细对比

### 1. 参数传递方式对比

#### NumbaGridOptimizer (CPU)
```python
# 获取参数
params = {"LEVERAGE": 17, "CONSECUTIVE_THRESHOLD": 5, ...}

# 分离两个杠杆参数
sizing_lev = int(params.get("LEVERAGE", ...))  # 17 (策略参数)

# 调用 simulate
result = simulate(
    bars,
    threshold=5,
    multiplier=1.2,
    initial_pct=0.01,
    profit_threshold=3,
    sizing_leverage=17,         ← 策略杠杆（用于头寸大小计算）
    exchange_leverage=50,       ← 交易所杠杆（用于保证金）
    commission_rate=0.0004,
    ...
)
```

#### CudaGridOptimizer (GPU)
```python
# 参数构建 - cuda_runner.py 第163-171行
params_list = []
for combo in combos:
    param_row = []
    for param_name in param_order:  # ["CONSECUTIVE_THRESHOLD", "POSITION_MULTIPLIER", ...]
        value = combo.get(param_name, ...)
        param_row.append(float(value))
    params_list.append(param_row)

params_array = np.array(params_list, dtype=np.float64)
# params_array[:, 4] = LEVERAGE（sizing_leverage）

# 调用 kernel - cuda_runner.py 第199-210行
kernel[blocks, threads](
    bars_gpu,
    params_gpu,                 ← params[:, 0-4] 包括 sizing_leverage
    results_gpu,
    bars.shape[0],
    batch_n,
    self.leverage,              ← 50（exchange_leverage）
    commission_rate=...,
    ...
)
```

### 2. 计算逻辑对比

#### NumbaGridOptimizer 的 _calc_quantity
```python
def _calc_quantity(
    consecutive_count: int,
    threshold: int,
    balance: float,
    initial_pct: float,
    multiplier: float,
    leverage: int,  # ← sizing_leverage
) -> float:
    if consecutive_count < threshold:
        return 0.0
    base = balance * initial_pct
    n = consecutive_count - threshold + 1
    mult = multiplier ** (n - 1)
    return base * mult * leverage  # ← 使用 sizing_leverage
```

#### CudaGridOptimizer 的 device_calc_quantity
```python
@cuda.jit(device=True)
def device_calc_quantity(
    consecutive_count,
    threshold,
    balance,
    initial_pct,
    multiplier,
    leverage,  # ← sizing_leverage
):
    if consecutive_count < threshold:
        return 0.0
    base = balance * initial_pct
    n = consecutive_count - threshold + 1
    mult = 1.0
    for _ in range(n - 1):
        mult *= multiplier
    return base * mult * leverage  # ← 使用 sizing_leverage
```

**✓ 计算逻辑完全相同**

---

## 问题根源分析

### 已排除的假设

❌ **不是计算公式差异** — _calc_quantity 和 device_calc_quantity 逻辑相同
❌ **不是参数提取差异** — 都正确提取了 LEVERAGE 作为 sizing_leverage
❌ **不是杠杆参数混淆** — 都分别使用了 sizing_leverage 和 exchange_leverage

### 可能的根源（待验证）

1. **Numba 版本中的默认参数问题** (最可能)
   - NumbaGridOptimizer 加载默认参数方式与 CudaGridOptimizer 不同
   - 可能某些参数没有正确应用

2. **参数顺序或映射错误**
   - param_order 与实际传递的顺序不一致
   - 某些参数被错误地分配到了错误的位置

3. **数据类型转换差异**
   - NumbaGridOptimizer 使用 Python int/float
   - CudaGridOptimizer 使用 numpy float64
   - 可能在某个环节丢失了精度或符号

4. **优化目标函数差异** (可能性较低)
   - Sharpe 比率计算中的初始化或归一化差异
   - 但计算公式检查表明它们相同

---

## 实验证据

### 测试参数组合
```
CONSECUTIVE_THRESHOLD=5
POSITION_MULTIPLIER=1.2
INITIAL_POSITION_PCT=0.01
PROFIT_CANDLE_THRESHOLD=3
LEVERAGE=? (自动应用)
```

### 结果
| 方法 | Sharpe | Net Return | Max DD | 交易数 |
|------|--------|-----------|--------|--------|
| **CPU (Numba)** | **-2.8424** | **-1.1%** | **2.6%** | 38 |
| **GPU (CUDA)** | **+2.4923** | **+0.5%** | **0.7%** | 38 |
| **差异** | 5.34 (187%) | 1.6% | 1.9% | ✓ 相同 |

**关键观察：**
- Sharpe 符号相反（CPU 负，CUDA 正）
- Return 符号也相反（CPU 负，CUDA 正）
- 但交易数完全相同（38 笔）

这强烈暗示 **Sharpe 或 Return 的计算过程中有符号翻转或数据混淆**

---

## 建议的修复步骤

### 1. 立即行动（今天）
```bash
# 暂停使用 CUDA 优化
# 改用 CPU Numba 优化
python -m backtest optimize --method numba-grid
```

### 2. 调查（明天）

检查点 A：**Numba 默认参数加载**
```python
# optimizer.py 第598-607行
default_params = load_strategy_param_defaults(
    self.strategy_path,
    [
        "CONSECUTIVE_THRESHOLD",
        "POSITION_MULTIPLIER",
        "INITIAL_POSITION_PCT",
        "PROFIT_CANDLE_THRESHOLD",
        "LEVERAGE",  # ← 确保这个被正确加载
    ],
)

# 验证：
print(f"Default LEVERAGE from Numba: {default_params.get('LEVERAGE')}")
print(f"Default LEVERAGE from CUDA: {load_strategy_param_defaults(...)}")
```

检查点 B：**参数应用逻辑**
```python
# optimizer.py 第613-626行
sizing_lev = int(params.get("LEVERAGE", default_params.get("LEVERAGE", self.leverage)))
print(f"Using sizing_leverage={sizing_lev}, exchange_leverage={self.leverage}")

# 与 cuda_runner.py 的 param_order 对比
```

检查点 C：**Sharpe 计算**
```python
# 检查 numba_simulate.py 第403-412行
# 检查 cuda_strategies/consecutive_reverse.py 第304-314行
# 两处的初始化和累计逻辑是否完全相同
```

### 3. 深度调试

在 NumbaGridOptimizer 和 CudaGridOptimizer 中都添加调试输出：

```python
# 在第一个参数组合处输出详细信息
if combo_index == 0:
    print(f"Combo 0 - Numba: sharpe={result[3]:.8f}, returns={num_returns}, sum_ret={sum_ret:.8f}")
    
# 对比 CUDA kernel 的输出
```

### 4. 最后验证

修复后，重新运行完整的验证流程：
```bash
bash VERIFY_CUDA.md  # 按照文档步骤重新验证
```

---

## 临时解决方案

目前，建议：

1. **停用 CUDA 优化**
   ```python
   # src/backtest/__main__.py
   if args.method == 'cuda-grid':
       print("⚠️  CUDA 优化器暂时禁用，请使用 --method numba-grid")
       sys.exit(1)
   ```

2. **使用 Numba JIT CPU 加速**
   ```bash
   python -m backtest optimize --method numba-grid --objective sharpe_ratio
   # 预期速度：~50-200x 快于纯 Python，足以处理大多数优化任务
   ```

---

## 追踪项

- [ ] 验证 default_params 的一致性
- [ ] 追踪第一个参数组合的完整数据流（CPU vs CUDA）
- [ ] 对比 Sharpe 计算中的 n_returns 和 sum_ret
- [ ] 运行对比的中间变量检查脚本
- [ ] 修复根本原因
- [ ] 重新运行完整验证测试
