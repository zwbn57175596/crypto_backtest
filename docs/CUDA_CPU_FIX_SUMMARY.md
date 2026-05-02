# CUDA vs CPU 差异修复总结

## 问题根源

CUDA核心 (`cuda_strategies/consecutive_reverse.py`) 在处理亏损K线时实现了**错误的逻辑**：

### ❌ 之前（错误）- Martingale 加仓逻辑
```python
else:  # Loss candle
    target_qty = device_calc_quantity(...)
    add_qty = target_qty - pos_qty
    if add_qty > 0:
        # 增加持仓 (加仓)
        pend_side_0 = BUY (if SHORT) or SELL (if LONG)
```

### ✓ 修复后（正确）- Close + Reopen 逻辑
```python
else:  # Loss candle - close immediately + try reopen
    # 第一步：平仓
    if pos_side == LONG:
        pend_side_0 = SELL
    else:
        pend_side_0 = BUY
    pend_qty_0 = pos_qty
    
    # 第二步：尝试反向重开
    reopen_qty = device_calc_quantity(...)
    if reopen_qty > 0:
        if direction == 1:
            pend_side_1 = SELL
        else:
            pend_side_1 = BUY
```

## 为什么会有差异

- **CPU** (`numba_simulate.py::simulate_close_reopen`): 正确实现close+reopen
- **CUDA** (`cuda_strategies/consecutive_reverse.py`): 错误实现为martingale加仓
- 结果: 两种方法生成不同的交易序列和最终指标

## 验证结果

修复前：
```
CPU 结果数: 4793
CUDA 结果数: 4793  
共同参数组合: 1083 (22.6% 匹配)
最大差异: 1.59
```

修复后：
```
CPU 结果数: 1000
CUDA 结果数: 1000
共同参数组合: 1000 (100% 匹配) ✓
最大差异: 8.68e-14 (浮点数舍入误差)
```

## 影响范围

**受影响的代码**:
- `src/backtest/cuda_strategies/consecutive_reverse.py` (修复)

**未受影响**:
- CPU optimizer (numba-grid, grid) - 一直正确
- 其他策略内核 - 需要逐个验证

## 下一步

- [ ] 验证 ConsecutiveReverseMartingaleStrategy CUDA 核心
- [ ] 验证其他策略（如果有）
- [ ] 更新文档

---

**提交**: 46d7f5c
**修复日期**: 2026-05-03
