# CUDA vs CPU 优化器一致性问题分析

这个目录包含对 NumbaGridOptimizer (CPU) 和 CudaGridOptimizer (GPU) 结果差异的完整调查。

## 文档导航

### 📌 从这里开始

1. **[CUDA_CPU_INVESTIGATION_SUMMARY.md](CUDA_CPU_INVESTIGATION_SUMMARY.md)** ⭐
   - 最终报告，包含所有关键发现
   - 问题概述、影响评估、修复方案
   - 适合：项目负责人、决策者

### 📋 详细分析文档

2. **[CUDA_CPU_ROOT_CAUSE.md](CUDA_CPU_ROOT_CAUSE.md)**
   - 根本原因深度分析
   - 3 个修复方案及工作量评估
   - 适合：开发者、架构师

3. **[CUDA_CPU_ANALYSIS.md](CUDA_CPU_ANALYSIS.md)**
   - 参数传递和计算逻辑对比
   - 关键差异点列表
   - 调试技巧和最佳实践
   - 适合：维护者、调试人员

4. **[VERIFY_CUDA.md](VERIFY_CUDA.md)**
   - 完整的验证流程文档
   - 快速验证步骤
   - 深度验证检查项
   - 可接受的误差范围
   - 适合：测试人员、用户验证

## 相关工具

### 验证和调试脚本

位置：`scripts/`

- **`scripts/debug_cuda_cpu_diff.py`**
  - 对比单个参数组合的 CPU vs CUDA 结果
  - 显示详细的中间计算过程
  - 用法：
    ```bash
    python scripts/debug_cuda_cpu_diff.py \
        --strategy strategies/consecutive_reverse.py \
        --symbol BTCUSDT --interval 1h \
        --start 2024-01-01 --end 2024-02-01
    ```

- **`scripts/verify_cuda_cpu.py`**
  - 从数据库中自动对比优化结果
  - 生成汇总报告
  - 用法：
    ```bash
    python scripts/verify_cuda_cpu.py \
        --strategy strategies/consecutive_reverse.py \
        --symbol BTCUSDT --interval 1h \
        --start 2024-01-01 --end 2024-02-01
    ```

## 问题概述

### 关键数据

| 指标 | CPU Numba | GPU CUDA | 差异 |
|------|-----------|----------|------|
| 交易数 | **31** ❌ | **38** ✓ | 22.6% |
| Sharpe | -2.84 ❌ | +2.49 ✓ | 187.7% |
| 净收益 | -1.13% ❌ | +0.53% ✓ | 146.9% |

### 根本原因

NumbaGridOptimizer 使用了错误的策略实现：

- **期望：** 执行 ConsecutiveReverseStrategy 的 **close+reopen** 逻辑
- **实际：** 执行 **Martingale 加仓** 逻辑
- **结果：** 优化结果完全错误

### 立即建议

**停止使用 `--method numba-grid`**，改用：

```bash
# GPU 用户（推荐）
--method cuda-grid

# CPU 用户
--method grid
```

## 后续行动

### 优先级

| 优先级 | 项目 | 工作量 | 截止日期 |
|--------|------|--------|----------|
| 🔴 高 | 禁用/警告 NumbaGridOptimizer | 1h | 今天 |
| 🔴 高 | 通知用户 | 1h | 今天 |
| 🟠 中 | 实现修复（方案 A） | 2-3h | 本周 |
| 🟡 低 | 架构对齐（方案 B） | 1 天 | 下周 |

## 快速参考

### 问题诊断

如果你在使用 `--method numba-grid` 时得到了异常的优化结果，可能是这个问题。

**症状：**
- 使用 NumbaGridOptimizer 得到负的 Sharpe 比率
- 交易数量异常多或异常少
- 结果与期望完全不符

**解决：**
1. 查看 [CUDA_CPU_INVESTIGATION_SUMMARY.md](CUDA_CPU_INVESTIGATION_SUMMARY.md)
2. 切换到 `--method cuda-grid` 或 `--method grid`
3. 重新运行优化

### 验证方法

```bash
# 快速验证（5分钟）
python scripts/debug_cuda_cpu_diff.py \
    --strategy strategies/consecutive_reverse.py \
    --symbol BTCUSDT --interval 1h \
    --start 2024-01-01 --end 2024-02-01

# 期望看到：
# CPU:  31 交易, Sharpe = -2.84 ❌
# CUDA: 38 交易, Sharpe = +2.49 ✓
```

## 技术详情

### 架构对比

```
NumbaGridOptimizer (❌ 有缺陷)
├── 只支持 1 种策略实现
├── 硬编码 Martingale 逻辑
└── 结果错误

CudaGridOptimizer (✓ 正确)
├── 支持 2 种策略实现
├── 根据策略名称动态选择
└── 结果正确
```

### 代码位置

- **问题源：** `src/backtest/numba_simulate.py` 第 371-384 行
- **期望实现：** `strategies/consecutive_reverse.py` 第 59-62 行
- **正确实现：** `src/backtest/cuda_strategies/consecutive_reverse.py` 第 561-579 行

## 更新历史

| 日期 | 事件 | 状态 |
|------|------|------|
| 2026-05-02 | 问题发现和分析完成 | ✓ 完成 |
| 待定 | 禁用 NummaGridOptimizer | ⏳ 待做 |
| 待定 | 实现修复 | ⏳ 待做 |
| 待定 | 发布修复版本 | ⏳ 待做 |

---

**最后更新：** 2026-05-02  
**联系：** 有问题？查看相关文档或运行调试脚本
