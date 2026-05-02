# 🚨 CUDA vs CPU 优化器严重问题报告

**发现日期：** 2026-05-02  
**严重程度：** 🔴 高  
**状态：** 待修复

## ⚡ 快速摘要

NumbaGridOptimizer (CPU) 执行了**错误的策略逻辑**，导致优化结果完全不可靠。

```
CPU (Numba):     31 交易, Sharpe = -2.84 ❌
GPU (CUDA):      38 交易, Sharpe = +2.49 ✓
期望 (Close+reopen): 38 交易（CUDA 正确）
```

**立即行动：** 停止使用 `--method numba-grid`，改用 `--method cuda-grid` 或 `--method grid`

## 📖 完整文档

所有分析文档已存放在 `docs/cuda_cpu_bug_analysis/` 目录下：

### 📌 从这里开始
- **[docs/cuda_cpu_bug_analysis/README.md](docs/cuda_cpu_bug_analysis/README.md)** — 导航指南
- **[docs/cuda_cpu_bug_analysis/CUDA_CPU_INVESTIGATION_SUMMARY.md](docs/cuda_cpu_bug_analysis/CUDA_CPU_INVESTIGATION_SUMMARY.md)** — 最终报告 ⭐

### 📋 详细分析
- **[docs/cuda_cpu_bug_analysis/CUDA_CPU_ROOT_CAUSE.md](docs/cuda_cpu_bug_analysis/CUDA_CPU_ROOT_CAUSE.md)** — 根本原因 + 3 个修复方案
- **[docs/cuda_cpu_bug_analysis/CUDA_CPU_ANALYSIS.md](docs/cuda_cpu_bug_analysis/CUDA_CPU_ANALYSIS.md)** — 参数对比分析
- **[docs/cuda_cpu_bug_analysis/VERIFY_CUDA.md](docs/cuda_cpu_bug_analysis/VERIFY_CUDA.md)** — 验证流程

## 🛠️ 调试工具

位置：`scripts/`

```bash
# 快速对比 CPU vs CUDA
python scripts/debug_cuda_cpu_diff.py \
    --strategy strategies/consecutive_reverse.py \
    --symbol BTCUSDT --interval 1h \
    --start 2024-01-01 --end 2024-02-01

# 自动对比优化结果
python scripts/verify_cuda_cpu.py \
    --strategy strategies/consecutive_reverse.py \
    --symbol BTCUSDT --interval 1h \
    --start 2024-01-01 --end 2024-02-01
```

## 🔍 问题原因

NumbaGridOptimizer 在处理亏损 K 线时执行了 **Martingale 加仓** 而非 **平仓-重开** 逻辑：

```python
# ❌ NumbaGridOptimizer 实际执行 (numba_simulate.py 第 371-384)
else:  # Loss candle
    add_qty = target_qty - pos_qty
    pend_side_0 = _BUY  # 加仓！

# ✓ 应该执行 (strategies/consecutive_reverse.py)
else:  # Loss candle
    self.close()        # 平仓
    self._try_open()    # 重开
```

## ⚠️ 受影响的用户

```bash
# ❌ 受影响（结果错误）
python -m backtest optimize --strategy strategies/consecutive_reverse.py \
    --method numba-grid

# ✓ 不受影响
python -m backtest optimize --strategy strategies/consecutive_reverse.py \
    --method cuda-grid  # CUDA 正确
python -m backtest optimize --strategy strategies/consecutive_reverse.py \
    --method grid       # 基础 CPU 网格搜索，正确
```

## 🎯 立即建议

### 现在就做

1. **切换优化方法**
   ```bash
   # GPU 用户（推荐，1000x+ 快速）
   --method cuda-grid
   
   # CPU 用户（正确但较慢）
   --method grid
   ```

2. **验证之前的结果**
   - 如果你曾使用 `--method numba-grid` 获得了优化参数，**不要在实盘中使用**
   - 重新使用正确的方法进行优化

### 本周计划

- [ ] 禁用 NumbaGridOptimizer（添加警告）
- [ ] 通知所有用户
- [ ] 实现修复（分离 simulate 函数为两个版本）
- [ ] 完整测试

## 📊 对比数据

完整的验证数据见 [CUDA_CPU_INVESTIGATION_SUMMARY.md](docs/cuda_cpu_bug_analysis/CUDA_CPU_INVESTIGATION_SUMMARY.md)

| 指标 | CPU | GPU | 相对误差 |
|------|-----|-----|---------|
| 交易数 | 31 ❌ | 38 ✓ | 22.6% |
| Sharpe | -2.84 ❌ | +2.49 ✓ | 187.7% |
| 净收益 | -1.13% ❌ | +0.53% ✓ | 146.9% |
| 胜率 | 69.2% ❌ | 47.4% ✓ | 31.6% |

## 🔗 相关文件

```
docs/cuda_cpu_bug_analysis/
├── README.md                              # 导航指南
├── CUDA_CPU_INVESTIGATION_SUMMARY.md      # 最终报告 ⭐
├── CUDA_CPU_ROOT_CAUSE.md                 # 根本原因 + 修复方案
├── CUDA_CPU_ANALYSIS.md                   # 详细参数分析
└── VERIFY_CUDA.md                         # 验证流程

scripts/
├── debug_cuda_cpu_diff.py                 # CPU vs CUDA 对比脚本
└── verify_cuda_cpu.py                     # 优化结果对比脚本

根目录:
└── CUDA_CPU_BUG_REPORT.md                 # 本文件（快速参考）
```

---

**详细信息请查看：** [docs/cuda_cpu_bug_analysis/README.md](docs/cuda_cpu_bug_analysis/README.md)

**问题发现者：** Claude Code  
**最后更新：** 2026-05-02
