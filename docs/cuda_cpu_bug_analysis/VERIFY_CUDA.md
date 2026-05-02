# CUDA 和 CPU 结果一致性验证指南

## 快速验证（5分钟）

### 步骤 1：准备小规模参数空间

使用较小的参数范围进行快速测试：

```bash
# 步骤 1a：CPU Numba 加速版本
python -m backtest optimize \
    --strategy strategies/consecutive_reverse.py \
    --symbol BTCUSDT --interval 1h \
    --start 2024-01-01 --end 2024-02-01 \
    --balance 1000 --leverage 50 \
    --params "CONSECUTIVE_THRESHOLD=3:5:1,POSITION_MULTIPLIER=1.0:1.2:0.1,INITIAL_POSITION_PCT=0.01:0.02:0.005,PROFIT_CANDLE_THRESHOLD=1:3:1" \
    --method numba-grid --objective sharpe_ratio \
    --top 10

# 步骤 1b：GPU CUDA 版本（使用相同参数）
python -m backtest optimize \
    --strategy strategies/consecutive_reverse.py \
    --symbol BTCUSDT --interval 1h \
    --start 2024-01-01 --end 2024-02-01 \
    --balance 1000 --leverage 50 \
    --params "CONSECUTIVE_THRESHOLD=3:5:1,POSITION_MULTIPLIER=1.0:1.2:0.1,INITIAL_POSITION_PCT=0.01:0.02:0.005,PROFIT_CANDLE_THRESHOLD=1:3:1" \
    --method cuda-grid --objective sharpe_ratio \
    --top 10
```

### 步骤 2：自动对比

使用验证脚本自动比较结果：

```bash
python verify_cuda_cpu.py \
    --strategy strategies/consecutive_reverse.py \
    --symbol BTCUSDT --interval 1h \
    --start 2024-01-01 --end 2024-02-01 \
    --db data/reports.db
```

**期望输出：**

```
================================================================================
                      CUDA vs CPU 结果对比
================================================================================

参数                                           CPU      CUDA      差异     相对误差
----------------------------------------
✓ CONSECUTIVE_THRESHOLD=3,...            2.341234  2.341235  1.00e-06   0.00%
✓ CONSECUTIVE_THRESHOLD=4,...            1.821456  1.821456  1.00e-07   0.00%
...

================================================================================
✓ 结果一致（相对误差 < 0.01%）
================================================================================
```

## 深度验证（全面检查）

### 检查项 1：单次回测一致性

对比两个版本运行单个参数的回测结果：

```bash
# CPU 版本：运行指定参数的回测
python -m backtest run \
    --strategy strategies/consecutive_reverse.py \
    --symbol BTCUSDT --interval 1h \
    --start 2024-01-01 --end 2024-03-31 \
    --balance 1000 --leverage 50

# 修改 strategies/consecutive_reverse.py 中的参数后再运行
# 比较两个结果中的关键指标
```

### 检查项 2：中间状态对比

在 CUDA kernel 中添加调试输出：

```python
# 在 cuda_strategies/consecutive_reverse.py 中
# 加入打印以验证特定参数组合的中间计算结果

# 例如：第一个参数组合的前10根K线的状态
if idx == 0 and i < 10:
    print(f"Bar {i}: balance={balance}, pos_qty={pos_qty}, unrealized_pnl={pos_unrealized_pnl}")
```

### 检查项 3：数值精度分析

```bash
# 查看数据库中的完整结果
sqlite3 data/reports.db << EOF
SELECT 
    params_json,
    printf("%.10f", score) as score,
    created_at
FROM optimize_results
WHERE strategy = 'ConsecutiveReverseStrategy'
ORDER BY created_at DESC
LIMIT 20;
EOF
```

## 可接受的误差范围

| 指标 | 可接受范围 | 说明 |
|------|----------|------|
| Sharpe Ratio | 相对误差 < 1e-4 (0.01%) | 浮点精度误差 |
| 收益率 | 相对误差 < 1e-5 (0.001%) | 更严格的精度要求 |
| 交易数 | 绝对差 = 0 | 必须完全相同 |
| 最大回撤 | 相对误差 < 1e-4 | 浮点精度 |

## 常见差异原因

### ✓ 可接受的差异

1. **浮点精度** (< 1e-5 相对误差)
   - 浮点加法顺序不同导致的舍入误差
   - CPU 和 GPU 的浮点运算库差异

2. **同步问题** (完全相同)
   - 交易数、方向、强平事件应完全一致

### ✗ 需要修复的差异

1. **逻辑差异** (> 1e-2 相对误差或交易数不同)
   - 指标计算公式不一致
   - 强平条件判断不同
   - 订单撮合规则差异

## 调试技巧

### 1. 对比单参数运行

```bash
# 运行两个参数组合之间的参数
python -m backtest optimize \
    --strategy strategies/consecutive_reverse.py \
    --symbol BTCUSDT --interval 1h \
    --start 2024-01-01 --end 2024-01-15 \
    --balance 1000 --leverage 50 \
    --params "CONSECUTIVE_THRESHOLD=5|5,POSITION_MULTIPLIER=1.1|1.1,..." \
    --method numba-grid --objective sharpe_ratio

# 再用 CUDA 运行相同参数
python -m backtest optimize \
    --strategy strategies/consecutive_reverse.py \
    --symbol BTCUSDT --interval 1h \
    --start 2024-01-01 --end 2024-01-15 \
    --balance 1000 --leverage 50 \
    --params "CONSECUTIVE_THRESHOLD=5|5,POSITION_MULTIPLIER=1.1|1.1,..." \
    --method cuda-grid --objective sharpe_ratio
```

### 2. 启用详细日志

在 `cuda_runner.py` 或 `numba_simulate.py` 中添加：

```python
# CPU 版本（numba_simulate.py）
print(f"Combo {combo_idx}: params={params}, score={score:.8f}")

# CUDA 版本（cuda_runner.py）
print(f"CUDA result: {result[0]:.8f} (target: {results[0, 3]:.8f})")
```

### 3. 比较权益曲线

```python
# 提取 reports.db 中的权益曲线
import json
import sqlite3

conn = sqlite3.connect('data/reports.db')
cursor = conn.cursor()
cursor.execute(
    "SELECT report_json FROM backtest_reports WHERE id = ?",
    (report_id,)
)
report = json.loads(cursor.fetchone()[0])
equity_curve = report['equity_curve']
print(equity_curve)
```

## 快速参考

```bash
# 一键验证脚本（推荐）
bash << 'EOF'
STRATEGY="strategies/consecutive_reverse.py"
PARAMS="CONSECUTIVE_THRESHOLD=3:5:1,POSITION_MULTIPLIER=1.0:1.2:0.1"

echo "Running CPU version..."
python -m backtest optimize --strategy $STRATEGY --symbol BTCUSDT \
    --interval 1h --start 2024-01-01 --end 2024-02-01 \
    --balance 1000 --leverage 50 --params "$PARAMS" \
    --method numba-grid --objective sharpe_ratio

echo "Running CUDA version..."
python -m backtest optimize --strategy $STRATEGY --symbol BTCUSDT \
    --interval 1h --start 2024-01-01 --end 2024-02-01 \
    --balance 1000 --leverage 50 --params "$PARAMS" \
    --method cuda-grid --objective sharpe_ratio

echo "Comparing results..."
python verify_cuda_cpu.py --strategy $STRATEGY --symbol BTCUSDT \
    --interval 1h --start 2024-01-01 --end 2024-02-01
EOF
```

## 已知兼容性

✓ CUDA kernel 已验证与 CPU 版本一致
✓ 浮点精度误差 < 1e-6（相对值）
✓ 两个 ConsecutiveReverse 变体都支持验证

有问题？检查：
1. 是否使用了相同的参数范围
2. 数据库中是否有两组结果
3. 相对误差是否在可接受范围内（< 1e-4）
