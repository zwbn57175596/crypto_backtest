#!/usr/bin/env python
"""
深度调试脚本：追踪 CUDA 和 CPU 优化器的差异来源

使用方式:
    python debug_cuda_cpu_diff.py \
        --strategy strategies/consecutive_reverse.py \
        --symbol BTCUSDT --interval 1h \
        --start 2024-01-01 --end 2024-02-01 \
        --balance 1000 --leverage 50
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def load_bars(db_path, symbol, interval, exchange, start_ts, end_ts):
    """Load bars from database."""
    import sqlite3

    conn = sqlite3.connect(db_path)
    query = (
        "SELECT timestamp, open, high, low, close, volume "
        "FROM klines WHERE symbol = ? AND interval = ? AND exchange = ? "
        "AND timestamp >= ? AND timestamp <= ? ORDER BY timestamp ASC"
    )
    cursor = conn.execute(query, (symbol, interval, exchange, start_ts, end_ts))
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return np.empty((0, 6), dtype=np.float64)
    return np.array(rows, dtype=np.float64)


def test_numba_version(args, bars):
    """Test CPU Numba version."""
    from backtest.numba_simulate import simulate
    from backtest.optimizer import load_strategy_param_defaults

    print("\n" + "="*100)
    print("测试参数组合：CPU Numba 版本")
    print("="*100)

    # 硬编码测试参数
    test_params = {
        "CONSECUTIVE_THRESHOLD": 5,
        "POSITION_MULTIPLIER": 1.2,
        "INITIAL_POSITION_PCT": 0.01,
        "PROFIT_CANDLE_THRESHOLD": 3,
        "LEVERAGE": 17,  # 策略杠杆
    }

    default_params = load_strategy_param_defaults(
        args.strategy,
        list(test_params.keys())
    )

    print(f"默认参数: {default_params}")
    print(f"测试参数: {test_params}")

    # 提取参数（与 NumbaGridOptimizer 相同的逻辑）
    threshold = int(test_params.get("CONSECUTIVE_THRESHOLD", default_params.get("CONSECUTIVE_THRESHOLD", 5)))
    multiplier = float(test_params.get("POSITION_MULTIPLIER", default_params.get("POSITION_MULTIPLIER", 1.1)))
    initial_pct = float(test_params.get("INITIAL_POSITION_PCT", default_params.get("INITIAL_POSITION_PCT", 0.01)))
    profit_threshold = int(test_params.get("PROFIT_CANDLE_THRESHOLD", default_params.get("PROFIT_CANDLE_THRESHOLD", 1)))
    sizing_lev = int(test_params.get("LEVERAGE", default_params.get("LEVERAGE", args.leverage)))

    print(f"\n提取的参数:")
    print(f"  threshold={threshold}")
    print(f"  multiplier={multiplier}")
    print(f"  initial_pct={initial_pct}")
    print(f"  profit_threshold={profit_threshold}")
    print(f"  sizing_leverage={sizing_lev}")
    print(f"  exchange_leverage={args.leverage}")

    # 调用 simulate
    result = simulate(
        bars,
        threshold=threshold,
        multiplier=multiplier,
        initial_pct=initial_pct,
        profit_threshold=profit_threshold,
        sizing_leverage=sizing_lev,
        exchange_leverage=args.leverage,
        commission_rate=0.0004,
        funding_rate=0.0001,
        maintenance_margin=0.005,
        initial_balance=args.balance,
    )

    metrics = {
        "net_return": result[0],
        "annual_return": result[1],
        "max_drawdown": result[2],
        "sharpe_ratio": result[3],
        "sortino_ratio": result[4],
        "win_rate": result[5],
        "profit_factor": result[6],
        "total_trades": int(result[7]),
    }

    print(f"\nNumba 结果:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.8f}" if isinstance(v, float) else f"  {k}: {v}")

    return metrics


def test_cuda_version(args, bars):
    """Test GPU CUDA version."""
    try:
        from numba import cuda
        import numpy as np
    except ImportError:
        print("❌ CUDA 不可用")
        return None

    try:
        if not cuda.is_available():
            print("❌ CUDA 运行时不可用")
            return None
    except Exception as exc:
        print(f"❌ CUDA 错误: {exc}")
        return None

    from backtest.cuda_strategies import CUDA_STRATEGIES
    from backtest.optimizer import load_strategy_param_defaults

    print("\n" + "="*100)
    print("测试参数组合：GPU CUDA 版本")
    print("="*100)

    # 硬编码测试参数（与 Numba 版本相同）
    test_params = {
        "CONSECUTIVE_THRESHOLD": 5,
        "POSITION_MULTIPLIER": 1.2,
        "INITIAL_POSITION_PCT": 0.01,
        "PROFIT_CANDLE_THRESHOLD": 3,
        "LEVERAGE": 17,
    }

    # 获取 CUDA kernel
    strategy_name = "ConsecutiveReverseStrategy"
    if strategy_name not in CUDA_STRATEGIES:
        print(f"❌ 策略 '{strategy_name}' 未在 CUDA registry 中")
        return None

    registry_entry = CUDA_STRATEGIES[strategy_name]
    kernel = registry_entry["kernel"]
    param_order = registry_entry["param_order"]

    default_params = load_strategy_param_defaults(args.strategy, param_order)

    print(f"默认参数: {default_params}")
    print(f"测试参数: {test_params}")
    print(f"参数顺序: {param_order}")

    # 构建 params 数组（与 CudaGridOptimizer 相同的逻辑）
    param_row = []
    for param_name in param_order:
        value = test_params.get(param_name, default_params.get(param_name, 1))
        param_row.append(float(value))
        print(f"  {param_name} = {value}")

    params_array = np.array([param_row], dtype=np.float64)

    print(f"\nparams_array[0] = {params_array[0]}")
    print(f"  [0] threshold = {params_array[0, 0]}")
    print(f"  [1] multiplier = {params_array[0, 1]}")
    print(f"  [2] initial_pct = {params_array[0, 2]}")
    print(f"  [3] profit_threshold = {params_array[0, 3]}")
    print(f"  [4] sizing_leverage = {params_array[0, 4]}")

    # 准备 GPU 数据
    bars_gpu = cuda.to_device(bars.astype(np.float64))
    params_gpu = cuda.to_device(params_array)
    results_gpu = cuda.device_array((1, 8), dtype=np.float64)

    # 启动 kernel
    kernel[1, 1](
        bars_gpu,
        params_gpu,
        results_gpu,
        bars.shape[0],
        1,  # n_combos=1
        args.leverage,  # exchange_leverage
        0.0004,  # commission_rate
        0.0001,  # funding_rate
        0.005,  # maintenance_margin
        args.balance,  # initial_balance
    )

    cuda.synchronize()

    # 复制结果
    result = results_gpu.copy_to_host()[0]

    metrics = {
        "net_return": float(result[0]),
        "annual_return": float(result[1]),
        "max_drawdown": float(result[2]),
        "sharpe_ratio": float(result[3]),
        "sortino_ratio": float(result[4]),
        "win_rate": float(result[5]),
        "profit_factor": float(result[6]),
        "total_trades": int(result[7]),
    }

    print(f"\nCUDA 结果:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.8f}" if isinstance(v, float) else f"  {k}: {v}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="深度调试 CUDA vs CPU 优化器差异")
    parser.add_argument("--strategy", required=True, help="策略文件路径")
    parser.add_argument("--symbol", default="BTCUSDT", help="交易对")
    parser.add_argument("--interval", default="1h", help="K线周期")
    parser.add_argument("--start", default="2024-01-01", help="开始日期")
    parser.add_argument("--end", default="2024-02-01", help="结束日期")
    parser.add_argument("--balance", type=float, default=1000.0, help="初始资金")
    parser.add_argument("--leverage", type=int, default=50, help="交易所杠杆")
    parser.add_argument("--db", default="data/klines.db", help="数据库路径")

    args = parser.parse_args()

    # 加载数据
    start_ts = int(
        datetime.strptime(f"{args.start} 00:00:00", "%Y-%m-%d %H:%M:%S")
        .replace(tzinfo=timezone.utc)
        .timestamp() * 1000
    )
    end_ts = int(
        datetime.strptime(f"{args.end} 23:59:59", "%Y-%m-%d %H:%M:%S")
        .replace(tzinfo=timezone.utc)
        .timestamp() * 1000
    )

    bars = load_bars(args.db, args.symbol, args.interval, "binance", start_ts, end_ts)
    print(f"加载 {bars.shape[0]} 根 K 线")

    if bars.shape[0] == 0:
        print("❌ 没有数据")
        return

    # 测试 Numba 版本
    numba_metrics = test_numba_version(args, bars)

    # 测试 CUDA 版本
    cuda_metrics = test_cuda_version(args, bars)

    # 对比
    if numba_metrics and cuda_metrics:
        print("\n" + "="*100)
        print("对比结果")
        print("="*100)
        print(f"\n{'指标':<20} {'CPU Numba':>15} {'GPU CUDA':>15} {'差异':>15} {'相对误差':>15}")
        print("-" * 80)

        for key in numba_metrics:
            cpu_val = numba_metrics[key]
            cuda_val = cuda_metrics[key]

            if isinstance(cpu_val, float):
                diff = abs(cpu_val - cuda_val)
                rel_error = diff / max(abs(cpu_val), abs(cuda_val), 1e-10) if max(abs(cpu_val), abs(cuda_val)) > 0 else 0

                print(
                    f"{key:<20} {cpu_val:>15.8f} {cuda_val:>15.8f} {diff:>15.2e} {rel_error*100:>14.2f}%"
                )
            else:
                print(f"{key:<20} {cpu_val:>15.0f} {cuda_val:>15.0f} {abs(cpu_val - cuda_val):>15.0f}")


if __name__ == "__main__":
    main()
