#!/usr/bin/env python
"""
验证 CUDA 和 CPU 优化结果的一致性。
比较相同参数组合在两个方法中的回测指标。

使用方式:
    python verify_cuda_cpu.py \
        --strategy strategies/consecutive_reverse.py \
        --symbol BTCUSDT --interval 1h \
        --start 2024-01-01 --end 2024-03-31 \
        --balance 1000 --leverage 50 \
        --params "CONSECUTIVE_THRESHOLD=3:5:1,POSITION_MULTIPLIER=1.0:1.2:0.1"
"""

import sqlite3
import json
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class OptimizeResult:
    """单个优化结果"""
    strategy: str
    symbol: str
    interval: str
    params_json: str
    score: float
    report_json: str
    method: str

    def get_params_dict(self) -> dict:
        return json.loads(self.params_json)

    def get_report_dict(self) -> dict:
        return json.loads(self.report_json)


def get_results_from_db(
    db_path: str,
    strategy: str,
    symbol: str,
    interval: str,
    start_date: str,
    end_date: str,
) -> dict:
    """从数据库中读取优化结果"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    query = """
    SELECT strategy, symbol, interval, params_json, score, report_json, created_at
    FROM optimize_results
    WHERE strategy = ? AND symbol = ? AND interval = ?
        AND DATE(created_at) BETWEEN ? AND ?
    ORDER BY created_at DESC
    """

    cursor.execute(query, (strategy, symbol, interval, start_date, end_date))
    rows = cursor.fetchall()
    conn.close()

    return {
        "count": len(rows),
        "results": [
            {
                "strategy": r[0],
                "symbol": r[1],
                "interval": r[2],
                "params": json.loads(r[3]),
                "score": r[4],
                "report": json.loads(r[5]),
                "created_at": r[6],
            }
            for r in rows
        ]
    }


def compare_results(cpu_results: list, cuda_results: list) -> None:
    """比较 CPU 和 CUDA 结果"""
    print("\n" + "="*80)
    print("CUDA vs CPU 结果对比".center(80))
    print("="*80)

    # 按参数分组
    cpu_by_params = {json.dumps(r["params"], sort_keys=True): r for r in cpu_results}
    cuda_by_params = {json.dumps(r["params"], sort_keys=True): r for r in cuda_results}

    all_params = set(cpu_by_params.keys()) | set(cuda_by_params.keys())

    if not all_params:
        print("没有找到共同的参数组合")
        return

    max_diff = 0.0
    max_diff_param = None

    print(f"\n{'参数':<60} {'CPU':>12} {'CUDA':>12} {'差异':>10} {'相对误差':>10}")
    print("-" * 110)

    for params_str in sorted(all_params):
        cpu_r = cpu_by_params.get(params_str)
        cuda_r = cuda_by_params.get(params_str)

        if cpu_r and cuda_r:
            cpu_score = cpu_r["score"]
            cuda_score = cuda_r["score"]
            diff = abs(cpu_score - cuda_score)
            rel_error = diff / max(abs(cpu_score), abs(cuda_score), 1e-10) if max(abs(cpu_score), abs(cuda_score)) > 0 else 0

            # 解析参数
            params = json.loads(params_str)
            param_str = ",".join(f"{k}={v}" for k, v in sorted(params.items()))
            param_str = param_str[:58] + "..." if len(param_str) > 60 else param_str

            status = "✓" if rel_error < 1e-4 else "⚠" if rel_error < 1e-2 else "✗"

            print(
                f"{status} {param_str:<58} {cpu_score:>12.6f} {cuda_score:>12.6f} "
                f"{diff:>10.2e} {rel_error*100:>9.2f}%"
            )

            if diff > max_diff:
                max_diff = diff
                max_diff_param = (params, cpu_score, cuda_score)
        elif cpu_r:
            print(f"✗ {params_str[:58]} (仅在 CPU 中)")
        else:
            print(f"✗ {params_str[:58]} (仅在 CUDA 中)")

    print("-" * 110)
    print(f"\n统计：")
    print(f"  CPU 结果数: {len(cpu_results)}")
    print(f"  CUDA 结果数: {len(cuda_results)}")
    print(f"  共同参数组合: {len([p for p in all_params if p in cpu_by_params and p in cuda_by_params])}")
    print(f"  最大差异: {max_diff:.2e}")

    if max_diff_param:
        params, cpu_score, cuda_score = max_diff_param
        print(f"\n最大差异的参数组合:")
        for k, v in sorted(params.items()):
            print(f"  {k} = {v}")
        print(f"  CPU:  {cpu_score:.8f}")
        print(f"  CUDA: {cuda_score:.8f}")

    # 判断一致性
    print("\n" + "="*80)
    if all(
        abs(cpu_by_params.get(p, {}).get("score", 0) - cuda_by_params.get(p, {}).get("score", 0))
        / max(abs(cpu_by_params.get(p, {}).get("score", 0)), abs(cuda_by_params.get(p, {}).get("score", 0)), 1e-10)
        < 1e-4
        for p in all_params if p in cpu_by_params and p in cuda_by_params
    ):
        print("✓ 结果一致（相对误差 < 0.01%）".center(80))
    else:
        print("⚠ 存在差异，请检查实现细节".center(80))
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="验证 CUDA 和 CPU 优化结果的一致性")
    parser.add_argument("--strategy", required=True, help="策略文件路径")
    parser.add_argument("--symbol", required=True, help="交易对")
    parser.add_argument("--interval", required=True, help="K线周期")
    parser.add_argument("--start", default="2024-01-01", help="开始日期")
    parser.add_argument("--end", default="2024-12-31", help="结束日期")
    parser.add_argument("--db", default="data/reports.db", help="数据库路径")

    args = parser.parse_args()

    # 提取策略名称
    strategy_path = Path(args.strategy)
    strategy_name = None

    # 尝试导入策略获取类名
    import sys
    sys.path.insert(0, str(Path.cwd()))
    try:
        spec = __import__("importlib.util").util.spec_from_file_location("strategy_module", strategy_path)
        module = __import__("importlib.util").util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # 寻找 BaseStrategy 的子类
        from backtest.strategy import BaseStrategy
        for name, obj in vars(module).items():
            if isinstance(obj, type) and issubclass(obj, BaseStrategy) and obj is not BaseStrategy:
                strategy_name = name
                break
    except Exception as e:
        print(f"警告：无法自动检测策略名称 ({e})")
        strategy_name = strategy_path.stem.replace("_", "").title() + "Strategy"

    print(f"\n策略: {strategy_name}")
    print(f"交易对: {args.symbol}, 周期: {args.interval}")
    print(f"日期范围: {args.start} ~ {args.end}")
    print(f"数据库: {args.db}")

    if not Path(args.db).exists():
        print(f"\n✗ 错误: 数据库文件不存在 ({args.db})")
        print("请先运行优化: python -m backtest optimize --method numba-grid ...")
        return

    # 读取结果
    conn = sqlite3.connect(args.db)
    cursor = conn.cursor()

    query = """
    SELECT params_json, score, report_json, created_at
    FROM optimize_results
    WHERE strategy = ? AND symbol = ? AND interval = ?
        AND DATE(created_at) BETWEEN ? AND ?
    """

    cursor.execute(query, (strategy_name, args.symbol, args.interval, args.start, args.end))
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        print(f"\n✗ 未找到优化结果")
        print(f"请先运行: python -m backtest optimize --strategy {args.strategy} \\")
        print(f"    --symbol {args.symbol} --interval {args.interval} \\")
        print(f"    --method numba-grid (或 cuda-grid)")
        return

    # 分离 CPU 和 CUDA 结果
    # 注意：当前数据库中没有方法标记，需要从 created_at 推断或手动标记
    # 这里假设最新的是最后运行的

    results = [
        {
            "params": json.loads(r[0]),
            "score": r[1],
            "report": json.loads(r[2]),
            "created_at": r[3],
        }
        for r in rows
    ]

    if len(results) < 2:
        print(f"\n⚠ 仅找到 {len(results)} 条结果，需要至少 2 条（CPU + CUDA）")
        print("请分别运行 CPU 和 CUDA 优化:")
        print(f"  python -m backtest optimize --strategy {args.strategy} --method numba-grid ...")
        print(f"  python -m backtest optimize --strategy {args.strategy} --method cuda-grid ...")
        return

    # 简单启发式：按时间分割
    results.sort(key=lambda x: x["created_at"])
    mid = len(results) // 2
    cpu_results = results[:mid]
    cuda_results = results[mid:]

    print(f"\n找到 {len(results)} 条结果（假设前 {len(cpu_results)} 条为 CPU，后 {len(cuda_results)} 条为 CUDA）")

    compare_results(cpu_results, cuda_results)


if __name__ == "__main__":
    main()
