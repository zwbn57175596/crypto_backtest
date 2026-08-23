"""IS/OOS 验证与上线红线检查（均值回归组合策略）。"""
from __future__ import annotations

import sys
from pathlib import Path

from backtest.mean_reversion import MRParams
from backtest.mean_reversion_engine import MeanReversionPortfolioEngine
from backtest.reporter import Reporter

MAX_DD_LIMIT = 0.25

DEFAULT_SYMBOLS = [
    "ETHUSDT", "BTCUSDT", "SOLUSDT", "DOGEUSDT", "WLDUSDT",
    "XRPUSDT", "PEPEUSDT", "AAVEUSDT", "UNIUSDT", "BNBUSDT",
]


def evaluate_gate(report: dict) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if report["net_return"] <= 0:
        reasons.append(f"net_return<=0 ({report['net_return']:+.4f})")
    if report["sharpe_ratio"] <= 0:
        reasons.append(f"sharpe_ratio<=0 ({report['sharpe_ratio']:.4f})")
    if report["max_drawdown"] >= MAX_DD_LIMIT:
        reasons.append(f"max_drawdown>={MAX_DD_LIMIT:.0%} ({report['max_drawdown']:.2%})")

    closes = [t for t in report["trades"] if t["pnl"] != 0.0]
    if closes:
        net_total = sum(t["pnl"] for t in report["trades"]) - sum(t["commission"] for t in report["trades"])
        avg_net = net_total / len(closes)
        if avg_net <= 0:
            reasons.append(f"avg_net_trade<=0 ({avg_net:+.4f} USDT/trade)")
    else:
        reasons.append("avg_net_trade<=0 (no closing trades)")

    return (len(reasons) == 0, reasons)


def run_window(
    db_path: str, symbols: list[str], exchange: str, interval: str,
    start: str, end: str, params: MRParams, balance: float, leverage: int,
) -> dict:
    engine = MeanReversionPortfolioEngine(
        db_path=db_path, symbols=symbols, interval=interval, exchange=exchange,
        params=params, balance=balance, leverage=leverage,
        start=f"{start} 00:00:00", end=f"{end} 23:59:59",
    )
    result = engine.run()
    report = Reporter.generate(result)
    report["miss_rate"] = result["miss_rate"]
    report["per_symbol_stats"] = result["per_symbol_stats"]
    return report


def _print_report(tag: str, report: dict) -> None:
    print(f"\n===== {tag} =====")
    print(f"  net_return:    {report['net_return']:+.2%}")
    print(f"  sharpe_ratio:  {report['sharpe_ratio']:.2f}")
    print(f"  max_drawdown:  {report['max_drawdown']:.2%}")
    print(f"  win_rate:      {report['win_rate']:.2%}")
    print(f"  total_trades:  {report['total_trades']}")
    print(f"  miss_rate:     {report['miss_rate']:.1%}")


def main() -> int:
    db_path = str(Path("data") / "klines.db")
    params = MRParams()  # 默认参数；调参后在此替换或从 CLI 传入
    symbols = DEFAULT_SYMBOLS
    balance, leverage, interval, exchange = 10000.0, 10, "1h", "okx"

    is_report = run_window(db_path, symbols, exchange, interval,
                           "2022-01-01", "2024-12-31", params, balance, leverage)
    oos_report = run_window(db_path, symbols, exchange, interval,
                            "2025-01-01", "2026-07-31", params, balance, leverage)

    _print_report("IN-SAMPLE 2022-01→2024-12", is_report)
    _print_report("OUT-OF-SAMPLE 2025-01→2026-07", oos_report)

    ok, reasons = evaluate_gate(oos_report)
    print("\n===== 验收红线（对 OOS）=====")
    if ok:
        print("  PASS —全部红线通过。")
        return 0
    print("  FAIL —未通过项：")
    for r in reasons:
        print(f"    - {r}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
