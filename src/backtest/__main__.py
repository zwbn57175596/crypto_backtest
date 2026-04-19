import argparse
import asyncio
import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path

from backtest.engine import BacktestEngine
from backtest.reporter import Reporter
from backtest.strategy import BaseStrategy


def _load_strategy(path: str) -> type[BaseStrategy]:
    spec = importlib.util.spec_from_file_location("user_strategy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for attr in dir(module):
        obj = getattr(module, attr)
        if isinstance(obj, type) and issubclass(obj, BaseStrategy) and obj is not BaseStrategy:
            return obj
    raise ValueError(f"No BaseStrategy subclass found in {path}")


def _parse_date_to_ms(date_str: str) -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def cmd_collect(args: argparse.Namespace) -> None:
    from backtest.collector import COLLECTORS
    collector_cls = COLLECTORS.get(args.exchange)
    if collector_cls is None:
        print(f"Unknown exchange: {args.exchange}. Choose from: {list(COLLECTORS.keys())}")
        sys.exit(1)
    db_path = args.db or str(Path("data") / "klines.db")
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    collector = collector_cls(db_path)
    start_ms = _parse_date_to_ms(args.start)
    end_ms = _parse_date_to_ms(args.end)
    print(f"Collecting {args.symbol} {args.interval} from {args.exchange} ...")
    asyncio.run(collector.fetch(args.symbol, args.interval, start_ms, end_ms))
    print("Done.")


def cmd_run(args: argparse.Namespace) -> None:
    import yaml
    config_path = Path("config/default.yaml")
    config = {}
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f).get("backtest", {})

    strategy_class = _load_strategy(args.strategy)
    db_path = args.db or str(Path("data") / "klines.db")

    engine = BacktestEngine(
        db_path=db_path, symbol=args.symbol, interval=args.interval,
        exchange=args.exchange or "binance", strategy_class=strategy_class,
        balance=args.balance or config.get("initial_balance", 10000),
        leverage=args.leverage or config.get("leverage", 10),
        commission_rate=config.get("commission_rate", 0.0004),
        funding_rate=config.get("funding_rate", 0.0001),
        maintenance_margin=config.get("maintenance_margin", 0.005),
        start=f"{args.start} 00:00:00" if args.start else None,
        end=f"{args.end} 23:59:59" if args.end else None,
    )

    print(f"Running backtest: {strategy_class.__name__} on {args.symbol} {args.interval} ...")
    result = engine.run()
    report = Reporter.generate(result)

    import json, sqlite3
    report_db = str(Path(db_path).parent / "reports.db")
    conn = sqlite3.connect(report_db)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy TEXT, symbol TEXT, interval TEXT,
            created_at TEXT, report_json TEXT
        )
    """)
    conn.execute(
        "INSERT INTO reports (strategy, symbol, interval, created_at, report_json) VALUES (?,?,?,?,?)",
        (strategy_class.__name__, args.symbol, args.interval,
         datetime.now(timezone.utc).isoformat(), json.dumps(report)),
    )
    conn.commit()
    conn.close()

    print(f"\nBacktest Complete: {strategy_class.__name__}")
    print(f"  Net Return:     {report['net_return']:.2%}")
    print(f"  Max Drawdown:   {report['max_drawdown']:.2%}")
    print(f"  Sharpe Ratio:   {report['sharpe_ratio']:.2f}")
    print(f"  Win Rate:       {report['win_rate']:.2%}")
    print(f"  Total Trades:   {report['total_trades']}")
    print(f"  Total Commission: {report['total_commission']:.2f} USDT")
    print(f"\nReport saved. View with: python -m backtest web")


def cmd_web(args: argparse.Namespace) -> None:
    import uvicorn
    from backtest.web.app import create_app
    db_path = args.db or str(Path("data") / "reports.db")
    app = create_app(db_path)
    print(f"Starting web server at http://localhost:{args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)


def main() -> None:
    parser = argparse.ArgumentParser(prog="backtest", description="Crypto futures backtester")
    sub = parser.add_subparsers(dest="command")

    p_collect = sub.add_parser("collect", help="Fetch historical klines")
    p_collect.add_argument("--exchange", required=True, choices=["binance", "okx", "htx"])
    p_collect.add_argument("--symbol", required=True)
    p_collect.add_argument("--interval", required=True)
    p_collect.add_argument("--start", required=True, help="YYYY-MM-DD")
    p_collect.add_argument("--end", required=True, help="YYYY-MM-DD")
    p_collect.add_argument("--db", default=None)

    p_run = sub.add_parser("run", help="Run backtest")
    p_run.add_argument("--strategy", required=True)
    p_run.add_argument("--symbol", required=True)
    p_run.add_argument("--interval", required=True)
    p_run.add_argument("--exchange", default="binance")
    p_run.add_argument("--start", default=None)
    p_run.add_argument("--end", default=None)
    p_run.add_argument("--balance", type=float, default=None)
    p_run.add_argument("--leverage", type=int, default=None)
    p_run.add_argument("--db", default=None)

    p_web = sub.add_parser("web", help="Start web report viewer")
    p_web.add_argument("--port", type=int, default=8000)
    p_web.add_argument("--db", default=None)

    args = parser.parse_args()
    if args.command == "collect":
        cmd_collect(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "web":
        cmd_web(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
