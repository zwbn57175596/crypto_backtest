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


def _collect_strategy_params(strategy_class: type) -> dict:
    """Snapshot UPPER_CASE class attributes (the strategy's tunable parameters)."""
    params: dict = {}
    for name in dir(strategy_class):
        if not name.isupper() or name.startswith("_"):
            continue
        value = getattr(strategy_class, name, None)
        if isinstance(value, (int, float, str, bool)):
            params[name] = value
    return params


def _apply_extra_params(strategy_class: type, extra_args: list[str]) -> None:
    it = iter(extra_args)
    for token in it:
        if not token.startswith("--"):
            continue
        key = token.lstrip("-")
        try:
            val_str = next(it)
        except StopIteration:
            continue
        # Try uppercase first (strategy convention), then lowercase
        attr_name = key.upper() if hasattr(strategy_class, key.upper()) else key
        existing = getattr(strategy_class, attr_name, None)
        if existing is not None:
            try:
                val = type(existing)(val_str)
            except (ValueError, TypeError):
                val = val_str
        else:
            try:
                val = int(val_str)
            except ValueError:
                try:
                    val = float(val_str)
                except ValueError:
                    val = val_str
        setattr(strategy_class, attr_name, val)
        print(f"  [param] {attr_name} = {val}")


def cmd_run(args: argparse.Namespace, extra_args: list[str] | None = None) -> None:
    import yaml
    config_path = Path("config/default.yaml")
    config = {}
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f).get("backtest", {})

    strategy_class = _load_strategy(args.strategy)
    if extra_args:
        _apply_extra_params(strategy_class, extra_args)
    if args.sizing_leverage is not None:
        strategy_class.LEVERAGE = args.sizing_leverage
        print(f"  [param] LEVERAGE (sizing) = {args.sizing_leverage}")
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
        margin_mode=args.margin_mode or config.get("margin_mode", "isolated"),
    )

    print(f"Running backtest: {strategy_class.__name__} on {args.symbol} {args.interval} ...")
    result = engine.run()
    report = Reporter.generate(result)

    report["strategy_params"] = _collect_strategy_params(strategy_class)
    report["run_leverage"] = engine.leverage
    report["run_balance"] = engine.balance
    report["run_start"] = args.start
    report["run_end"] = args.end

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


def cmd_optimize(args: argparse.Namespace) -> None:
    from backtest.optimizer import (
        GridSearchOptimizer, NumbaGridOptimizer, OptunaOptimizer,
        load_strategy_optimize_space, parse_params_string, save_results,
    )

    strategy_class = _load_strategy(args.strategy)
    if args.params:
        param_space = parse_params_string(args.params)
    else:
        param_space = load_strategy_optimize_space(args.strategy)

    summary = (
        f"Optimizing {strategy_class.__name__}: {param_space.total_combinations} combinations, "
        f"{args.n_jobs or 'auto'} workers, objective={args.objective}"
    )
    if args.method == "cuda-auto":
        summary += ", stage2=auto-refine"
    print(summary)

    if args.method == "optuna":
        optimizer = OptunaOptimizer(
            db_path=args.db or str(Path("data") / "klines.db"),
            strategy_path=args.strategy,
            symbol=args.symbol,
            interval=args.interval,
            start=args.start,
            end=args.end,
            balance=args.balance,
            leverage=args.leverage,
            exchange=args.exchange,
            param_space=param_space,
            objective=args.objective,
            n_trials=args.n_trials,
            n_jobs=args.n_jobs or 1,
            margin_mode=args.margin_mode,
        )
    elif args.method == "numba-grid":
        optimizer = NumbaGridOptimizer(
            db_path=args.db or str(Path("data") / "klines.db"),
            strategy_path=args.strategy,
            symbol=args.symbol,
            interval=args.interval,
            start=args.start,
            end=args.end,
            balance=args.balance,
            leverage=args.leverage,
            exchange=args.exchange,
            param_space=param_space,
            objective=args.objective,
            n_jobs=args.n_jobs,
        )
    elif args.method == "cuda-grid":
        from backtest.cuda_runner import CudaGridOptimizer
        optimizer = CudaGridOptimizer(
            db_path=args.db or str(Path("data") / "klines.db"),
            strategy_path=args.strategy,
            symbol=args.symbol,
            interval=args.interval,
            start=args.start,
            end=args.end,
            balance=args.balance,
            leverage=args.leverage,
            exchange=args.exchange,
            param_space=param_space,
            objective=args.objective,
        )
    elif args.method == "cuda-auto":
        from backtest.cuda_runner import CudaAutoOptimizer
        optimizer = CudaAutoOptimizer(
            db_path=args.db or str(Path("data") / "klines.db"),
            strategy_path=args.strategy,
            symbol=args.symbol,
            interval=args.interval,
            start=args.start,
            end=args.end,
            balance=args.balance,
            leverage=args.leverage,
            exchange=args.exchange,
            param_space=param_space,
            objective=args.objective,
        )
    else:
        optimizer = GridSearchOptimizer(
            db_path=args.db or str(Path("data") / "klines.db"),
            strategy_path=args.strategy,
            symbol=args.symbol,
            interval=args.interval,
            start=args.start,
            end=args.end,
            balance=args.balance,
            leverage=args.leverage,
            exchange=args.exchange,
            param_space=param_space,
            objective=args.objective,
            n_jobs=args.n_jobs,
            margin_mode=args.margin_mode,
        )

    result = optimizer.run()

    # Print results table
    print(f"\nOptimization Complete: {result.total_trials} trials, "
          f"best {result.objective} = {result.best_score:.4f} "
          f"({result.elapsed_seconds:.1f}s)\n")

    # Table header
    param_names = list(result.best_params.keys()) if result.best_params else []
    header = " Rank | " + " | ".join(f"{p:>12}" for p in param_names)
    header += f" | {'Score':>8} | {'Return':>8} | {'MaxDD':>8}"
    print(header)
    print("-" * len(header))

    top_n = min(args.top, len(result.all_trials))
    for i, trial in enumerate(result.all_trials[:top_n]):
        row = f" {i+1:>4} | "
        row += " | ".join(f"{trial['params'].get(p, ''):>12}" for p in param_names)
        report = trial.get("report", {})
        row += f" | {trial['score']:>8.4f}"
        row += f" | {report.get('net_return', 0):>+7.1%}"
        row += f" | {report.get('max_drawdown', 0):>-7.1%}"
        print(row)

    # Save to database
    report_db = str(Path(args.db or str(Path("data") / "klines.db")).parent / "reports.db")
    save_n = args.save_top if args.save_top > 0 else None
    save_results(
        db_path=report_db,
        strategy=strategy_class.__name__,
        symbol=args.symbol,
        interval=args.interval,
        start_date=args.start,
        end_date=args.end,
        result=result,
        top_n=save_n,
        leverage=args.leverage,
    )
    saved = save_n if save_n else result.total_trials
    print(f"\nResults saved to database ({min(saved, result.total_trials)} rows).")

    # Auto-save top N as full reports
    from backtest.optimizer import save_top_reports
    save_top_reports(
        result=result,
        top_n=min(args.report_top, len(result.all_trials)),
        db_path=args.db or str(Path("data") / "klines.db"),
        report_db_path=report_db,
        strategy_path=args.strategy,
        symbol=args.symbol,
        interval=args.interval,
        start=args.start,
        end=args.end,
        balance=args.balance,
        leverage=args.leverage,
        exchange=args.exchange,
    )
    print(f"Top {args.report_top} results saved as reports. View with: python -m backtest web")


def _load_env_file(path: str) -> dict[str, str]:
    """Parse a .env file (KEY=VALUE lines). Ignores comments and blank lines."""
    env: dict[str, str] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            env[key.strip()] = val.strip().strip('"').strip("'")
    return env


def cmd_live(args: argparse.Namespace) -> None:
    import hashlib
    import os
    from backtest.live_connector import BinanceConnector
    from backtest.live_engine import LiveEngine
    from backtest.live_history import LiveHistoryDB

    if args.env_file:
        try:
            env = _load_env_file(args.env_file)
        except FileNotFoundError:
            print(f"Error: env file not found: {args.env_file}")
            sys.exit(1)
        api_key = env.get("BINANCE_API_KEY", "")
        secret = env.get("BINANCE_SECRET", "")
    else:
        api_key = os.environ.get("BINANCE_API_KEY", "")
        secret = os.environ.get("BINANCE_SECRET", "")

    if (not api_key or not secret) and not args.dry_run:
        print(
            "Error: BINANCE_API_KEY and BINANCE_SECRET must be set via environment variables "
            "or --env-file. (Not required for --dry-run)"
        )
        sys.exit(1)

    strategy_class = _load_strategy(args.strategy)
    if args.extra_params:
        _apply_extra_params(strategy_class, args.extra_params)

    if args.exchange == "binance":
        connector = BinanceConnector(api_key=api_key, secret=secret, testnet=not args.no_testnet)
    else:
        print(f"Error: unsupported exchange: {args.exchange}")
        sys.exit(1)

    account_id = hashlib.sha256(api_key.encode()).hexdigest()[:16] if api_key else "dry_run"
    history_db = LiveHistoryDB(args.history_db)

    engine = LiveEngine(
        strategy_class=strategy_class,
        symbol=args.symbol,
        interval=args.interval,
        leverage=args.leverage,
        connector=connector,
        history_db=history_db,
        account_id=account_id,
        commission_rate=args.commission_rate,
        dry_run=args.dry_run,
        state_db=args.state_db,
        sync_interval=args.sync_interval,
    )
    engine.run()


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
    p_run.add_argument("--leverage", type=int, default=None,
                       help="Exchange leverage (for margin & liquidation)")
    p_run.add_argument("--margin-mode", choices=["isolated", "cross"], default=None, dest="margin_mode",
                       help="Margin mode (isolated=per-position, cross=account-wide)")
    p_run.add_argument("--sizing-leverage", type=int, default=None, dest="sizing_leverage",
                       help="Strategy sizing leverage (sets LEVERAGE class attr, used in _calc_quantity)")
    p_run.add_argument("--db", default=None)

    p_web = sub.add_parser("web", help="Start web report viewer")
    p_web.add_argument("--port", type=int, default=8000)
    p_web.add_argument("--db", default=None)

    p_opt = sub.add_parser("optimize", help="Optimize strategy parameters")
    p_opt.add_argument("--strategy", required=True)
    p_opt.add_argument("--symbol", required=True)
    p_opt.add_argument("--interval", required=True)
    p_opt.add_argument("--exchange", default="binance")
    p_opt.add_argument("--start", required=True, help="YYYY-MM-DD")
    p_opt.add_argument("--end", required=True, help="YYYY-MM-DD")
    p_opt.add_argument("--balance", type=float, default=10000.0)
    p_opt.add_argument("--leverage", type=int, default=10)
    p_opt.add_argument("--margin-mode", choices=["isolated", "cross"], default="isolated", dest="margin_mode",
                       help="Margin mode (isolated=per-position, cross=account-wide)")
    p_opt.add_argument("--params", default=None, help="e.g. X=1:10:2,Y=a|b|c; omit to use strategy OPTIMIZE_SPACE")
    p_opt.add_argument("--objective", default="sharpe_ratio",
                       choices=["sharpe_ratio", "net_return", "sortino_ratio", "profit_factor", "win_rate"])
    p_opt.add_argument("--method", default="grid", choices=["grid", "optuna", "numba-grid", "cuda-grid", "cuda-auto"])
    p_opt.add_argument("--n-jobs", type=int, default=None)
    p_opt.add_argument("--n-trials", type=int, default=100, help="For optuna method")
    p_opt.add_argument("--top", type=int, default=10, help="Show top N results in console")
    p_opt.add_argument("--save-top", type=int, default=1000, help="Save top N trials to DB (0=all)")
    p_opt.add_argument("--report-top", type=int, default=3, help="Save top N full reports (with equity curve)")
    p_opt.add_argument("--db", default=None)

    p_live = sub.add_parser("live", help="Run strategy in live trading mode")
    p_live.add_argument("--strategy", required=True, help="Path to strategy .py file")
    p_live.add_argument("--symbol", required=True)
    p_live.add_argument("--interval", required=True)
    p_live.add_argument("--leverage", type=int, required=True)
    p_live.add_argument("--exchange", default="binance", choices=["binance"],
                        help="Exchange connector (default: binance)")
    p_live.add_argument("--commission-rate", type=float, default=0.0004, dest="commission_rate")
    p_live.add_argument("--no-testnet", action="store_true", default=False,
                        help="Use mainnet (default is testnet)")
    p_live.add_argument("--dry-run", action="store_true", default=False,
                        help="Log orders without sending them (no API key required)")
    p_live.add_argument("--state-db", default="data/live_state.db", dest="state_db",
                        help="SQLite path for strategy state (default: data/live_state.db)")
    p_live.add_argument("--history-db", default="data/live_history.db", dest="history_db",
                        help="SQLite path for trading history (default: data/live_history.db)")
    p_live.add_argument("--sync-interval", type=int, default=300, dest="sync_interval",
                        help="Seconds between background exchange syncs (default: 300)")
    p_live.add_argument("--env-file", default=None, dest="env_file",
                        help="Path to .env file containing BINANCE_API_KEY and BINANCE_SECRET")
    p_live.add_argument("extra_params", nargs=argparse.REMAINDER,
                        help="Extra strategy params e.g. --CONSECUTIVE_THRESHOLD 5")

    args, extra = parser.parse_known_args()
    if args.command == "collect":
        cmd_collect(args)
    elif args.command == "run":
        cmd_run(args, extra)
    elif args.command == "web":
        cmd_web(args)
    elif args.command == "optimize":
        cmd_optimize(args)
    elif args.command == "live":
        cmd_live(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
