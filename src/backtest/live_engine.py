# src/backtest/live_engine.py
import json
import signal
import sqlite3
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from backtest.live_connector import BaseExchangeConnector
from backtest.live_exchange import LiveExchange
from backtest.live_feed import LiveFeed
from backtest.live_history import LiveHistoryDB
from backtest.models import Bar
from backtest.strategy import BaseStrategy


class _StateDB:
    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS live_state (
                    account_id  TEXT NOT NULL,
                    strategy    TEXT NOT NULL,
                    symbol      TEXT NOT NULL,
                    interval    TEXT NOT NULL,
                    state_json  TEXT NOT NULL,
                    updated_at  TEXT NOT NULL,
                    PRIMARY KEY (account_id, strategy, symbol, interval)
                )
            """)

    def load(self, account_id: str, strategy: str, symbol: str, interval: str) -> dict | None:
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT state_json FROM live_state "
                "WHERE account_id=? AND strategy=? AND symbol=? AND interval=?",
                (account_id, strategy, symbol, interval),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def save(self, account_id: str, strategy: str, symbol: str, interval: str, state: dict) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO live_state "
                "(account_id, strategy, symbol, interval, state_json, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (account_id, strategy, symbol, interval,
                 json.dumps(state), datetime.now(timezone.utc).isoformat()),
            )


class LiveEngine:
    def __init__(
        self,
        strategy_class: type[BaseStrategy],
        symbol: str,
        interval: str,
        leverage: int,
        connector: BaseExchangeConnector,
        history_db: LiveHistoryDB,
        account_id: str,
        commission_rate: float = 0.0004,
        dry_run: bool = False,
        state_db: str = "data/live_state.db",
        sync_interval: int = 300,
    ):
        self.strategy_class = strategy_class
        self.symbol = symbol
        self.interval = interval
        self.leverage = leverage
        self.commission_rate = commission_rate
        self.dry_run = dry_run
        self._connector = connector
        self._history_db = history_db
        self._account_id = account_id
        self._state_db = _StateDB(state_db)
        self._sync_interval = sync_interval
        self._stop_sync: threading.Event | None = None

    def run(self) -> None:
        if not self.dry_run:
            try:
                self._connector.change_leverage(self.symbol, self.leverage)
            except Exception as e:
                print(f"[WARN] change_leverage: {e}", file=sys.stderr)

        live_exchange = LiveExchange(
            connector=self._connector,
            history_db=self._history_db,
            account_id=self._account_id,
            symbol=self.symbol,
            leverage=self.leverage,
            commission_rate=self.commission_rate,
            dry_run=self.dry_run,
        )
        live_exchange.sync()

        strategy = self.strategy_class(exchange=live_exchange, symbol=self.symbol)
        strategy.on_init()

        strategy_name = self.strategy_class.__name__
        saved = self._state_db.load(self._account_id, strategy_name, self.symbol, self.interval)
        if saved:
            strategy.load_state(saved)
            print(f"[INFO] Restored state: account={self._account_id} "
                  f"{strategy_name}/{self.symbol}/{self.interval}")

        if not self.dry_run:
            self._do_sync()

        self._print_startup_summary(live_exchange)

        if not self.dry_run:
            self._stop_sync = threading.Event()
            sync_thread = threading.Thread(target=self._sync_loop, daemon=True)
            sync_thread.start()

        def _handle_sigterm(_sig, _frame):
            self._on_exit(live_exchange)
            sys.exit(0)

        signal.signal(signal.SIGTERM, _handle_sigterm)

        feed = LiveFeed(connector=self._connector, symbol=self.symbol, interval=self.interval)
        try:
            for bar in feed:
                self._process_bar(bar, strategy, live_exchange)
        except KeyboardInterrupt:
            self._on_exit(live_exchange)

    def _process_bar(self, bar: Bar, strategy: BaseStrategy, live_exchange: LiveExchange) -> None:
        ts = datetime.fromtimestamp(bar.timestamp / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
        try:
            live_exchange.sync()
            strategy._push_bar(bar)
            live_exchange.wait_fills(timeout=30.0)
            self._save_state(strategy)
            pos = live_exchange.get_position(self.symbol)
            pos_str = f"{pos.side} {pos.quantity:.2f}@{pos.entry_price:.2f}" if pos else "flat"
            print(f"[{ts}] balance={live_exchange.balance:.2f} equity={live_exchange.equity:.2f} pos={pos_str}")
        except KeyboardInterrupt:
            raise
        except Exception as e:
            self._alert(f"[ERROR] bar {ts}: {e}")
            traceback.print_exc(file=sys.stderr)

    def _save_state(self, strategy: BaseStrategy) -> None:
        state = strategy.save_state()
        self._state_db.save(
            self._account_id, self.strategy_class.__name__, self.symbol, self.interval, state
        )
        self._history_db.record_state_snapshot(
            account_id=self._account_id,
            exchange=self._connector.exchange_name,
            strategy=self.strategy_class.__name__,
            symbol=self.symbol,
            interval=self.interval,
            state=state,
            ts=int(time.time() * 1000),
        )

    def _do_sync(self) -> None:
        exchange = self._connector.exchange_name
        since_orders = self._history_db.latest_ts(self._account_id, exchange, self.symbol, "orders")
        orders = self._connector.fetch_orders(self.symbol, since_ms=since_orders or None)
        for order in orders:
            self._history_db.upsert_order(self._account_id, exchange, order)

        since_trades = self._history_db.latest_ts(self._account_id, exchange, self.symbol, "trades")
        trades = self._connector.fetch_trades(self.symbol, since_ms=since_trades or None)
        self._history_db.upsert_trades(self._account_id, exchange, trades)

    def _sync_loop(self) -> None:
        while not self._stop_sync.wait(self._sync_interval):
            try:
                self._do_sync()
            except Exception as e:
                print(f"[WARN] background sync error: {e}", file=sys.stderr)

    def _alert(self, msg: str) -> None:
        print(msg, file=sys.stderr)

    def _print_startup_summary(self, live_exchange: LiveExchange) -> None:
        dry = " [DRY-RUN]" if self.dry_run else ""
        print(f"\n=== LiveEngine{dry} ===")
        print(f"Strategy: {self.strategy_class.__name__}  Symbol: {self.symbol}  "
              f"Interval: {self.interval}  Leverage: {self.leverage}x")
        print(f"Account ID: {self._account_id}  Exchange: {self._connector.exchange_name}")
        print(f"Balance: {live_exchange.balance:.2f} USDT  Equity: {live_exchange.equity:.2f} USDT")
        pos = live_exchange.get_position(self.symbol)
        if pos:
            print(f"Position: {pos.side} {pos.quantity:.2f} USDT @ {pos.entry_price:.2f}  "
                  f"PnL: {pos.unrealized_pnl:.2f}")
        else:
            print("Position: flat")
        print()

    def _on_exit(self, live_exchange: LiveExchange) -> None:
        if self._stop_sync is not None:
            self._stop_sync.set()
        pos = live_exchange.get_position(self.symbol)
        print("\n=== Stopping LiveEngine ===")
        if pos:
            print(f"Open position: {pos.side} {pos.quantity:.2f} USDT @ {pos.entry_price:.2f}")
            try:
                answer = input("Close position before exit? [y/N] ")
            except EOFError:
                answer = "n"
            if answer.strip().lower() == "y":
                side = "sell" if pos.side == "long" else "buy"
                live_exchange.submit_order(self.symbol, side, "market", pos.quantity)
                live_exchange.wait_fills(timeout=30.0)
                print("Position closed.")
        else:
            print("No open position.")
