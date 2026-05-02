import json
import os
import signal
import sys
import traceback
from datetime import datetime, timezone

from backtest.live_exchange import LiveExchange
from backtest.live_feed import LiveFeed
from backtest.strategy import BaseStrategy

try:
    from binance.um_futures import UMFutures
except ImportError:
    UMFutures = None

_TESTNET_URL = "https://testnet.binancefuture.com"
_MAINNET_URL = "https://fapi.binance.com"


class LiveEngine:
    def __init__(
        self,
        strategy_class: type[BaseStrategy],
        symbol: str,
        interval: str,
        leverage: int,
        commission_rate: float = 0.0004,
        api_key: str = "",
        secret: str = "",
        testnet: bool = True,
        dry_run: bool = False,
        state_dir: str = "live_state",
    ):
        self.strategy_class = strategy_class
        self.symbol = symbol
        self.interval = interval
        self.leverage = leverage
        self.commission_rate = commission_rate
        self.api_key = api_key
        self.secret = secret
        self.testnet = testnet
        self.dry_run = dry_run
        self.state_dir = state_dir
        self._state_file = os.path.join(state_dir, f"{symbol}_{interval}.json")

    def run(self) -> None:
        os.makedirs(self.state_dir, exist_ok=True)

        base_url = _TESTNET_URL if self.testnet else _MAINNET_URL
        client = UMFutures(key=self.api_key, secret=self.secret, base_url=base_url)

        try:
            client.change_leverage(symbol=self.symbol, leverage=self.leverage)
        except Exception as e:
            print(f"[WARN] change_leverage: {e}", file=sys.stderr)

        live_exchange = LiveExchange(
            client=client, symbol=self.symbol, leverage=self.leverage,
            commission_rate=self.commission_rate, dry_run=self.dry_run,
        )
        live_exchange.sync()

        strategy = self.strategy_class(exchange=live_exchange, symbol=self.symbol)
        strategy.on_init()

        if os.path.exists(self._state_file):
            with open(self._state_file) as f:
                strategy.load_state(json.load(f))
            print(f"[INFO] Restored state from {self._state_file}")

        self._print_startup_summary(live_exchange)

        def _handle_sigterm(sig, frame):
            self._on_exit(live_exchange)
            sys.exit(0)

        signal.signal(signal.SIGTERM, _handle_sigterm)

        feed = LiveFeed(client=client, symbol=self.symbol, interval=self.interval)
        try:
            for bar in feed:
                self._process_bar(bar, strategy, live_exchange)
        except KeyboardInterrupt:
            self._on_exit(live_exchange)

    def _process_bar(self, bar, strategy: BaseStrategy, live_exchange: LiveExchange) -> None:
        ts = datetime.fromtimestamp(bar.timestamp / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
        live_exchange.sync()
        strategy._push_bar(bar)
        live_exchange.wait_fills(timeout=30.0)
        self._save_state(strategy)
        pos = live_exchange.get_position(self.symbol)
        pos_str = f"{pos.side} {pos.quantity:.2f}@{pos.entry_price:.2f}" if pos else "flat"
        print(f"[{ts}] balance={live_exchange.balance:.2f} equity={live_exchange.equity:.2f} pos={pos_str}")

    def _save_state(self, strategy: BaseStrategy) -> None:
        state = strategy.save_state()
        with open(self._state_file, "w") as f:
            json.dump(state, f)

    def _alert(self, msg: str) -> None:
        print(msg, file=sys.stderr)

    def _print_startup_summary(self, live_exchange: LiveExchange) -> None:
        mode = "TESTNET" if self.testnet else "MAINNET"
        dry = " [DRY-RUN]" if self.dry_run else ""
        print(f"\n=== LiveEngine {mode}{dry} ===")
        print(f"Strategy: {self.strategy_class.__name__}  Symbol: {self.symbol}  "
              f"Interval: {self.interval}  Leverage: {self.leverage}x")
        print(f"Balance: {live_exchange.balance:.2f} USDT  Equity: {live_exchange.equity:.2f} USDT")
        pos = live_exchange.get_position(self.symbol)
        if pos:
            print(f"Position: {pos.side} {pos.quantity:.2f} USDT @ {pos.entry_price:.2f}  "
                  f"PnL: {pos.unrealized_pnl:.2f}")
        else:
            print("Position: flat")
        print()

    def _on_exit(self, live_exchange: LiveExchange) -> None:
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
