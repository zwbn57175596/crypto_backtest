# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Start

### Installation
```bash
source .venv/bin/activate
pip install -e .
```

### Token-Efficient Tooling (RTK)

A global Claude Code `PreToolUse` hook (`rtk hook claude`) auto-rewrites Bash
commands to `rtk <cmd>` for compact output (e.g. `pytest tests/` →
`rtk pytest tests/`, `git status` → `rtk git status`). This is transparent and
saves 60–90% of tokens on dev operations.

**Auto-rewritten** (no action needed): `pytest`, `git`, `gh`, `grep`, `find`,
`ls`, `tree`, `wc`, `diff`, `curl`, `docker`, `npm`, `cargo`, etc.

**NOT auto-rewritten** — invoke `rtk` explicitly when output is large:
- `rtk read PATH` — filtered file read for large reports / optimization output
- `rtk summary <cmd>` — heuristic 2-line summary of any command
- `rtk err <cmd>` — show only errors / warnings from a command

Project commands (`python -m backtest run/optimize/collect/web`) are not
wrapped by rtk; their CLI output is already compact.

Meta: `rtk gain` shows current session savings, `rtk --version` verifies install
(>= 0.38.0 expected). If rtk is not installed, the hook is a no-op — commands
run unwrapped without errors, so contributors without rtk are unaffected.

### Common Commands
```bash
# Run all tests
pytest tests/ -v

# Run a single test file
pytest tests/test_exchange.py -v

# Run a specific test
pytest tests/test_exchange.py::TestMarketOrder::test_buy_market_opens_long -v

# View CLI help
python -m backtest --help

# Fetch historical klines from exchange
python -m backtest collect --exchange binance --symbol BTCUSDT --interval 1h \
    --start 2024-01-01 --end 2024-12-31

# Run backtest with a strategy
python -m backtest run --strategy strategies/example_ma_cross.py \
    --symbol BTCUSDT --interval 1h \
    --start 2024-01-01 --end 2024-12-31 \
    --balance 10000 --leverage 10

# Start web server to view reports
python -m backtest web --port 8000

# CUDA GPU-accelerated optimization (requires NVIDIA GPU + numba)
python -m backtest optimize --strategy strategies/consecutive_reverse.py \
    --symbol BTCUSDT --interval 1h \
    --start 2024-01-01 --end 2024-12-31 \
    --balance 1000 --leverage 50 \
    --params "CONSECUTIVE_THRESHOLD=3:8:1,POSITION_MULTIPLIER=1.0:1.5:0.1,INITIAL_POSITION_PCT=0.005:0.03:0.005,PROFIT_CANDLE_THRESHOLD=1:5:1" \
    --method cuda-grid --objective sharpe_ratio
```

## Architecture Overview

Event-driven backtesting engine with a clear data flow:

### Core Components
- **BacktestEngine** — Main event loop coordinator
- **DataFeed** — Streams klines from SQLite by timestamp
- **SimExchange** — Simulated exchange handling order matching, positions, margin, fees, funding, liquidations
- **BaseStrategy** — User-implemented strategy base class
- **Reporter** — Collects trades and equity curve, computes 14 performance metrics
- **DataCollector** — Fetches historical klines from Binance/OKX/HTX REST API → SQLite
- **NumbaGridOptimizer** — CPU JIT-compiled grid search with multiprocessing
- **CudaGridOptimizer** — GPU CUDA-accelerated grid search with auto-batching

### Event Loop Data Flow
```
DataFeed (pushes klines from SQLite)
    ↓
BacktestEngine.run() iterates for each bar:
    1. SimExchange.on_new_bar(bar)
       ├── Settle funding payments (every 8h at 00:00, 08:00, 16:00 UTC)
       ├── Match pending orders (market → open price, limit → meets bid/ask)
       ├── Update unrealized PnL
       ├── Check liquidations
       └── Record equity snapshot
    2. Strategy._push_bar(bar)
       └── Call user's on_bar() callback
    3. Strategy submits orders via buy()/sell()/close()
       └── Orders queued in exchange._pending_orders for next bar
```

### SimExchange Per-Bar Processing
Each bar triggers this sequence in `on_new_bar()`:

1. **_settle_funding()** — At UTC hours 0, 8, 16: `funding_payment = position.quantity × funding_rate`
   - Long positions pay (or receive if negative)
   - Short positions receive (or pay if negative)

2. **_match_orders()** — For each pending order matching the bar's symbol:
   - Market orders fill at `bar.open`
   - Limit buy orders fill if `bar.low <= order.price`
   - Limit sell orders fill if `bar.high >= order.price`

3. **_update_unrealized_pnl()** — For existing positions:
   - Long: `pnl = qty × (close - entry_price) / entry_price`
   - Short: `pnl = qty × (entry_price - close) / entry_price`

4. **_check_liquidation()** — If `margin / (margin + unrealized_pnl) >= (1 / maintenance_margin)`:
   - Force close entire position at current bar close
   - Record liquidation trade with pnl = -margin

5. **_record_equity()** — Snapshot equity: `balance + sum(margin + unrealized_pnl)`

## Key Design Decisions

### All Prices in USDT (U-margined)
- PnL always: `qty × (price_diff) / entry_price` in USDT
- No base asset calculations

### Order Execution
- **Quantity** — Order quantity is in USDT value (notional), not contract count
- **Commission** — Deducted from balance immediately on fill: `commission = quantity × commission_rate`
- **Margin** — Required margin: `quantity / leverage`; returned on position close

### Position Accounting
- All positions stored by symbol in `SimExchange._positions` dict
- Entry price averages across multiple orders in same direction
- Closing partially (order qty < position qty) reduces quantity; remainder stays open
- Closing fully deletes the position

### Storage Architecture
- **SQLite klines table** — Shared storage for both collectors and backtest engine
  - Table: `klines` with PK `(exchange, symbol, interval, timestamp)`
  - Collectors append new data; engine queries by time range
- **SQLite reports table** — Stores backtest results in `reports.db`
  - Fields: id, strategy, symbol, interval, created_at, report_json

### Web Layer
- **FastAPI** with 3 API routes + 1 static HTML
  - `GET /` — Returns index.html
  - `GET /api/reports` — List all reports
  - `GET /api/reports/{id}` — Get single report details
  - `POST /api/backtest/run` — Trigger backtest (optional)
- **Frontend** — Single HTML file using ECharts for charts
- **No state** — Web layer is stateless, all data from SQLite

### Strategy Pattern
- User creates a class inheriting `BaseStrategy`
- Implements `on_bar(bar: Bar)` callback
- Calls `self.buy(qty, price=None)` / `self.sell(qty, price=None)` / `self.close()`
- Accesses `self.position`, `self.balance`, `self.equity`, `self.history(n)`
- CLI auto-discovers the BaseStrategy subclass in the provided Python file

### Reporter Metrics (14 indicators)
1. `net_return` — (final_equity - initial) / initial
2. `annual_return` — Annualized based on trading days
3. `max_drawdown` — Peak-to-trough decline in equity curve
4. `max_dd_duration` — Bars to recover from max drawdown
5. `sharpe_ratio` — Risk-adjusted return
6. `sortino_ratio` — Only considers downside volatility
7. `win_rate` — (Profitable trades) / (Total trades)
8. `profit_factor` — (Total profit) / (Total loss)
9. `total_trades` — Count of all filled orders
10. `long_trades` — Buy-to-close orders
11. `short_trades` — Sell-to-close orders
12. `avg_hold_time` — Average bars held per position
13. `total_commission` — Sum of all commissions
14. `total_funding` — Sum of all funding payments

## Code Structure

```
src/backtest/
├── __main__.py           # CLI entry point: collect/run/web/optimize
├── engine.py             # BacktestEngine — event loop
├── data_feed.py          # DataFeed — SQLite → klines stream
├── exchange.py           # SimExchange — order matching + position tracking
├── strategy.py           # BaseStrategy — user strategy base class
├── models.py             # Dataclasses: Bar, Order, Position, Trade
├── reporter.py           # Reporter — metric calculation
├── optimizer.py          # Grid/Optuna/NumbaGrid optimizers
├── numba_simulate.py     # Numba JIT CPU-compiled simulation kernel
├── cuda_exchange.py      # CUDA device functions (fill_order, calc_quantity)
├── cuda_runner.py        # CudaGridOptimizer — GPU grid search with auto-batching
├── cuda_strategies/
│   ├── __init__.py       # Strategy registry (name → kernel mapping)
│   └── consecutive_reverse.py  # ConsecutiveReverse CUDA kernel
├── collector/
│   ├── base.py           # BaseCollector — async API client
│   ├── binance.py        # Binance collector (1500 klines/req)
│   ├── okx.py            # OKX collector (100 klines/req)
│   └── htx.py            # HTX collector (2000 klines/req)
└── web/
    ├── app.py            # FastAPI app factory
    ├── routes.py         # API route handlers
    └── static/index.html # Single-page ECharts report viewer

strategies/
└── example_ma_cross.py   # Example: Moving average crossover strategy

config/
└── default.yaml          # Default backtest parameters

tests/
├── conftest.py           # Pytest fixtures
├── test_engine.py        # BacktestEngine tests
├── test_exchange.py      # SimExchange tests
├── test_reporter.py      # Reporter tests
└── test_web.py           # Web routes tests
```

## Data Models

All are Python dataclasses in `src/backtest/models.py`:

### Bar (OHLCV)
```python
@dataclass
class Bar:
    symbol: str          # e.g., "BTCUSDT"
    timestamp: int       # milliseconds
    open: float
    high: float
    low: float
    close: float
    volume: float        # in base asset
    interval: str        # "1m", "5m", "1h", "4h", "1d"
```

### Order
```python
@dataclass
class Order:
    id: str
    symbol: str
    side: str            # "buy" | "sell"
    type: str            # "market" | "limit"
    quantity: float      # USDT value
    price: float | None  # limit price; None for market
    status: str          # "pending" → "filled" / "canceled"
    filled_price: float
    filled_at: int       # timestamp
    commission: float
```

### Position
```python
@dataclass
class Position:
    symbol: str
    side: str            # "long" | "short"
    quantity: float      # USDT notional
    entry_price: float   # average entry price
    leverage: int        # e.g., 10
    unrealized_pnl: float
    margin: float        # required margin
```

### Trade
```python
@dataclass
class Trade:
    id: str
    order_id: str
    symbol: str
    side: str            # "buy" | "sell"
    price: float
    quantity: float      # USDT
    pnl: float           # realized pnl; 0 for opens
    commission: float
    timestamp: int       # milliseconds
```

## Code Conventions

- **Python 3.11+** — Uses `float | None` union syntax
- **Type hints** — All function signatures use proper type hints
- **Dataclasses** — All data models use `@dataclass` from stdlib
- **Async** — Only in data collectors (`httpx.AsyncClient`)
- **No sync database** — Uses synchronous sqlite3 everywhere
- **Tests** — Located in `tests/` with fixtures in `conftest.py`
- **Entry point** — `backtest` CLI command via `src/backtest/__main__.py`

## Testing

Run tests with pytest (pythonpath configured in pyproject.toml):

```bash
# All tests
pytest tests/

# Verbose output
pytest tests/ -v

# Single file
pytest tests/test_exchange.py

# Single test with full output
pytest tests/test_exchange.py::TestMarketOrder::test_buy_market_opens_long -v -s

# With coverage
pytest tests/ --cov=backtest
```

Fixtures are defined in `tests/conftest.py` (e.g., mock exchanges, sample bars).
