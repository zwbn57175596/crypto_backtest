# 多币均值回归组合策略 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一个多币、均值回归、maker 限价入场 + TP/SL/时间止损 bracket 出场 + 固定风险仓位（无马丁）的组合回测引擎，并用 IS/OOS 验证其样本外净期望。

**Architecture:** 复用现有事件驱动引擎。分三层扩展：(1) `SimExchange` 增加 maker/taker 双费率、限价单有效期、bar 级 bracket（TP/SL/时间止损）解析；(2) 纯函数模块 `mean_reversion.py` 承载信号/入场价/bracket 价/仓位公式（无状态、易测）；(3) 新引擎 `MeanReversionPortfolioEngine` 编排多币循环，不修改现有 `PortfolioBacktestEngine`（马丁版仍供旧策略复现）。CLI 增加 `mr-portfolio-run` 子命令，配套数据采集与 IS/OOS 验证脚本。

**Tech Stack:** Python 3.11+，stdlib dataclasses，sqlite3，pytest。无新增第三方依赖。

**Spec:** `docs/superpowers/specs/2026-07-13-multi-symbol-mean-reversion-portfolio-design.md`

## Global Constraints

- **不修改** `strategies/consecutive_reverse.py`（实盘保留）与 `src/backtest/portfolio_engine.py`（马丁版复现用）。
- **不修改** `SimExchange` 现有行为的默认路径：`maker_commission_rate` / `taker_commission_rate` 缺省为 `None` 时回退到 `commission_rate`，`expire_ts` 缺省为 `0` 表示永不过期。既有测试必须继续通过。
- **费率非对称建模（强制）**：止盈以 maker 平仓、止损与时间止损以 taker 平仓。OKX 合约：maker 0.02%/边（0.0002），taker 0.05%/边（0.0005）。
- **所有可调策略参数全币共享一套**（threshold、tp_pct、sl_pct、time_stop_bars、risk_pct、entry_offset_pct、entry_ttl_bars），装在单个 `MRParams`，抗过拟合。
- **反手方向**：连涨（streak_dir=+1）→ 做空（sell）；连跌（streak_dir=-1）→ 做多（buy）。
- **一币最多一个持仓**；组合并发上限 `max_position_symbols`；`total_margin/equity` 超过 `margin_ratio_limit` 时不再开新币。
- **金额约定**：`quantity` 为 USDT 名义（notional），沿用现有引擎语义。
- 币池（OKX USDT 永续，1h）：`ETH, BTC, SOL, DOGE, WLD, XRP, PEPE, AAVE, UNI, BNB`。DB 中 symbol 命名沿用现有 `data/klines.db` 风格（如 `BTCUSDT`），`exchange` 列为 `okx`。
- 提交格式：commit message 末尾附 `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`。

---

## File Structure

- `src/backtest/models.py`（修改）— `Order` 增加 `expire_ts: int = 0` 字段。
- `src/backtest/exchange.py`（修改）— 双费率、限价过期、`set_bracket` / `_resolve_brackets` / `_close_position_at`。
- `src/backtest/mean_reversion.py`（新建）— 纯函数信号/价格/仓位模块 + `MRParams`。
- `src/backtest/mean_reversion_engine.py`（新建）— `MeanReversionPortfolioEngine`。
- `src/backtest/__main__.py`（修改）— `mr-portfolio-run` 子命令。
- `scripts/mr_validate.py`（新建）— IS/OOS 验证 + 验收红线。
- `tests/test_exchange_bracket.py`（新建）— 双费率/过期/bracket 测试。
- `tests/test_mean_reversion.py`（新建）— 纯函数测试。
- `tests/test_mean_reversion_engine.py`（新建）— 引擎端到端（合成数据）测试。
- `tests/test_mr_validate.py`（新建）— 验收红线函数测试。

---

## Task 1: Order 增加 expire_ts 字段

**Files:**
- Modify: `src/backtest/models.py:16-27`
- Test: `tests/test_exchange_bracket.py`

**Interfaces:**
- Produces: `Order.expire_ts: int`（默认 `0`，表示永不过期；毫秒时间戳，`bar.timestamp >= expire_ts` 时若未成交则撤单）。

- [ ] **Step 1: 写失败测试**

在新文件 `tests/test_exchange_bracket.py` 顶部写：

```python
from backtest.models import Bar, Order
from backtest.exchange import SimExchange


def _bar(symbol, ts, o, h, l, c, interval="1h"):
    return Bar(symbol=symbol, timestamp=ts, open=o, high=h, low=l, close=c,
               volume=1.0, interval=interval)


def test_order_has_expire_ts_default_zero():
    order = Order(id="x", symbol="BTCUSDT", side="buy", type="limit", quantity=100.0)
    assert order.expire_ts == 0
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_exchange_bracket.py::test_order_has_expire_ts_default_zero -v`
Expected: FAIL —`AttributeError: 'Order' object has no attribute 'expire_ts'`

- [ ] **Step 3: 最小实现**

在 `src/backtest/models.py` 的 `Order` dataclass 末尾（`commission: float = 0.0` 之后）新增一行：

```python
@dataclass
class Order:
    id: str
    symbol: str
    side: str  # "buy" | "sell"
    type: str  # "market" | "limit"
    quantity: float
    price: float | None = None
    status: str = "pending"
    filled_price: float = 0.0
    filled_at: int = 0
    commission: float = 0.0
    expire_ts: int = 0  # ms; 0 = never expires. Unfilled limit dropped when bar.timestamp >= expire_ts
```

- [ ] **Step 4: 运行确认通过**

Run: `pytest tests/test_exchange_bracket.py::test_order_has_expire_ts_default_zero -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/backtest/models.py tests/test_exchange_bracket.py
git commit -m "feat: add expire_ts field to Order model"
```

---

## Task 2: SimExchange 双费率（maker/taker）

**Files:**
- Modify: `src/backtest/exchange.py:8-23`（`__init__` 签名与字段）
- Modify: `src/backtest/exchange.py:101-108`（`_execute_fill` 费率选择）
- Test: `tests/test_exchange_bracket.py`

**Interfaces:**
- Consumes: `Order.type`（`"limit"` → maker，`"market"` → taker）。
- Produces: `SimExchange(..., maker_commission_rate: float | None = None, taker_commission_rate: float | None = None)`；实例属性 `self.maker_commission_rate` / `self.taker_commission_rate`（`None` 时回退到 `commission_rate`）。

- [ ] **Step 1: 写失败测试**

追加到 `tests/test_exchange_bracket.py`：

```python
def test_limit_order_uses_maker_rate():
    ex = SimExchange(balance=1000, leverage=10, commission_rate=0.0004,
                     funding_rate=0.0, maintenance_margin=0.005,
                     maker_commission_rate=0.0002, taker_commission_rate=0.0005)
    ex.submit_order("BTCUSDT", "buy", "limit", 100.0, price=100.0)
    ex.on_new_bar(_bar("BTCUSDT", 1000, 101, 101, 99, 100))
    trades = ex.get_trades()
    assert trades[0].commission == 100.0 * 0.0002


def test_market_order_uses_taker_rate():
    ex = SimExchange(balance=1000, leverage=10, commission_rate=0.0004,
                     funding_rate=0.0, maintenance_margin=0.005,
                     maker_commission_rate=0.0002, taker_commission_rate=0.0005)
    ex.submit_order("BTCUSDT", "buy", "market", 100.0)
    ex.on_new_bar(_bar("BTCUSDT", 1000, 100, 101, 99, 100))
    assert ex.get_trades()[0].commission == 100.0 * 0.0005


def test_rates_fall_back_to_commission_rate():
    ex = SimExchange(balance=1000, leverage=10, commission_rate=0.0004,
                     funding_rate=0.0, maintenance_margin=0.005)
    assert ex.maker_commission_rate == 0.0004
    assert ex.taker_commission_rate == 0.0004
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_exchange_bracket.py -k "maker_rate or taker_rate or fall_back" -v`
Expected: FAIL —`TypeError: __init__() got an unexpected keyword argument 'maker_commission_rate'`

- [ ] **Step 3: 最小实现**

在 `src/backtest/exchange.py` 修改 `__init__` 签名（`commission_rate` 之后新增两个可选参数），并在字段区新增两行：

```python
    def __init__(
        self,
        balance: float,
        leverage: int,
        commission_rate: float,
        funding_rate: float,
        maintenance_margin: float,
        margin_mode: str = "isolated",
        maker_commission_rate: float | None = None,
        taker_commission_rate: float | None = None,
    ):
        self.initial_balance = balance
        self.balance = balance
        self.leverage = leverage
        self.commission_rate = commission_rate
        self.maker_commission_rate = (
            maker_commission_rate if maker_commission_rate is not None else commission_rate
        )
        self.taker_commission_rate = (
            taker_commission_rate if taker_commission_rate is not None else commission_rate
        )
        self.funding_rate = funding_rate
        self.maintenance_margin = maintenance_margin
        self.margin_mode = margin_mode

        self._positions: dict[str, Position] = {}
        self._pending_orders: list[Order] = []
        self._trades: list[Trade] = []
        self._equity_curve: list[tuple[int, float]] = []
        self._brackets: dict[str, dict] = {}
```

（`self._brackets` 在此提前初始化，Task 4 会用到。）

在 `_execute_fill` 中把费率计算改为按订单类型选择：

```python
    def _execute_fill(self, order: Order, price: float, bar: Bar) -> None:
        order.status = "filled"
        order.filled_price = price
        order.filled_at = bar.timestamp

        rate = self.maker_commission_rate if order.type == "limit" else self.taker_commission_rate
        commission = order.quantity * rate
        order.commission = commission
        self.balance -= commission
```

（该方法其余部分保持不变。）

- [ ] **Step 4: 运行确认通过**

Run: `pytest tests/test_exchange_bracket.py -k "maker_rate or taker_rate or fall_back" -v`
Expected: PASS

- [ ] **Step 5: 回归既有测试**

Run: `pytest tests/test_exchange.py -v`
Expected: 全部 PASS（因缺省回退，行为不变）。

- [ ] **Step 6: Commit**

```bash
git add src/backtest/exchange.py tests/test_exchange_bracket.py
git commit -m "feat: add maker/taker commission rates to SimExchange"
```

---

## Task 3: 限价单有效期（expiry）

**Files:**
- Modify: `src/backtest/exchange.py:34-56`（`submit_order` 增加 `expire_ts` 参数）
- Modify: `src/backtest/exchange.py:78-89`（`_match_orders` 增加过期撤单）
- Test: `tests/test_exchange_bracket.py`

**Interfaces:**
- Consumes: `Order.expire_ts`（Task 1）。
- Produces: `submit_order(symbol, side, type_, quantity, price=None, expire_ts=0) -> Order`；`_match_orders` 中未成交且 `bar.timestamp >= order.expire_ts`（`expire_ts != 0`）的订单被标记 `"canceled"` 并移除。成交仍优先于过期检查（过期 bar 内价格触及仍成交）。

- [ ] **Step 1: 写失败测试**

追加到 `tests/test_exchange_bracket.py`：

```python
def test_limit_order_expires_when_unfilled():
    ex = SimExchange(1000, 10, 0.0004, 0.0, 0.005)
    ex.submit_order("BTCUSDT", "buy", "limit", 100.0, price=90.0, expire_ts=2000)
    ex.on_new_bar(_bar("BTCUSDT", 1000, 100, 101, 95, 100))  # low 95 > 90: no fill, not expired
    assert len(ex._pending_orders) == 1
    ex.on_new_bar(_bar("BTCUSDT", 2000, 100, 101, 95, 100))  # ts >= expire, still no fill: canceled
    assert len(ex._pending_orders) == 0
    assert ex.get_position("BTCUSDT") is None


def test_limit_order_fills_on_expiry_bar_before_cancel():
    ex = SimExchange(1000, 10, 0.0004, 0.0, 0.005)
    ex.submit_order("BTCUSDT", "buy", "limit", 100.0, price=90.0, expire_ts=2000)
    ex.on_new_bar(_bar("BTCUSDT", 2000, 100, 101, 89, 100))  # low 89 <= 90: fills even on expiry bar
    assert ex.get_position("BTCUSDT") is not None
    assert len(ex._pending_orders) == 0


def test_expire_ts_zero_never_expires():
    ex = SimExchange(1000, 10, 0.0004, 0.0, 0.005)
    ex.submit_order("BTCUSDT", "buy", "limit", 100.0, price=90.0)  # expire_ts default 0
    ex.on_new_bar(_bar("BTCUSDT", 9_999_999, 100, 101, 95, 100))
    assert len(ex._pending_orders) == 1  # still alive
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_exchange_bracket.py -k "expire" -v`
Expected: FAIL —`submit_order()` 不接受 `expire_ts`（TypeError）。

- [ ] **Step 3: 最小实现**

修改 `submit_order`（新增 `expire_ts` 参数并写入 Order）：

```python
    def submit_order(
        self, symbol: str, side: str, type_: str, quantity: float,
        price: float | None = None, expire_ts: int = 0,
    ) -> Order:
        if self.is_bankrupt:
            return Order(
                id="rejected",
                symbol=symbol,
                side=side,
                type=type_,
                quantity=quantity,
                price=price,
                status="canceled",
            )
        order = Order(
            id=uuid.uuid4().hex[:8],
            symbol=symbol,
            side=side,
            type=type_,
            quantity=quantity,
            price=price,
            expire_ts=expire_ts,
        )
        self._pending_orders.append(order)
        return order
```

修改 `_match_orders`（成交优先，未成交再判过期）：

```python
    def _match_orders(self, bar: Bar) -> None:
        still_pending = []
        for order in self._pending_orders:
            if order.symbol != bar.symbol:
                still_pending.append(order)
                continue
            filled_price = self._try_fill(order, bar)
            if filled_price is not None:
                self._execute_fill(order, filled_price, bar)
                continue
            if order.expire_ts and bar.timestamp >= order.expire_ts:
                order.status = "canceled"
                continue
            still_pending.append(order)
        self._pending_orders = still_pending
```

- [ ] **Step 4: 运行确认通过**

Run: `pytest tests/test_exchange_bracket.py -k "expire" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/backtest/exchange.py tests/test_exchange_bracket.py
git commit -m "feat: add limit-order expiry to SimExchange"
```

---

## Task 4: bar 级 bracket 解析（TP/SL/时间止损）

**Files:**
- Modify: `src/backtest/exchange.py:58-64`（`on_new_bar` 插入 `_resolve_brackets`）
- Modify: `src/backtest/exchange.py`（新增 `set_bracket` / `_resolve_brackets` / `_close_position_at`）
- Test: `tests/test_exchange_bracket.py`

**Interfaces:**
- Consumes: `self._brackets`（Task 2 已初始化）、`self.maker_commission_rate` / `self.taker_commission_rate`。
- Produces:
  - `set_bracket(symbol: str, tp_price: float, sl_price: float, expire_ts: int) -> None` — 登记/覆盖某 symbol 的 bracket。`expire_ts=0` 表示不做时间止损。
  - `_resolve_brackets(bar)` 在 `on_new_bar` 中于 `_update_unrealized_pnl` 之后、`_check_liquidation` 之前调用。
  - 平仓 Trade 的 `order_id` 取值：止盈 `"bracket-tp"`（maker 费）、止损 `"bracket-sl"`（taker 费）、时间止损 `"bracket-time"`（taker 费，按 `bar.close` 平）。
  - **保守规则**：同一根 bar 同触 TP 与 SL 时，先判 SL。多头 SL 触发条件 `bar.low <= sl_price`、TP `bar.high >= tp_price`；空头 SL `bar.high >= sl_price`、TP `bar.low <= tp_price`。
  - 持仓已不存在（如被强平）时，`_resolve_brackets` 清除该 symbol 的过期 bracket。

- [ ] **Step 1: 写失败测试**

追加到 `tests/test_exchange_bracket.py`。辅助函数开一个多头仓（market 成交于 open）：

```python
def _open_long_at_100(ex):
    ex.submit_order("BTCUSDT", "buy", "market", 100.0)
    ex.on_new_bar(_bar("BTCUSDT", 1000, 100, 100, 100, 100))
    assert ex.get_position("BTCUSDT").entry_price == 100.0


def test_bracket_take_profit_long_uses_maker():
    ex = SimExchange(1000, 10, 0.0, 0.0, 0.005,
                     maker_commission_rate=0.0002, taker_commission_rate=0.0005)
    _open_long_at_100(ex)
    ex.set_bracket("BTCUSDT", tp_price=105.0, sl_price=95.0, expire_ts=0)
    ex.on_new_bar(_bar("BTCUSDT", 2000, 100, 106, 99, 104))  # high 106 >= 105 → TP
    assert ex.get_position("BTCUSDT") is None
    t = ex.get_trades()[-1]
    assert t.order_id == "bracket-tp"
    assert t.price == 105.0
    assert abs(t.pnl - 5.0) < 1e-9              # 100*(105-100)/100
    assert abs(t.commission - 100.0 * 0.0002) < 1e-12


def test_bracket_stop_loss_long_uses_taker_and_sl_wins_ties():
    ex = SimExchange(1000, 10, 0.0, 0.0, 0.005,
                     maker_commission_rate=0.0002, taker_commission_rate=0.0005)
    _open_long_at_100(ex)
    ex.set_bracket("BTCUSDT", tp_price=105.0, sl_price=95.0, expire_ts=0)
    ex.on_new_bar(_bar("BTCUSDT", 2000, 100, 106, 94, 96))  # both hit → SL first
    t = ex.get_trades()[-1]
    assert t.order_id == "bracket-sl"
    assert t.price == 95.0
    assert abs(t.pnl - (-5.0)) < 1e-9
    assert abs(t.commission - 100.0 * 0.0005) < 1e-12


def test_bracket_time_stop_closes_at_close_price_taker():
    ex = SimExchange(1000, 10, 0.0, 0.0, 0.005,
                     maker_commission_rate=0.0002, taker_commission_rate=0.0005)
    _open_long_at_100(ex)
    ex.set_bracket("BTCUSDT", tp_price=105.0, sl_price=95.0, expire_ts=2000)
    ex.on_new_bar(_bar("BTCUSDT", 2000, 100, 102, 99, 101))  # neither TP/SL, ts >= expire
    t = ex.get_trades()[-1]
    assert t.order_id == "bracket-time"
    assert t.price == 101.0
    assert abs(t.pnl - 1.0) < 1e-9
    assert abs(t.commission - 100.0 * 0.0005) < 1e-12


def test_bracket_short_take_profit():
    ex = SimExchange(1000, 10, 0.0, 0.0, 0.005,
                     maker_commission_rate=0.0002, taker_commission_rate=0.0005)
    ex.submit_order("BTCUSDT", "sell", "market", 100.0)
    ex.on_new_bar(_bar("BTCUSDT", 1000, 100, 100, 100, 100))
    assert ex.get_position("BTCUSDT").side == "short"
    ex.set_bracket("BTCUSDT", tp_price=95.0, sl_price=105.0, expire_ts=0)  # short: tp below, sl above
    ex.on_new_bar(_bar("BTCUSDT", 2000, 100, 101, 94, 96))  # low 94 <= 95 → TP
    t = ex.get_trades()[-1]
    assert t.order_id == "bracket-tp"
    assert t.price == 95.0
    assert abs(t.pnl - 5.0) < 1e-9  # short: 100*(100-95)/100
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_exchange_bracket.py -k "bracket" -v`
Expected: FAIL —`SimExchange` 无 `set_bracket`（AttributeError）。

- [ ] **Step 3: 最小实现**

在 `on_new_bar` 序列中插入 `_resolve_brackets`（放在 `_update_unrealized_pnl` 之后、`_check_liquidation` 之前）：

```python
    def on_new_bar(self, bar: Bar) -> None:
        self._settle_funding(bar)
        self._match_orders(bar)
        self._update_unrealized_pnl(bar)
        self._resolve_brackets(bar)
        self._check_liquidation(bar)
        self._ensure_no_bankruptcy(bar)
        self._record_equity(bar)
```

在类中新增三个方法（建议放在 `_update_unrealized_pnl` 之后）：

```python
    def set_bracket(self, symbol: str, tp_price: float, sl_price: float, expire_ts: int) -> None:
        self._brackets[symbol] = {
            "tp_price": tp_price,
            "sl_price": sl_price,
            "expire_ts": expire_ts,
        }

    def _resolve_brackets(self, bar: Bar) -> None:
        bracket = self._brackets.get(bar.symbol)
        if bracket is None:
            return
        pos = self._positions.get(bar.symbol)
        if pos is None:
            self._brackets.pop(bar.symbol, None)  # stale (position closed elsewhere)
            return
        tp = bracket["tp_price"]
        sl = bracket["sl_price"]
        expire_ts = bracket["expire_ts"]

        if pos.side == "long":
            if bar.low <= sl:  # conservative: SL before TP
                self._close_position_at(bar.symbol, sl, self.taker_commission_rate, bar, "sl")
                return
            if bar.high >= tp:
                self._close_position_at(bar.symbol, tp, self.maker_commission_rate, bar, "tp")
                return
        else:  # short
            if bar.high >= sl:
                self._close_position_at(bar.symbol, sl, self.taker_commission_rate, bar, "sl")
                return
            if bar.low <= tp:
                self._close_position_at(bar.symbol, tp, self.maker_commission_rate, bar, "tp")
                return

        if expire_ts and bar.timestamp >= expire_ts:
            self._close_position_at(bar.symbol, bar.close, self.taker_commission_rate, bar, "time")

    def _close_position_at(
        self, symbol: str, price: float, fee_rate: float, bar: Bar, reason: str
    ) -> None:
        pos = self._positions.get(symbol)
        if pos is None:
            return
        if pos.side == "long":
            pnl = pos.quantity * (price - pos.entry_price) / pos.entry_price
        else:
            pnl = pos.quantity * (pos.entry_price - price) / pos.entry_price
        commission = pos.quantity * fee_rate
        self.balance += pnl
        self.balance += pos.margin
        self.balance -= commission
        self._trades.append(Trade(
            id=uuid.uuid4().hex[:8],
            order_id=f"bracket-{reason}",
            symbol=symbol,
            side="sell" if pos.side == "long" else "buy",
            price=price,
            quantity=pos.quantity,
            pnl=pnl,
            commission=commission,
            timestamp=bar.timestamp,
        ))
        del self._positions[symbol]
        self._brackets.pop(symbol, None)
```

- [ ] **Step 4: 运行确认通过**

Run: `pytest tests/test_exchange_bracket.py -k "bracket" -v`
Expected: PASS

- [ ] **Step 5: 全量回归**

Run: `pytest tests/ -q`
Expected: 全部 PASS。

- [ ] **Step 6: Commit**

```bash
git add src/backtest/exchange.py tests/test_exchange_bracket.py
git commit -m "feat: add TP/SL/time-stop bracket resolution to SimExchange"
```

---

## Task 5: mean_reversion.py 纯函数模块

**Files:**
- Create: `src/backtest/mean_reversion.py`
- Test: `tests/test_mean_reversion.py`

**Interfaces:**
- Produces（供 Task 6 引擎调用）：
  - `MRParams`（dataclass，全字段默认值）：`threshold: int`, `tp_pct: float`, `sl_pct: float`, `time_stop_bars: int`, `risk_pct: float`, `entry_offset_pct: float`, `entry_ttl_bars: int`。
  - `bar_direction(open_: float, close: float) -> int` — `+1/-1/0`。
  - `update_streak(count: int, streak_dir: int, direction: int) -> tuple[int, int]` — 返回新的 `(count, streak_dir)`；`direction==0` 原样返回。
  - `fade_side(streak_dir: int) -> str` — `+1→"sell"`，`-1→"buy"`（下单方向）。
  - `entry_price(close: float, streak_dir: int, offset_pct: float) -> float` — 连涨在上方挂、连跌在下方挂。
  - `bracket_prices(entry: float, position_side: str, tp_pct: float, sl_pct: float) -> tuple[float, float]` — 返回 `(tp_price, sl_price)`；`position_side ∈ {"long","short"}`。
  - `position_notional(equity: float, risk_pct: float, sl_pct: float) -> float` — `equity*risk_pct/sl_pct`；`sl_pct<=0` 返回 `0.0`。

- [ ] **Step 1: 写失败测试**

`tests/test_mean_reversion.py`：

```python
from backtest.mean_reversion import (
    MRParams, bar_direction, update_streak, fade_side,
    entry_price, bracket_prices, position_notional,
)


def test_mrparams_defaults_present():
    p = MRParams()
    assert p.threshold >= 2
    assert 0 < p.tp_pct < 0.1 and 0 < p.sl_pct < 0.1
    assert p.time_stop_bars >= 1
    assert 0 < p.risk_pct < 0.1
    assert p.entry_offset_pct >= 0
    assert p.entry_ttl_bars >= 1


def test_bar_direction():
    assert bar_direction(100, 101) == 1
    assert bar_direction(101, 100) == -1
    assert bar_direction(100, 100) == 0


def test_update_streak_extends_and_resets():
    assert update_streak(0, 0, 1) == (1, 1)      # first up
    assert update_streak(1, 1, 1) == (2, 1)      # extend up
    assert update_streak(3, 1, -1) == (1, -1)    # flip to down
    assert update_streak(3, 1, 0) == (3, 1)      # doji: unchanged


def test_fade_side():
    assert fade_side(1) == "sell"   # up-streak → short
    assert fade_side(-1) == "buy"   # down-streak → long


def test_entry_price_offsets_pullback_side():
    assert entry_price(100.0, 1, 0.001) == 100.0 * 1.001   # sell above
    assert entry_price(100.0, -1, 0.001) == 100.0 * 0.999  # buy below


def test_bracket_prices_long_and_short():
    tp, sl = bracket_prices(100.0, "long", 0.01, 0.005)
    assert tp == 101.0 and sl == 99.5
    tp, sl = bracket_prices(100.0, "short", 0.01, 0.005)
    assert tp == 99.0 and sl == 100.5


def test_position_notional():
    assert position_notional(10000.0, 0.005, 0.01) == 5000.0
    assert position_notional(10000.0, 0.005, 0.0) == 0.0
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_mean_reversion.py -v`
Expected: FAIL —`ModuleNotFoundError: No module named 'backtest.mean_reversion'`

- [ ] **Step 3: 最小实现**

`src/backtest/mean_reversion.py`：

```python
from dataclasses import dataclass


@dataclass
class MRParams:
    """全币共享的一套均值回归参数（抗过拟合）。"""
    threshold: int = 4            # 连续同色 K 线数达到后反手
    tp_pct: float = 0.006         # 止盈距离（比例）
    sl_pct: float = 0.006         # 止损距离（比例）
    time_stop_bars: int = 12      # 最长持仓 bar 数
    risk_pct: float = 0.005       # 每笔风险占净值比例
    entry_offset_pct: float = 0.001  # maker 限价相对 close 的偏移
    entry_ttl_bars: int = 2       # 限价单有效期（bar 数）


def bar_direction(open_: float, close: float) -> int:
    if close > open_:
        return 1
    if close < open_:
        return -1
    return 0


def update_streak(count: int, streak_dir: int, direction: int) -> tuple[int, int]:
    if direction == 0:
        return count, streak_dir
    if direction == streak_dir:
        return count + 1, streak_dir
    return 1, direction


def fade_side(streak_dir: int) -> str:
    # 连涨 → 做空（sell）；连跌 → 做多（buy）
    return "sell" if streak_dir == 1 else "buy"


def entry_price(close: float, streak_dir: int, offset_pct: float) -> float:
    # 在反弹侧挂 maker 限价
    if streak_dir == 1:                 # 做空：挂在 close 上方
        return close * (1 + offset_pct)
    return close * (1 - offset_pct)      # 做多：挂在 close 下方


def bracket_prices(
    entry: float, position_side: str, tp_pct: float, sl_pct: float
) -> tuple[float, float]:
    if position_side == "long":
        return entry * (1 + tp_pct), entry * (1 - sl_pct)
    return entry * (1 - tp_pct), entry * (1 + sl_pct)


def position_notional(equity: float, risk_pct: float, sl_pct: float) -> float:
    if sl_pct <= 0:
        return 0.0
    return equity * risk_pct / sl_pct
```

- [ ] **Step 4: 运行确认通过**

Run: `pytest tests/test_mean_reversion.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/backtest/mean_reversion.py tests/test_mean_reversion.py
git commit -m "feat: add mean_reversion pure-function module"
```

---

## Task 6: MeanReversionPortfolioEngine 引擎

**Files:**
- Create: `src/backtest/mean_reversion_engine.py`
- Test: `tests/test_mean_reversion_engine.py`

**Interfaces:**
- Consumes: `SimExchange`（Task 2–4）、`MultiSymbolDataFeed`、`mean_reversion` 全部纯函数（Task 5）。
- Produces:
  - `MeanReversionPortfolioEngine(db_path, symbols, interval, exchange, params: MRParams, balance=10000.0, leverage=10, commission_rate=0.0005, funding_rate=0.0001, maintenance_margin=0.005, maker_commission_rate=0.0002, taker_commission_rate=0.0005, max_position_symbols=10, margin_ratio_limit=0.8, start=None, end=None)`。
  - `run() -> dict`，键：`trades, trades_count, equity_curve, final_equity, initial_balance, portfolio_symbols, per_symbol_stats, entries_submitted, entries_filled, miss_rate`。
  - 循环语义：每个 timestamp 先对所有 symbol 调 `sim.on_new_bar(bar)`（撮合+解 bracket），再逐 symbol 调 `_process_symbol` 更新连续计数、给新持仓补挂 bracket、或发新入场限价单。
- 备注：入场用 maker 限价 + `expire_ts`；占位统计把「已有持仓的 symbol」「有未成交挂单的 symbol」「本 bar 已提交的 symbol」都算作占用，避免同 bar 超发。

- [ ] **Step 1: 写失败测试**

`tests/test_mean_reversion_engine.py`（构造一个临时 SQLite，塞入能触发一整笔盈利交易的合成 K 线）：

```python
import sqlite3

from backtest.mean_reversion import MRParams
from backtest.mean_reversion_engine import MeanReversionPortfolioEngine

INTERVAL_MS = 3_600_000


def _make_db(tmp_path, rows):
    db = tmp_path / "k.db"
    conn = sqlite3.connect(db)
    conn.execute("""
        CREATE TABLE klines (
            exchange TEXT, symbol TEXT, interval TEXT, timestamp INTEGER,
            open REAL, high REAL, low REAL, close REAL, volume REAL,
            PRIMARY KEY (exchange, symbol, interval, timestamp)
        )
    """)
    conn.executemany(
        "INSERT INTO klines VALUES (?,?,?,?,?,?,?,?,?)", rows
    )
    conn.commit()
    conn.close()
    return str(db)


def _row(ts, o, h, l, c):
    return ("okx", "AAAUSDT", "1h", ts, o, h, l, c, 1.0)


def test_engine_runs_one_profitable_short_trade(tmp_path):
    # 3 根连涨 → 触发做空；随后限价成交，价格回落触发止盈
    rows = [
        _row(0 * INTERVAL_MS, 100, 101, 100, 101),   # up  count=1
        _row(1 * INTERVAL_MS, 101, 102, 101, 102),   # up  count=2
        _row(2 * INTERVAL_MS, 102, 103, 102, 103),   # up  count=3 → submit sell limit @103*1.001=103.103
        _row(3 * INTERVAL_MS, 103, 104, 102.5, 103.5),  # high 104 >= 103.103 → sell fills; bracket set
        _row(4 * INTERVAL_MS, 103, 103.2, 101, 102),    # low 101 <= tp(103.103*0.99≈102.07) → TP
        _row(5 * INTERVAL_MS, 102, 102, 101.9, 101.95),
    ]
    db = _make_db(tmp_path, rows)
    params = MRParams(threshold=3, tp_pct=0.01, sl_pct=0.01, time_stop_bars=10,
                      risk_pct=0.01, entry_offset_pct=0.001, entry_ttl_bars=3)
    engine = MeanReversionPortfolioEngine(
        db_path=db, symbols=["AAAUSDT"], interval="1h", exchange="okx",
        params=params, balance=10000.0, leverage=10,
    )
    result = engine.run()

    order_ids = [t.order_id for t in result["trades"]]
    sides = [t.side for t in result["trades"]]
    assert "bracket-tp" in order_ids          # 止盈平仓
    assert sides[0] == "sell"                  # 首单为反手做空
    assert result["final_equity"] > result["initial_balance"]   # 单笔净盈利（含双边 maker 费）
    assert result["entries_submitted"] >= 1
    assert result["entries_filled"] >= 1
    assert 0.0 <= result["miss_rate"] <= 1.0


def test_engine_reports_miss_when_limit_never_fills(tmp_path):
    # 连涨触发做空挂单，但价格再不上探 → 挂单过期，miss_rate 上升，无成交
    rows = [
        _row(0 * INTERVAL_MS, 100, 101, 100, 101),
        _row(1 * INTERVAL_MS, 101, 102, 101, 102),
        _row(2 * INTERVAL_MS, 102, 103, 102, 103),   # count=3 → sell limit @103.103, ttl=2 bars
        _row(3 * INTERVAL_MS, 102, 102.5, 101, 101.5),  # high 102.5 < 103.103: no fill
        _row(4 * INTERVAL_MS, 101, 101.4, 100, 100.5),  # high < 103.103, ts >= expire: canceled
        _row(5 * INTERVAL_MS, 100, 100.2, 99, 99.5),
    ]
    db = _make_db(tmp_path, rows)
    params = MRParams(threshold=3, tp_pct=0.01, sl_pct=0.01, time_stop_bars=10,
                      risk_pct=0.01, entry_offset_pct=0.001, entry_ttl_bars=2)
    engine = MeanReversionPortfolioEngine(
        db_path=db, symbols=["AAAUSDT"], interval="1h", exchange="okx",
        params=params, balance=10000.0, leverage=10,
    )
    result = engine.run()
    assert result["entries_submitted"] >= 1
    assert result["entries_filled"] == 0
    assert result["miss_rate"] == 1.0
    assert result["final_equity"] == result["initial_balance"]  # 无成交无费
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_mean_reversion_engine.py -v`
Expected: FAIL —`ModuleNotFoundError: No module named 'backtest.mean_reversion_engine'`

- [ ] **Step 3: 最小实现**

`src/backtest/mean_reversion_engine.py`：

```python
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone

from backtest.exchange import SimExchange
from backtest.mean_reversion import (
    MRParams, bar_direction, bracket_prices, entry_price, fade_side,
    position_notional, update_streak,
)
from backtest.models import Bar, Trade
from backtest.multi_data_feed import MultiSymbolDataFeed

_INTERVAL_MS: dict[str, int] = {
    "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
    "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000, "6h": 21_600_000,
    "8h": 28_800_000, "12h": 43_200_000, "1d": 86_400_000,
}


class MeanReversionPortfolioEngine:
    """
    多币均值回归组合引擎：连续反向信号 + maker 限价入场 +
    TP/SL/时间止损 bracket 出场 + 固定风险仓位（无马丁）。

    与 PortfolioBacktestEngine 并列，不共享代码（后者是马丁复现用）。
    """

    def __init__(
        self,
        db_path: str,
        symbols: list[str],
        interval: str,
        exchange: str,
        params: MRParams,
        balance: float = 10000.0,
        leverage: int = 10,
        commission_rate: float = 0.0005,
        funding_rate: float = 0.0001,
        maintenance_margin: float = 0.005,
        maker_commission_rate: float = 0.0002,
        taker_commission_rate: float = 0.0005,
        max_position_symbols: int = 10,
        margin_ratio_limit: float = 0.8,
        start: str | None = None,
        end: str | None = None,
    ):
        self.db_path = db_path
        self.symbols = symbols
        self.interval = interval
        self.exchange_name = exchange
        self.params = params
        self.balance = balance
        self.leverage = leverage
        self.commission_rate = commission_rate
        self.funding_rate = funding_rate
        self.maintenance_margin = maintenance_margin
        self.maker_commission_rate = maker_commission_rate
        self.taker_commission_rate = taker_commission_rate
        self.max_position_symbols = max_position_symbols
        self.margin_ratio_limit = margin_ratio_limit
        self.start_ts = self._parse_time(start) if start else None
        self.end_ts = self._parse_time(end) if end else None
        if interval not in _INTERVAL_MS:
            raise ValueError(f"Unsupported interval: {interval!r}")
        self._interval_ms = _INTERVAL_MS[interval]
        self._entries_submitted = 0

    @staticmethod
    def _parse_time(s: str) -> int:
        dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)

    def run(self) -> dict:
        sim = SimExchange(
            balance=self.balance,
            leverage=self.leverage,
            commission_rate=self.commission_rate,
            funding_rate=self.funding_rate,
            maintenance_margin=self.maintenance_margin,
            margin_mode="cross",
            maker_commission_rate=self.maker_commission_rate,
            taker_commission_rate=self.taker_commission_rate,
        )

        states: dict[str, dict] = {
            sym: {"count": 0, "streak_dir": 0, "pending_expire_ts": 0}
            for sym in self.symbols
        }

        feed = MultiSymbolDataFeed(
            db_path=self.db_path,
            symbols=self.symbols,
            interval=self.interval,
            exchange=self.exchange_name,
            start_ts=self.start_ts,
            end_ts=self.end_ts,
        )

        for _ts, bars in feed:
            for bar in bars.values():
                sim.on_new_bar(bar)
            newly_submitted: set[str] = set()
            for sym, bar in bars.items():
                self._process_symbol(sym, bar, states[sym], sim, newly_submitted)

        trades = sim.get_trades()

        # 去重净值曲线：on_new_bar 每 symbol 记一条，保留同 ts 最后一条
        seen: dict[int, float] = {}
        for ts, eq in sim.get_equity_curve():
            seen[ts] = eq
        equity_curve = sorted(seen.items())

        entries_filled = sum(
            1 for t in trades
            if not t.order_id.startswith("bracket") and t.order_id != "liquidation" and t.pnl == 0.0
        )
        miss_rate = (
            1.0 - entries_filled / self._entries_submitted
            if self._entries_submitted > 0 else 0.0
        )

        return {
            "trades": trades,
            "trades_count": len(trades),
            "equity_curve": equity_curve,
            "final_equity": sim.equity,
            "initial_balance": self.balance,
            "portfolio_symbols": self.symbols,
            "per_symbol_stats": self._per_symbol_stats(trades),
            "entries_submitted": self._entries_submitted,
            "entries_filled": entries_filled,
            "miss_rate": miss_rate,
        }

    # ── 单币处理 ────────────────────────────────────────────────
    def _process_symbol(
        self, sym: str, bar: Bar, state: dict, sim: SimExchange, newly_submitted: set[str]
    ) -> None:
        direction = bar_direction(bar.open, bar.close)
        state["count"], state["streak_dir"] = update_streak(
            state["count"], state["streak_dir"], direction
        )

        pos = sim._positions.get(sym)

        # 1) 持仓存在但尚无 bracket（本 bar 刚成交）→ 补挂 bracket
        if pos is not None:
            if sym not in sim._brackets:
                tp, sl = bracket_prices(
                    pos.entry_price, pos.side, self.params.tp_pct, self.params.sl_pct
                )
                expire_ts = bar.timestamp + self.params.time_stop_bars * self._interval_ms
                sim.set_bracket(sym, tp, sl, expire_ts)
            return  # 一币一仓，不叠加

        # 2) 空仓但有仍在有效期内的入场挂单 → 等待
        if state["pending_expire_ts"] and bar.timestamp < state["pending_expire_ts"]:
            return
        state["pending_expire_ts"] = 0  # 挂单已成交或已被交易所撤销

        # 3) 新信号？
        if direction == 0 or state["count"] < self.params.threshold:
            return
        if not self._account_allows_new_slot(sim, newly_submitted):
            return

        # 4) 反手侧挂 maker 限价
        side = fade_side(state["streak_dir"])
        px = entry_price(bar.close, state["streak_dir"], self.params.entry_offset_pct)
        notional = position_notional(sim.equity, self.params.risk_pct, self.params.sl_pct)
        if notional <= 0:
            return
        expire_ts = bar.timestamp + self.params.entry_ttl_bars * self._interval_ms
        sim.submit_order(sym, side, "limit", notional, price=px, expire_ts=expire_ts)
        state["pending_expire_ts"] = expire_ts
        newly_submitted.add(sym)
        self._entries_submitted += 1

    def _account_allows_new_slot(self, sim: SimExchange, newly_submitted: set[str]) -> bool:
        occupied = (
            set(sim._positions.keys())
            | {o.symbol for o in sim._pending_orders}
            | newly_submitted
        )
        if len(occupied) >= self.max_position_symbols:
            return False
        equity = sim.equity
        if equity > 0:
            total_margin = sum(p.margin for p in sim._positions.values())
            if total_margin / equity > self.margin_ratio_limit:
                return False
        return True

    # ── 每币统计 ────────────────────────────────────────────────
    def _per_symbol_stats(self, trades: list[Trade]) -> dict:
        gross_pnl: dict[str, float] = defaultdict(float)
        commissions: dict[str, float] = defaultdict(float)
        closes: dict[str, list[float]] = defaultdict(list)

        for t in trades:
            commissions[t.symbol] += t.commission
            if t.pnl != 0.0:
                gross_pnl[t.symbol] += t.pnl
                closes[t.symbol].append(t.pnl)

        stats = {}
        for sym in self.symbols:
            pnls = closes[sym]
            total = len(pnls)
            wins = sum(1 for p in pnls if p > 0)
            stats[sym] = {
                "total_trades": total,
                "win_rate": wins / total if total > 0 else 0.0,
                "net_pnl": gross_pnl[sym] - commissions[sym],
            }
        return stats
```

- [ ] **Step 4: 运行确认通过**

Run: `pytest tests/test_mean_reversion_engine.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/backtest/mean_reversion_engine.py tests/test_mean_reversion_engine.py
git commit -m "feat: add MeanReversionPortfolioEngine"
```

---

## Task 7: CLI 子命令 mr-portfolio-run

**Files:**
- Modify: `src/backtest/__main__.py`（新增 `cmd_mr_portfolio_run`、parser、dispatch）
- Test: 手动 smoke（无独立单测，复用引擎测试覆盖逻辑）

**Interfaces:**
- Consumes: `MeanReversionPortfolioEngine`、`MRParams`、`Reporter`。
- Produces: `python -m backtest mr-portfolio-run --symbols ... --interval 1h --exchange okx --start ... --end ...`，参数：`--balance --leverage --threshold --tp-pct --sl-pct --time-stop-bars --risk-pct --entry-offset-pct --entry-ttl-bars --maker-rate --taker-rate --max-position-symbols --margin-ratio-limit --db`。打印组合指标 + 每币统计 + 漏单率，并写入 `reports.db`（strategy 名 `MeanReversionPortfolio`）。

- [ ] **Step 1: 新增命令处理函数**

在 `src/backtest/__main__.py` 中（`cmd_portfolio_run` 之后）新增：

```python
def cmd_mr_portfolio_run(args: argparse.Namespace) -> None:
    import json
    import sqlite3
    from backtest.mean_reversion import MRParams
    from backtest.mean_reversion_engine import MeanReversionPortfolioEngine
    from backtest.reporter import Reporter

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    db_path = args.db or str(Path("data") / "klines.db")

    params = MRParams(
        threshold=args.threshold,
        tp_pct=args.tp_pct,
        sl_pct=args.sl_pct,
        time_stop_bars=args.time_stop_bars,
        risk_pct=args.risk_pct,
        entry_offset_pct=args.entry_offset_pct,
        entry_ttl_bars=args.entry_ttl_bars,
    )

    engine = MeanReversionPortfolioEngine(
        db_path=db_path,
        symbols=symbols,
        interval=args.interval,
        exchange=args.exchange,
        params=params,
        balance=args.balance,
        leverage=args.leverage,
        maker_commission_rate=args.maker_rate,
        taker_commission_rate=args.taker_rate,
        max_position_symbols=args.max_position_symbols,
        margin_ratio_limit=args.margin_ratio_limit,
        start=f"{args.start} 00:00:00" if args.start else None,
        end=f"{args.end} 23:59:59" if args.end else None,
    )

    print(
        f"[均值回归组合] 标的={len(symbols)} 区间={args.start}~{args.end} "
        f"余额={args.balance} 杠杆={args.leverage}x "
        f"threshold={args.threshold} tp={args.tp_pct} sl={args.sl_pct} "
        f"H={args.time_stop_bars} risk={args.risk_pct}"
    )

    result = engine.run()
    report = Reporter.generate(result)
    report["portfolio_symbols"] = symbols
    report["per_symbol_stats"] = result["per_symbol_stats"]
    report["miss_rate"] = result["miss_rate"]
    report["entries_submitted"] = result["entries_submitted"]
    report["entries_filled"] = result["entries_filled"]
    report["run_leverage"] = args.leverage
    report["run_balance"] = args.balance
    report["run_start"] = args.start
    report["run_end"] = args.end

    print("\n=== 组合指标 ===")
    print(f"  net_return:       {report['net_return']:+.2%}")
    print(f"  annual_return:    {report['annual_return']:+.2%}")
    print(f"  max_drawdown:     {report['max_drawdown']:.2%}")
    print(f"  sharpe_ratio:     {report['sharpe_ratio']:.2f}")
    print(f"  sortino_ratio:    {report['sortino_ratio']:.2f}")
    print(f"  win_rate:         {report['win_rate']:.2%}")
    print(f"  profit_factor:    {report['profit_factor']:.2f}")
    print(f"  total_trades:     {report['total_trades']}")
    print(f"  total_commission: {report['total_commission']:.2f} USDT")
    print(f"  entries_submitted:{result['entries_submitted']}")
    print(f"  entries_filled:   {result['entries_filled']}")
    print(f"  miss_rate:        {result['miss_rate']:.1%}")

    print("\n=== 各标的统计 ===")
    for sym, stats in result["per_symbol_stats"].items():
        print(
            f"  {sym:12s}  交易={stats['total_trades']:>4}  "
            f"胜率={stats['win_rate']:.1%}  净PnL={stats['net_pnl']:+.2f} USDT"
        )

    report_db = str(Path(db_path).parent / "reports.db")
    conn = sqlite3.connect(report_db)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy TEXT, symbol TEXT, interval TEXT,
            created_at TEXT, report_json TEXT
        )
    """)
    cur = conn.execute(
        "INSERT INTO reports (strategy, symbol, interval, created_at, report_json) "
        "VALUES (?,?,?,?,?)",
        ("MeanReversionPortfolio", ",".join(symbols), args.interval,
         datetime.now(timezone.utc).isoformat(), json.dumps(report)),
    )
    conn.commit()
    report_id = cur.lastrowid
    conn.close()

    print(f"\n报告已存入 reports.db (id={report_id})")
    print("查看: python -m backtest web")
```

- [ ] **Step 2: 注册 parser 与 dispatch**

在 `main()` 里 `p_portfolio` 定义块之后新增 parser：

```python
    p_mr = sub.add_parser("mr-portfolio-run",
                          help="Run mean-reversion portfolio backtest (maker entry + TP/SL bracket)")
    p_mr.add_argument("--symbols", required=True,
                      help="Comma-separated symbols e.g. BTCUSDT,ETHUSDT")
    p_mr.add_argument("--interval", required=True)
    p_mr.add_argument("--exchange", default="okx")
    p_mr.add_argument("--start", required=True, help="YYYY-MM-DD")
    p_mr.add_argument("--end", required=True, help="YYYY-MM-DD")
    p_mr.add_argument("--balance", type=float, default=10000.0)
    p_mr.add_argument("--leverage", type=int, default=10)
    p_mr.add_argument("--threshold", type=int, default=4)
    p_mr.add_argument("--tp-pct", type=float, default=0.006, dest="tp_pct")
    p_mr.add_argument("--sl-pct", type=float, default=0.006, dest="sl_pct")
    p_mr.add_argument("--time-stop-bars", type=int, default=12, dest="time_stop_bars")
    p_mr.add_argument("--risk-pct", type=float, default=0.005, dest="risk_pct")
    p_mr.add_argument("--entry-offset-pct", type=float, default=0.001, dest="entry_offset_pct")
    p_mr.add_argument("--entry-ttl-bars", type=int, default=2, dest="entry_ttl_bars")
    p_mr.add_argument("--maker-rate", type=float, default=0.0002, dest="maker_rate")
    p_mr.add_argument("--taker-rate", type=float, default=0.0005, dest="taker_rate")
    p_mr.add_argument("--max-position-symbols", type=int, default=10, dest="max_position_symbols")
    p_mr.add_argument("--margin-ratio-limit", type=float, default=0.8, dest="margin_ratio_limit")
    p_mr.add_argument("--db", default=None)
```

在 dispatch 分支（`elif args.command == "portfolio-run":` 之后）新增：

```python
    elif args.command == "mr-portfolio-run":
        cmd_mr_portfolio_run(args)
```

- [ ] **Step 3: Smoke（帮助信息）**

Run: `python -m backtest mr-portfolio-run --help`
Expected: 打印子命令帮助，含 `--threshold --tp-pct --sl-pct` 等参数，无异常。

- [ ] **Step 4: Smoke（现有 4 币真实数据快跑）**

Run:
```bash
python -m backtest mr-portfolio-run \
    --symbols BTCUSDT,ETHUSDT,SOLUSDT,DOGEUSDT \
    --interval 1h --exchange okx \
    --start 2024-01-01 --end 2024-06-30 \
    --balance 10000 --leverage 10 --threshold 4
```
Expected: 正常打印组合指标、每币统计、`miss_rate`，并存 `reports.db`。（数值不做断言，只验证端到端不报错。）

- [ ] **Step 5: 全量回归**

Run: `pytest tests/ -q`
Expected: 全部 PASS。

- [ ] **Step 6: Commit**

```bash
git add src/backtest/__main__.py
git commit -m "feat: add mr-portfolio-run CLI subcommand"
```

---

## Task 8: 采集缺失币种数据（WLD/XRP/PEPE/AAVE/UNI/BNB）

**Files:**
- 无代码改动；使用现有 `collect` 子命令写入 `data/klines.db`。

**Interfaces:**
- Consumes: `python -m backtest collect --exchange okx ...`（OKX collector 已有）。
- Produces: `data/klines.db` 中 6 个新 symbol 的 `okx / 1h` 数据。

- [ ] **Step 1: 采集 6 个币（1h，OKX）**

逐个执行（若某币上市较晚，OKX 会自动只返回可用区间；起始日给足够早即可）：

```bash
for SYM in WLDUSDT XRPUSDT PEPEUSDT AAVEUSDT UNIUSDT BNBUSDT; do
  python -m backtest collect --exchange okx --symbol "$SYM" --interval 1h \
      --start 2022-01-01 --end 2026-08-15
done
```

- [ ] **Step 2: 校验各币行数与时间跨度**

Run:
```bash
python - <<'PY'
import sqlite3
conn = sqlite3.connect("data/klines.db")
for sym in ["WLDUSDT","XRPUSDT","PEPEUSDT","AAVEUSDT","UNIUSDT","BNBUSDT"]:
    row = conn.execute(
        "SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM klines "
        "WHERE exchange='okx' AND symbol=? AND interval='1h'", (sym,)
    ).fetchone()
    n, lo, hi = row
    import datetime as dt
    def d(ms): return dt.datetime.utcfromtimestamp(ms/1000).date() if ms else None
    print(f"{sym:10s} rows={n:>7}  {d(lo)} → {d(hi)}")
conn.close()
PY
```
Expected: 每币 `rows > 0`，且时间跨度覆盖到 2026-08 附近；至少 4 个币应有 ≥18 个月历史（符合币池规则）。若某币行数为 0，重试该币的 collect（可能网络/符号命名问题，OKX 永续常用 `<BASE>-USDT-SWAP`，确认 collector 的符号映射）。

- [ ] **Step 3: （无需 commit 代码）记录采集结果**

数据文件 `data/klines.db` 通常不入库（检查 `.gitignore`）。仅在提交说明中记录采集完成，无需 `git add` 数据库。

---

## Task 9: IS/OOS 验证脚本与验收红线

**Files:**
- Create: `scripts/mr_validate.py`
- Test: `tests/test_mr_validate.py`

**Interfaces:**
- Consumes: `MeanReversionPortfolioEngine`、`MRParams`、`Reporter`。
- Produces:
  - `evaluate_gate(report: dict) -> tuple[bool, list[str]]` — 输入含 `net_return, sharpe_ratio, max_drawdown, trades`（trades 为 dict 列表，含 `pnl`/`commission`）的报告；返回 `(是否全过, 失败原因列表)`。红线：`net_return>0`、`sharpe_ratio>0`、平均单笔净收益（∑pnl−∑commission）/收盘笔数 `>0`、`max_drawdown<0.25`。
  - `run_window(db_path, symbols, exchange, interval, start, end, params, balance, leverage) -> dict` — 跑一段窗口，返回 `Reporter.generate` 结果并补 `miss_rate`。
  - `main()` — 跑 IS（2022-01-01→2024-12-31）与 OOS（2025-01-01→2026-07-31），打印两段指标，并对 OOS 执行 `evaluate_gate`，退出码 0=通过 / 1=未过。

- [ ] **Step 1: 写失败测试（gate 纯函数）**

`tests/test_mr_validate.py`：

```python
from scripts.mr_validate import evaluate_gate


def _report(net, sharpe, mdd, trades):
    return {"net_return": net, "sharpe_ratio": sharpe,
            "max_drawdown": mdd, "trades": trades}


def _t(pnl, comm):
    return {"pnl": pnl, "commission": comm}


def test_gate_passes_when_all_green():
    trades = [_t(0.0, 0.02), _t(10.0, 0.02), _t(0.0, 0.02), _t(8.0, 0.02)]
    ok, reasons = evaluate_gate(_report(0.05, 0.8, 0.10, trades))
    assert ok is True
    assert reasons == []


def test_gate_fails_on_negative_net_return():
    trades = [_t(0.0, 0.02), _t(-5.0, 0.02)]
    ok, reasons = evaluate_gate(_report(-0.02, 0.5, 0.10, trades))
    assert ok is False
    assert any("net_return" in r for r in reasons)


def test_gate_fails_on_high_drawdown():
    trades = [_t(0.0, 0.02), _t(10.0, 0.02)]
    ok, reasons = evaluate_gate(_report(0.05, 0.8, 0.30, trades))
    assert ok is False
    assert any("max_drawdown" in r for r in reasons)


def test_gate_fails_on_nonpositive_avg_net_trade():
    # 毛利被手续费吃光：两笔收盘各 +0.01，佣金合计更大
    trades = [_t(0.0, 5.0), _t(0.01, 5.0), _t(0.0, 5.0), _t(0.01, 5.0)]
    ok, reasons = evaluate_gate(_report(0.05, 0.8, 0.10, trades))
    assert ok is False
    assert any("avg_net_trade" in r for r in reasons)
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_mr_validate.py -v`
Expected: FAIL —`ModuleNotFoundError: No module named 'scripts.mr_validate'`（或 `scripts` 无 `__init__`）。

- [ ] **Step 3: 最小实现**

先确保 `scripts/` 可被 import（若无则新建空文件 `scripts/__init__.py`）。

`scripts/mr_validate.py`：

```python
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
```

- [ ] **Step 4: 运行确认通过**

Run: `pytest tests/test_mr_validate.py -v`
Expected: PASS

- [ ] **Step 5: 跑一次真实 IS/OOS（数据就绪后）**

Run: `python scripts/mr_validate.py`
Expected: 打印 IS/OOS 指标与红线结果；退出码反映是否通过。（此步是策略评估，非通过/失败断言——若 OOS 未过，按 spec §8 升级到备选方案 C，不上线。）

- [ ] **Step 6: Commit**

```bash
git add scripts/mr_validate.py tests/test_mr_validate.py
[ -f scripts/__init__.py ] && git add scripts/__init__.py
git commit -m "feat: add IS/OOS validation harness with acceptance gate"
```

---

## Self-Review

**1. Spec coverage（对照 spec 各节）：**
- §3 币池 → Task 8 采集 6 币 + Task 9 `DEFAULT_SYMBOLS`。✓
- §4 信号（方向/连续/反手/参数全币共享）→ Task 5 `bar_direction`/`update_streak`/`fade_side` + `MRParams` 单套。✓
- §5 maker 入场 + 限价有效期 + 漏单建模 → Task 3 expiry + Task 5 `entry_price` + Task 6 `miss_rate`。✓
- §6 bracket 出场（TP maker / SL taker / 时间止损 taker，SL 先于 TP，非对称费）→ Task 2 + Task 4。✓
- §7 固定风险仓位 + 一币一仓 + 并发上限 + 保证金上限 → Task 5 `position_notional` + Task 6 `_account_allows_new_slot`。✓
- §8 IS/OOS + 4 条红线 + 报告指标（含漏单率、每币贡献）→ Task 9 `evaluate_gate`/`run_window` + Task 6 `per_symbol_stats`/`miss_rate`。✓
- §9 代码改动范围（新策略逻辑、不改老引擎、数据采集、验证脚本）→ Task 4–9；显式不改 `consecutive_reverse.py` 与 `portfolio_engine.py`（Global Constraints）。✓
- §10 1h + TP/SL bracket → 默认 interval=1h、出场仅 bracket。✓
- §11 风险披露 → 保留在 spec；验收红线为落地闸门。✓
- **缺口**：spec §7「总保证金占用上限」已由 `margin_ratio_limit` 覆盖；并发币数上限由 `max_position_symbols` 覆盖。无未覆盖项。

**2. Placeholder scan：** 每个实现步均含完整代码；无 TBD/TODO/"handle edge cases" 之类占位。✓

**3. Type consistency：**
- `submit_order(..., price=None, expire_ts=0)` 在 Task 3 定义，Task 6 以 `price=px, expire_ts=expire_ts` 调用。✓
- `set_bracket(symbol, tp_price, sl_price, expire_ts)` 在 Task 4 定义，Task 6 以 `sim.set_bracket(sym, tp, sl, expire_ts)` 调用。✓
- `bracket_prices(entry, position_side, tp_pct, sl_pct)` 返回 `(tp, sl)`，Task 6 用 `pos.side`（"long"/"short"）传入，一致。✓
- `MRParams` 字段名（threshold/tp_pct/sl_pct/time_stop_bars/risk_pct/entry_offset_pct/entry_ttl_bars）在 Task 5 定义，Task 6/7/9 一致引用。✓
- bracket 平仓 `order_id` 取值 `bracket-tp/bracket-sl/bracket-time`，Task 4 测试与 Task 6 `entries_filled` 过滤（`startswith("bracket")`）一致。✓
- 引擎 `run()` 返回键与 Task 6 测试、Task 7 CLI、Task 9 harness 使用的键一致。✓
