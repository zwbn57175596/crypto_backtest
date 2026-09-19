"""OKX implementation of the carry Exchange protocol (spot + USDT-swap, net mode).

The four API objects are injected python-okx-shaped clients, so every path is unit-testable
with fakes. `clock()` returns seconds. Reads go through a short cache that writes clear.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Callable

from carry.types import DEFAULT_INSTRUMENTS, Instrument, OrderResult

TRANSIENT_CODES = frozenset({"50001", "50004", "50011", "50013", "50026"})   # verify on demo
# UNVERIFIED on demo: OKX duplicate-clOrdId and order-not-found sCodes/codes. Isolated here so the
# demo run can correct them; the adapter never relies on them for safety (it probes get_order first).
DUPLICATE_CLORDID_CODES = frozenset({"51016"})
NOT_FOUND_CODES = frozenset({"51603"})
FUNDING_REFETCH_MIN_SEC = 60.0
TERMINAL_STATES = frozenset({"filled", "canceled", "mmp_canceled"})
SETTLEMENT_MS = 8 * 3600 * 1000


class ExchangeError(RuntimeError):
    def __init__(self, msg: str, code: str | None = None) -> None:
        super().__init__(msg)
        self.code = code


def _f(x: Any) -> float:
    if x is None or x == "":
        return 0.0
    v = float(x)
    if not math.isfinite(v):
        raise ExchangeError(f"non-finite number {x!r}")
    return v


def _fmt(x: float) -> str:
    return format(Decimal(str(round(x, 12))).normalize(), "f")


@dataclass
class _FundingCache:
    rates: list[float]
    limit: int
    expires_ms: float
    fetched_at: float = 0.0


class OkxExchange:
    def __init__(self, trade: Any, account: Any, public: Any, market: Any,
                 insts: dict[str, Instrument] | None = None, *, spot_td_mode: str = "cash",
                 max_basis: float = 0.01, max_tick_age_sec: float = 30.0, order_timeout_sec: float = 10.0,
                 poll_sec: float = 0.5, retries: int = 3, backoff_sec: float = 0.5, read_ttl_sec: float = 1.0,
                 clock: Callable[[], float] = time.time, sleep: Callable[[float], None] = time.sleep,
                 alert: Callable[[str], None] | None = None) -> None:
        if retries < 1:
            raise ValueError("retries must be >= 1")
        self.trade, self.account, self.public, self.market = trade, account, public, market
        self.insts = insts or DEFAULT_INSTRUMENTS
        self.spot_td_mode, self.max_basis, self.max_tick_age_sec = spot_td_mode, max_basis, max_tick_age_sec
        self.order_timeout_sec, self.poll_sec, self.retries = order_timeout_sec, poll_sec, retries
        self.backoff_sec, self.read_ttl_sec, self.clock, self._sleep = backoff_sec, read_ttl_sec, clock, sleep
        self._funding: dict[str, _FundingCache] = {}
        self._reads: dict[tuple, tuple[float, Any]] = {}
        self._seq = 0
        self._alert_fn = alert
        self._unconfirmed: dict[str, str] = {}       # clOrdId -> instId of orders not proven terminal
        self._blocked_alerted: frozenset[str] | None = None

    def _alert(self, msg: str) -> None:
        if self._alert_fn is None:
            return
        try:
            self._alert_fn(msg)
        except Exception:  # noqa: BLE001 - an alert sink must never break the order path
            pass

    # --- plumbing ----------------------------------------------------------
    def _call(self, fn: Callable[..., dict], what: str, **kw: Any) -> dict:
        last: ExchangeError | None = None
        for attempt in range(self.retries):
            try:
                resp = fn(**kw)
            except Exception as e:  # noqa: BLE001 - network/timeouts are retried
                last = ExchangeError(f"{what}: {type(e).__name__}: {e}")
            else:
                code = str(resp.get("code", ""))
                if code == "0":
                    return resp
                last = ExchangeError(f"{what}: code={code} msg={resp.get('msg', '')}", code=code)
                if code not in TRANSIENT_CODES:
                    raise last
            if attempt < self.retries - 1:
                self._sleep(self.backoff_sec * (2 ** attempt))
        if last is None:
            raise ExchangeError(f"{what}: no attempt made")
        raise last

    def _cached(self, key: tuple, fn: Callable[[], Any]) -> Any:
        now = self.clock()
        hit = self._reads.get(key)
        if hit is not None and now - hit[0] < self.read_ttl_sec:
            return hit[1]
        val = fn()
        self._reads[key] = (now, val)
        return val

    # --- reads -------------------------------------------------------------
    def _ticker(self, inst_id: str) -> float:
        resp = self._call(self.market.get_ticker, f"ticker {inst_id}", instId=inst_id)
        try:
            d = resp["data"][0]
            last, ts = _f(d["last"]), int(d["ts"])
        except ExchangeError:
            raise
        except (KeyError, IndexError, TypeError, ValueError) as e:
            raise ExchangeError(f"{inst_id}: malformed ticker payload ({type(e).__name__}: {e})") from e
        if abs(self.clock() * 1000 - ts) > self.max_tick_age_sec * 1000:
            raise ExchangeError(f"{inst_id}: stale ticker (age {abs(self.clock() * 1000 - ts) / 1000:.0f}s)")
        if not last > 0:
            raise ExchangeError(f"{inst_id}: bad price {last}")
        return last

    def price(self, coin: str) -> float:
        inst = self.insts[coin]
        spot, swap = self._ticker(inst.spot_id), self._ticker(inst.swap_id)
        if abs(spot - swap) / spot > self.max_basis:
            raise ExchangeError(f"{coin}: spot {spot} and swap {swap} diverge by more than {self.max_basis:.1%}")
        return spot

    def funding_history(self, coin: str, n: int) -> list[float]:
        if n <= 0:
            return []
        if n > 100:
            raise ExchangeError("funding_history: n must be <= 100")
        c = self._funding.get(coin)
        now = self.clock()
        stale = c is not None and now * 1000 >= c.expires_ms and now - c.fetched_at >= FUNDING_REFETCH_MIN_SEC
        if c is None or c.limit < n or stale:
            rows = self._call(self.public.funding_rate_history, f"funding history {coin}",
                              instId=self.insts[coin].swap_id, limit=str(n))["data"]
            # fundingRate is the announced rate; the demo run decides whether `realizedRate` is the truth.
            pts = sorted((int(r["fundingTime"]), _f(r["fundingRate"])) for r in rows)
            if not pts:
                raise ExchangeError(f"{coin}: no funding history returned")
            c = _FundingCache([r for _, r in pts], n, pts[-1][0] + SETTLEMENT_MS + 60_000, now)
            self._funding[coin] = c
        return c.rates[-n:]

    def spot_balance(self, coin: str) -> float:
        def fetch() -> float:
            data = self._call(self.account.get_account_balance, f"balance {coin}", ccy=coin)["data"]
            for det in (data[0].get("details", []) if data else []):
                if det.get("ccy") == coin:
                    return _f(det.get("cashBal"))
            return 0.0
        return self._cached(("bal", coin), fetch)

    def _positions(self, coin: str) -> list[dict]:
        inst_id = self.insts[coin].swap_id

        def fetch() -> list[dict]:
            rows = self._call(self.account.get_positions, f"positions {coin}",
                              instType="SWAP", instId=inst_id)["data"]
            return [r for r in rows if r.get("instId") == inst_id]
        return self._cached(("pos", coin), fetch)

    def swap_contracts(self, coin: str) -> float:
        total = 0.0
        for r in self._positions(coin):
            p, side = _f(r.get("pos")), r.get("posSide", "net")
            if side == "short":
                total += abs(p)
            elif side == "long":
                if p != 0:
                    raise ExchangeError(f"{coin}: unexpected LONG swap position {p}")
            else:                                                 # net mode: negative = short
                if p > 0:
                    raise ExchangeError(f"{coin}: unexpected LONG swap position {p}")
                total += -p
        return total

    def swap_adl_rank(self, coin: str) -> int:
        best = 0
        for r in self._positions(coin):
            try:
                best = max(best, int(r.get("adl") or 0))
            except ValueError:
                pass
        return best

    def mgn_ratio(self) -> float | None:
        def fetch() -> float | None:
            data = self._call(self.account.get_account_balance, "account balance")["data"]
            raw = data[0].get("mgnRatio") if data else None
            if raw in (None, ""):
                return None
            try:
                v = float(raw)
            except (ValueError, TypeError):
                return None
            return v if math.isfinite(v) else None
        return self._cached(("mgn",), fetch)

    # --- writes (market orders, idempotent by clOrdId) ------------------------
    def _next_cl_ord_id(self) -> str:
        self._seq += 1
        return f"carry{int(self.clock() * 1000)}{self._seq % 1000:03d}"

    def _place(self, inst_id: str, side: str, sz: float, td_mode: str, *,
               reduce_only: bool = False, base_target: bool = False) -> OrderResult:
        if not math.isfinite(sz) or sz <= 0 or float(_fmt(sz)) <= 0:
            return OrderResult(False, error="invalid size")
        pending = self._unconfirmed_pending()
        if pending:
            key = frozenset(pending)
            if key != self._blocked_alerted:
                self._blocked_alerted = key
                self._alert(f"[CRITICAL] order placement BLOCKED: unconfirmed order(s) {sorted(pending)}")
            return OrderResult(False, error=f"BLOCKED: unconfirmed order(s) {sorted(pending)}")
        self._reads.clear()
        cl = self._next_cl_ord_id()
        kw: dict[str, Any] = dict(instId=inst_id, tdMode=td_mode, side=side, ordType="market",
                                  sz=_fmt(sz), clOrdId=cl)
        if reduce_only:
            kw["reduceOnly"] = "true"
        if base_target:
            kw["tgtCcy"] = "base_ccy"
        ambiguous = ""
        for attempt in range(self.retries):
            state, msg = self._post(kw)
            if state == "rejected":
                return OrderResult(False, error=msg)
            if state == "accepted":
                ambiguous = ""
                break
            ambiguous = msg
            exists = self._probe(inst_id, cl)         # never re-POST before proving the order absent
            if exists is not False:
                break                                  # found, or unknown: poll it (unconfirmed protocol)
            if attempt == self.retries - 1:
                return OrderResult(False, error=msg)
            self._sleep(self.backoff_sec * (2 ** attempt))
        res = self._wait_terminal(inst_id, cl)
        self._reads.clear()
        if ambiguous and not res.ok:
            if res.error.startswith("UNCONFIRMED"):
                return OrderResult(False, res.filled, res.avg_price, f"{res.error} (placement: {ambiguous})")
            return OrderResult(False, res.filled, res.avg_price, f"{ambiguous}; {res.error}")
        return res

    def _post(self, kw: dict[str, Any]) -> tuple[str, str]:
        """One placement attempt -> ("accepted"|"rejected"|"ambiguous", text). Only a row-level
        sCode may produce "rejected"; everything else that is not a clean success is ambiguous."""
        try:
            resp = self.trade.place_order(**kw)
        except Exception as e:  # noqa: BLE001
            return "ambiguous", f"place_order: {type(e).__name__}: {e}"
        try:
            code, msg = str(resp.get("code", "")), str(resp.get("msg", ""))
            rows = resp.get("data") or []
            row = rows[0] if rows and isinstance(rows[0], dict) else None
        except Exception as e:  # noqa: BLE001 - malformed envelope
            return "ambiguous", f"place_order: malformed response ({type(e).__name__}: {e})"
        scode = str(row.get("sCode", "")) if row is not None else ""
        if scode not in ("", "0"):
            if scode in DUPLICATE_CLORDID_CODES:
                return "ambiguous", f"place_order: duplicate clOrdId sCode={scode}"   # the order exists
            return "rejected", f"sCode={scode} {row.get('sMsg', '')}".strip()
        if code == "0":
            return "accepted", ""
        return "ambiguous", f"place_order: code={code} msg={msg}"

    def _probe(self, inst_id: str, cl: str) -> bool | None:
        """Does the order exist? True / False (proven absent) / None (cannot tell)."""
        try:
            data = self._call(self.trade.get_order, "get_order", instId=inst_id, clOrdId=cl).get("data")
        except ExchangeError as e:
            return False if e.code in NOT_FOUND_CODES else None
        return bool(data)

    def _read_order(self, inst_id: str, cl: str) -> dict | None:
        try:
            data = self._call(self.trade.get_order, "get_order", instId=inst_id, clOrdId=cl)["data"]
            return data[0] if data else None
        except (ExchangeError, KeyError, IndexError, TypeError):
            return None                                          # not found yet / transient

    def _cancel_and_confirm(self, inst_id: str, cl: str) -> dict | None:
        for _ in range(3):
            try:
                self._call(self.trade.cancel_order, "cancel_order", instId=inst_id, clOrdId=cl)
            except ExchangeError:
                pass
            row = self._read_order(inst_id, cl)
            if row is not None and row.get("state") in TERMINAL_STATES:
                return row
            self._sleep(self.poll_sec)
        return None

    def _wait_terminal(self, inst_id: str, cl: str) -> OrderResult:
        deadline = self.clock() + self.order_timeout_sec
        last: dict | None = None
        while True:
            row = self._read_order(inst_id, cl)
            if row is not None:
                last = row
                if row.get("state") in TERMINAL_STATES:
                    break
            if self.clock() >= deadline:
                break
            self._sleep(self.poll_sec)
        if last is None or last.get("state") not in TERMINAL_STATES:
            confirmed = self._cancel_and_confirm(inst_id, cl)
            if confirmed is None:
                self._unconfirmed[cl] = inst_id
                self._alert(f"[CRITICAL] order {cl} on {inst_id} could not be confirmed terminal; "
                            "new orders are BLOCKED until it resolves")
                try:
                    filled = _f(last.get("accFillSz")) if last else 0.0
                except (ExchangeError, ValueError, TypeError):
                    filled = 0.0
                return OrderResult(False, filled, 0.0,
                                   "UNCONFIRMED: status unknown after cancel attempts; order may still be live")
            last = confirmed
        state = str(last.get("state"))
        good = state == "filled"
        return OrderResult(good, _f(last.get("accFillSz")), _f(last.get("avgPx")),
                           "" if good else f"state={state}")

    def _unconfirmed_pending(self) -> list[str]:
        for cl, inst_id in list(self._unconfirmed.items()):
            row = self._read_order(inst_id, cl)
            if row is not None and row.get("state") in TERMINAL_STATES:
                del self._unconfirmed[cl]
        if not self._unconfirmed:
            self._blocked_alerted = None
        return list(self._unconfirmed)

    def clear_unconfirmed(self) -> None:
        """Operator override: forget unconfirmed orders (the operator has verified them by hand)."""
        self._unconfirmed.clear()
        self._blocked_alerted = None

    def sell_swap(self, coin: str, contracts: float) -> OrderResult:
        return self._place(self.insts[coin].swap_id, "sell", contracts, "cross")

    def buy_swap(self, coin: str, contracts: float) -> OrderResult:
        return self._place(self.insts[coin].swap_id, "buy", contracts, "cross", reduce_only=True)

    def buy_spot(self, coin: str, qty: float) -> OrderResult:
        return self._place(self.insts[coin].spot_id, "buy", qty, self.spot_td_mode, base_target=True)

    def sell_spot(self, coin: str, qty: float) -> OrderResult:
        return self._place(self.insts[coin].spot_id, "sell", qty, self.spot_td_mode, base_target=True)
