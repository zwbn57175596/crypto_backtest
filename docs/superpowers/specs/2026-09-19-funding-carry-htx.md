# Funding carry on HTX (USDT-M linear swap) vs Binance / OKX

**Method / sources.** Public GETs only. Data: `https://api.hbdm.com/linear-swap-api/v1/swap_historical_funding_rate` (BTC-USDT, ETH-USDT; cached in `.superpowers/analysis/htx/`), `swap_contract_info`, `swap_funding_rate`; Binance from `data/carry.db` (read-only). Scripts `dl.py`, `an.py`; outputs `an.out`, `an_aligned.out`. Official pages (htx.com/support/...): 900001326466 (funding calc), 900000089923 (swap fees), 64914554002695 (ADL), 45000308207138 + 95018034388943 (multi-assets collateral), 84919709792809 / 34945253379665 (unified account, API), 44912606476945 (margin loan rates, 2022), 360000312282 (spot tiers, 2019); API docs huobiapi.github.io/docs/usdt_swap/v1/en/. APR = rate x 1095.

## 1. Funding data and mechanics
- Endpoint above: **6476 rows per symbol, 2020-10-21 16:00 UTC -> 2026-09-19** (full contract life, ~5.9y; no depth problem, unlike OKX's 3 months). Paging: docs say page_size <= 50, the server accepted 100 (65 pages). Interval 8h (settle 00/08/16 UTC); 18 irregular gaps (3-11h, 2021-2023, 2026-01-15 4h) out of 6476. `realized_rate` is null in every row; docs (FAQ Q14) say it equals `funding_rate` except when paying would liquidate the account (fee waived/reduced, see help page rule III).
- Formula (official): rate = clamp{ P + clamp(0.01% - P, +-0.05%) } / (8/N), then capped at BTC +-0.375% per 8h. **Identical structure and 0.01%/8h base (10.95% APR) as Binance/OKX**; premium index from impact bid/ask at 25,000 USDT (BTC/ETH), 5 s samples, time-weighted. Interval can be changed dynamically by HTX under stress (help page risk note).
- **Timestamp problem:** HTX `funding_time` labels correlate best with Binance settlement at t-8h before 2024 and at t (lag 0) from 2024 (quarterly cross-corr, an.py). I aligned pre-2024 rows one grid step back; results below use that. Yearly sums are unaffected.

## 2. HTX vs Binance (aligned, 6474 common settlements; APR %)
| | BTC all | BTC 2024+ | BTC last 365d | ETH all | ETH 2024+ | ETH last 365d |
|---|---|---|---|---|---|---|
| mean HTX / BN | 10.4 / 11.1 | 7.3 / 7.0 | 4.7 / 3.4 | 11.1 / 12.3 | 9.6 / 7.0 | 4.6 / 2.4 |
| median HTX / BN | 11.0 / 8.8 | 11.0 / 6.2 | 10.8 / 3.9 | 11.0 / 8.8 | 11.0 / 6.2 | 9.2 / 3.1 |
| stdev HTX / BN | 21.8 / 21.2 | 12.7 / 9.6 | 9.9 / 5.1 | 24.2 / 28.1 | 13.0 / 10.1 | 9.3 / 5.8 |
| corr | 0.76 | 0.63 | 0.33 | 0.71 | 0.68 | 0.05 |
| sign disagreement | 18% | 20% | 26% | 20% | 22% | 38% |
| at exact base 0.01%: HTX / BN | 51 / 32% | 51 / 24% | 49 / 8% | 54 / 30% | 56 / 23% | 46 / 6% |
Mean diff (HTX-BN): BTC -0.7, ETH -1.1 pp over full period (median 0); **+1.4 / +2.3 pp last 365d**. HTX is not systematically higher over the cycle (lower in the 2020-21 boom), slightly higher and pinned at the floor in the recent low-funding regime; more volatile than Binance since 2024 and **decorrelated from Binance in 2025-26** (ETH corr ~0), so the two venues are not interchangeable data proxies now. Worst 3-day mean: BTC -55% (BN -40%), ETH -112% (BN -140%).

## 3. Earlier rule on HTX (trailing-3d >=6% enter, <2% exit, funding only, no basis/borrow), net % notional per year
| | 2020* | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 YTD |
|---|---|---|---|---|---|---|---|
| BTC HTX @0.30% RT / BN @0.30% | 1.1 / 3.7 | 23.8 / 29.1 | 0.5 / -2.4 | 6.7 / 5.5 | 8.9 / 10.9 | 2.4 / 2.1 | -0.5 / 1.5 |
| ETH HTX @0.30% / BN @0.30% | 3.0 / 6.4 | 24.5 / 35.1 | -2.6 / -3.8 | 7.8 / 5.9 | 14.9 / 12.4 | 3.6 / 0.4 | -0.1 / 0.0 |
| BTC HTX @0.52% RT (HTX real taker) | 0.1 | 22.3 | -2.8 | 5.1 | 6.9 | -0.3 | -2.9 |
| ETH HTX @0.52% | 2.5 | 23.7 | -5.9 | 6.0 | 14.2 | 0.9 | -2.0 |
*2020 from Oct 21 only. Both venues comparable per year; 66 (BTC) / 54 (ETH) entry cycles in 5.9y. At HTX's likely costs 2025-26 is ~zero or negative; the venue does not fix the 2025+ low-funding problem.

## 4. Venue facts
- **Fees (swap):** official page's Prime 0 example: maker 0.02% / taker 0.06% (illustrative; real tier in My Fees). **Spot base:** 0.20% (2019 tier doc; current UNVERIFIED); Prime 3 = 0.10%; HT/TRX deduction gives ~25% spot / ~10% futures discount (help pages). Taker round trip ~ 2x(0.20+0.06)=**0.52%**, vs 0.30% OKX.
- **Borrow:** margin loans are dynamic, hourly interest = daily rate/24, limit by Prime level. Last published table (2022-08): USDT 0.0098%/day at prime0 = ~3.6% APR; **current rate UNVERIFIED** (public page is login-rendered; the API loan-info endpoint needs auth).
- **Collateral:** spot is a separate account. USDT-M has a Multi-Assets Collateral mode (cross only, since 2025-05): only **USDT, BTC, ETH** as collateral, 5-10% haircut, assets must be transferred to the futures account. So BTC/ETH carry = hold the coin as futures collateral against the short (no spot leg trading needed); alts not supported. Risk metric: initial margin ratio >=100% -> auto order cancel + reduce-only; maintenance margin ratio >=100% -> liquidation. In-account auto-borrow of USDT: UNVERIFIED. Unified account (cross+isolated merge, 2022) is a different thing.
- **ADL:** official page (2025-11-19): only after insurance-fund depletion; rank = PnL ratio / margin ratio (cross: account margin ratio); 5-light UI indicator; SMS/email after ADL. Public/private API field `adl_risk_percent` exists in the linear-swap SDK; not confirmed in the docs I could read (UNVERIFIED).
- **API:** REST + private WS (`positions_cross.*`, `accounts_cross.*`, `accounts_unify.USDT`), spot<->futures transfer endpoint documented; v5 `/v5/market/risk/limit` exists. Rate limit 800/s/IP REST. Two USDT-M system upgrades/migrations are announced (2026-04-03 and 09-03 titles, contents not readable) - API churn risk.
- **Contracts (API):** BTC-USDT 0.001 BTC/cont, ETH-USDT 0.01 ETH/cont (BTC ~$81k -> ~$81/cont). Min order = 1 contract: UNVERIFIED. Trading is paused during settlement.
- Counterparty/withdrawal: nothing officially documented was assessed; no view.

## 5. Verdict
| | HTX | OKX | Binance |
|---|---|---|---|
| Funding history | 5.9y, 8h, mild timestamp shift | ~3 months | 7y |
| Funding level 2024+ / last 12m (BTC) | 7.3 / 4.7% | ~ base | 7.0 / 3.4% |
| Taker RT fee | ~0.52% | 0.30% | ~0.30-0.34% (not re-verified) |
| USDT borrow | ~3.6% (2022 table, UNVERIFIED now) | ~3.5% | not researched |
| Spot-as-collateral for short | BTC/ETH only, in futures account | yes, unified acct (in use) | not researched |
| ADL transparency | 5-light UI, rules public | not researched here | not researched here |
| API | complete, migration churn | in use | not researched |

**Recommendation: not worth pursuing as the carry venue; keep only as a secondary data cross-check** (and even that is weak since 2025 correlation is 0.05-0.33). Reasons: (1) funding is not higher in any regime that matters - +1-2pp last year is inside noise and pinned at the 0.01% floor half the time; (2) fees ~0.52% vs 0.30% round trip wipe out the edge (2025-26 net ~0 or negative on the same rule); (3) collateral flexibility is limited to BTC/ETH with haircuts and a separate spot account, versus the OKX unified account already configured; (4) no changes to the strategy economics that the 2025+ low-funding regime does not already set; (5) unverified borrow rate and announced system migrations add operational risk.

## Caveats
Single 5.9y sample, one boom (2021) dominates every total; 8h-grid snapping and the manual pre-2024 one-step alignment affect correlation/sign stats but not yearly funding sums; realized_rate is null so waived-funding events are invisible; simple accrual, no basis, borrow, or liquidation. UNVERIFIED: spot base fee today, current USDT margin rate, in-account borrowing, min order, ADL API field. Binance/OKX column entries other than funding and OKX fees/borrow come from memory or earlier docs, not re-verified.
