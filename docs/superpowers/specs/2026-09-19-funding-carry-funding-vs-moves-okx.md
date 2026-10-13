# Funding vs big moves, re-done on OKX (BTC/ETH USDT-swap) — does the Binance conclusion hold?

## Step 0: how deep does OKX funding history go?
- REST `/api/v5/public/funding-rate-history` (paged with `after`): hard-capped at 281 rows = 2026-06-17 -> 2026-09-19 (~3 months). `after=2020` returns nothing.
- **Official historical-data service works and is public**: `GET /api/v5/public/market-data-history?module=3&instType=SWAP&instFamilyList=BTC-USDT|ETH-USDT&dateAggrType=monthly&begin&end` (<=10 months per call) lists monthly zips on `static.okx.com/cdn/okex/traderecords/swaprates/monthly/YYYYMM/<inst>-fundingrates-YYYY-MM.zip` (csv: funding_rate, funding_time). 57 files per symbol downloaded (114 total, tiny), stored in `.superpowers/analysis/okx/`.
- **Earliest obtainable: 2021-08-31 (BTC and ETH USDT-swap; BTC-USD coin-margined starts the same month).** Months 2021-10, -11, -12 and 2022-01-01..15 do not exist on the CDN (404), so there is a 108-day hole. Files end 2026-08; 2026-06..09 come from the API. Files and API agree exactly on the 225 overlapping rows. Total 5,209 settlements per symbol (~4.7 years, 2021-09-01 -> 2026-09-19); nothing earlier exists, so **2020-03 and 2021-05 crashes and the 2020-21 bull cannot be tested on OKX**.
- Prices: `/market/history-candles` bar=`1Dutc` (plain `1D` bars open at 00:00 UTC+8 and would misalign with funding days); 2020-01 -> 2026-09.

## Method
Same code as the Binance report (`.superpowers/analysis/an.py` adapted, `okx/an_okx.py`): day = UTC day, 3 settlements/day, APR = rate x 1095, big move = bottom/top 1.5% daily close-to-close, 7-day declustering, trailing-3d buckets, 6%/2% rule. Sample n = 1,734 complete days per symbol (14 BTC / 13 ETH crashes, 19 / 20 rallies; pooled Q3/Q4 n = 3,442 / 3,414 overlapping days, ~490 effective 7d blocks). For a like-for-like comparison I also re-ran the Binance report on **exactly the same days** (`MODE=bn`), because the original Binance study (2019+) includes 2020-21 which OKX lacks. Scripts: `an_okx.py`, `cmp.py`, `sim.py`.

## 1. OKX funding vs Binance funding (5,209 aligned settlements; mean APR %, diff = OKX - Binance)
| | BTC OKX / BN / diff | ETH OKX / BN / diff | BTC sign differs (ETH) |
|---|---|---|---|
| 2021-09 (90 stl) | 4.6 / 14.0 / -9.5 | 9.4 / 19.1 / -9.7 | 38% (32%) |
| 2022 | 1.6 / 4.0 / -2.4 | -4.2 / 0.6 / -4.8 | 35% (42%) |
| 2023 | 6.0 / 7.9 / -1.8 | 6.6 / 8.3 / -1.7 | 29% (30%) |
| 2024 | 11.5 / 11.9 / -0.4 | 12.2 / 13.0 / -0.8 | 19% (16%) |
| 2025 | 4.9 / 5.1 / -0.3 | 3.4 / 4.9 / -1.5 | 20% (25%) |
| 2026 YTD | 3.3 / 2.9 / +0.5 | 2.6 / 1.7 / +0.9 | 21% (26%) |
| last 3 months | 5.3 / 6.3 / -1.0 | 3.9 / 4.1 / -0.3 | 8% (20%) |
| all | 5.6 / 6.7 / -1.1 | 4.4 / 6.2 / -1.8 | 25% (28%) |
- Per-settlement corr 0.67 (BTC) / 0.64 (ETH); corr of daily means 0.81; corr of trailing-3d APR **0.87 / 0.88** with OKX ~ -1.2 + 1.02 x BN (BTC), -1.8 + 0.99 x BN (ETH); mean abs error of trailing-3d APR 3.3pp / 4.1pp. Level error is small since 2024 (<=1.5pp) but per-settlement noise is large (sign disagrees 16-26% of settlements; OKX negative 27% / 32% of settlements vs Binance 15% / 18%).
- **OKX funding is systematically lower, not higher**: median -0.8 / -1.1pp overall, -1 to -10pp in 2021-2023, ~0 in 2025-26. OKX was more often negative in bear/chop (ETH 2022: 57% negative settlements, mean -4.2% APR).
- **Structural difference**: Binance sits exactly on the 0.0001 base rate 28.5% / 27.1% of the time; OKX **never** did before 2025-04 (0.0%), and since then ~15-20% (4.5-5.0% of all settlements). OKX funding was a free-floating premium-based rate until then; today OKX behaves much like Binance (2025-26 means within 0.3-1.5pp).
- Proxy verdict: Binance trailing-3d funding is a decent state variable for OKX (corr 0.87) but overstates it: when Binance t3 >= 6% (49% of days) OKX t3 >= 6% only 60% (BTC) / 63% (ETH) of the time, and OKX is above 6% on 34% of days (vs 49% / 46%).

## 2. Q2 event study on OKX vs Binance (same days; funding APR mean / median; n = 14 BTC, 13 ETH crashes; 19 / 20 rallies)
| window | BTC crash OKX | BTC crash BN | ETH crash OKX | ETH crash BN | BTC rally OKX / BN | ETH rally OKX / BN |
|---|---|---|---|---|---|---|
| pre 3d | 6.3 / 5.0 | 7.1 / 6.4 | -2.6 / 1.6 | -1.2 / 4.9 | 4.8 / 0.9 vs 7.7 / 5.1 | 3.7 / 2.8 vs 4.9 / 5.1 |
| +1..7d | -1.9 / -1.7 | 1.1 / 0.7 | -5.7 / -6.6 | -0.9 / -2.4 | 11.2 / 6.9 vs 11.2 / 8.8 | 7.8 / 5.8 vs 8.3 / 7.4 |
| % events neg in +1..7 | 57 | 43 | 85 | 62 | 5 vs 5 | 30 vs 10 |
| short earned [-3d,+7d] % notional mean / min | 0.02 / -0.29 | 0.09 / -0.32 | -0.17 / -1.38 | -0.04 / -1.26 | 0.28 vs 0.31 | 0.19 vs 0.22 |
| crash windows with negative funding | 5 of 14 | 2 of 14 | 10 of 13 | 5 of 13 | | |
Crashes: funding collapses in the same way but OKX collapses **further below zero** for a week (ETH: 10 of 13 crash windows cost funding vs 5 of 13 on Binance; average -0.17% of notional, worst -1.38%, around the 2022-09-15 ETH Merge-week crash). Rallies: the same ~1.5-2x lift, both venues. Episodes are the same days (see 4). Worst single 11-day windows on OKX (-0.3% BTC, -1.4% ETH) are still small next to leg/ADL risk.

## 3. Predictive buckets and persistence (OKX, pooled BTC+ETH)
- Q3 (trailing-3d band -> forward crash/rally): on OKX the >20% band (n=202, ~29 blocks) shows **no** crash lift (P(3d<=-10%) 4.0% vs 3.5% overall; Binance same days 4.6% vs 3.5%; original Binance 2019+ study 8.2% vs 4.3%). The high-funding => higher-tail-risk effect came from 2020-21 and is not testable/present on OKX. Trailing-vol terciles still separate risk cleanly (identical prices): P(3d<=-10%) 1.9 / 3.5 / 5.1%, P(7d DD<=-10%) 11 / 21 / 25%. Only 1 of 27 OKX crashes followed pre-3d funding >20%; 24 of 27 followed <11%.
- Q4 persistence (trailing-3d -> next-7d mean APR): corr 0.69 (winsor 0.72), slope 0.61, intercept 1.9 (Binance same days 0.71 / 0.79 / 0.62 / 2.3). 6%/2% rule: P(next7 >= 2% | t3 >= 6%) = **89.3%** (1047/1173) vs Binance 94.5% on the same days; P(next7>=2% | t3<6%) 52.9% vs 66.3%; mean / median / p10 of next-7d APR given t3>=6%: 10.9 / 7.9 / 1.8 (BN 10.3 / 8.7 / 3.8); P(next7<0 | on) 5.8% vs 2.5%. Since 2024-09: 92.7% (491 obs; P<0 3.1%) vs 94.8% (BN). A fast-drop exit still adds nothing (OKX t3>=6% & fell>40%: P(next7<2%) 6.4% vs 11.1% otherwise). The 6/2 rule is a valid state variable on OKX but weaker (~5pp lower hit rate, fatter negative tail) and it is *on* less often (t3>=6% on 34% of days vs 49%).

## 4. Prices: OKX vs Binance (1,844 days)
Daily returns corr 1.0000; mean |diff| 0.016% BTC / 0.017% ETH, max 0.13% / 0.11%; OKX-swap close / Binance-swap close - 1: mean +0.02%, p1..p99 -0.07%..+0.08%, abs max 0.13%. The top/bottom 1.5% big-move days are **identical** (57/57 BTC, 57/57 ETH; thresholds BTC -6.5/+7.4 vs -6.5/+7.5); the crash list is the same (BTC: 2021-09-07 -11.0, 2022-01-21, 2022-05-09, 2022-06-13 -15.4, 2022-08-19, 2022-11-09 -14.2, 2026-02-05 -14.0 on both). Price-side conclusions transfer exactly; no visible OKX-specific basis at daily-close resolution (intraday wicks/mark price, e.g. 2025-10-10, not examined).

## 5. What the 6%/2% rule would have done (1.0x, 0.30% round trip, no borrow/slippage; independent per-year cold start; net APR % / time-in-market)
| | BTC OKX | BTC BN | ETH OKX | ETH BN |
|---|---|---|---|---|
| 2022 | -4.8 / 35% | -2.6 / 55% | -3.8 / 17% | -4.4 / 42% |
| 2023 | +0.4 / 48% | +5.2 / 78% | +0.9 / 59% | +5.6 / 80% |
| 2024 | +7.3 / 64% | +10.4 / 85% | +7.4 / 68% | +11.9 / 88% |
| 2025 | -0.3 / 55% | +1.7 / 65% | +0.9 / 43% | -0.0 / 58% |
| 2026 YTD | -1.2 / 44% | +1.6 / 46% | -0.7 / 34% | -0.4 / 36% |
| last 3 months | +0.8 / 72% (3 trips) | +4.7 / 88% (1) | +0.4 / 47% | +0.3 / 54% |
| since 2024-09 | +1.1 | +3.1 | +1.9 | +1.9 |
| all 4.7y (excl. hole) | +0.5 (66 trips) | +3.8 (37) | +1.3 (54) | +3.2 (44) |
OKX earns ~3pp less than Binance for BTC and ~2pp less for ETH over the whole sample, mostly in 2023-24; in 2025-26 the two venues are within about +/-2pp with either one ahead. OKX also trades more (2-3x the round trips, 66 vs 37 for BTC), which is where the extra cost comes from.

## Verdict
- **Carries over with confidence**: price-side facts (identical big-move days, same vol-tercile risk, same crash days); funding gets to ~0 or negative for a week after crashes (larger on OKX) and jumps ~1.5-2x after rallies; persistence of trailing-3d funding (corr ~0.7, no fast-drop exit gain); carry is a regime effect (2024 good, 2022/2025-26 near zero), and crash funding cost is small next to hedge/ADL risk.
- **Does not carry over / cannot be tested on OKX**: the 2020-03 and 2021-05 crash tails, the 2020-21 high-funding regime, the ">20% funding => 2x crash frequency" effect (no lift in the OKX sample), ADL/liquidation behaviour, intraday mark-price wicks, and anything before 2021-09 (108-day hole 2021-10..2022-01-15).
- **Expected OKX carry vs the Binance-based net APR in research section 6**: about 2-3pp lower on average (0-1pp lower in weak years such as 2025-26 where both are ~0; 3-4pp lower in a strong year like 2024; ETH -0.8 to -1.5pp mean funding). Using the trailing-3d regression: OKX t3 ~ Binance t3 - 1.2..1.8pp. With the 6/2 rule expect lower time in market (~45-55% vs 65%), 2x the trips, and slightly higher odds of a funding-negative week (5.8% vs 2.5% of entry signals). Since 2024-09 the OKX rule earned +1.1% (BTC) / +1.9% (ETH), before borrow/slippage: still consistent with the earlier "do not build now, monitor" recommendation, but slightly more negative for OKX.
- Consider raising the entry threshold (e.g. 8%) or adding a minimum hold on OKX; not tested here (would be in-sample tuning).

## Caveats
One 4.7-year regime (2021-09 -> 2026-09, no 2020-21 mania, 2022 bear + 2024 bull + long chop); 14 / 13 crashes, heavily correlated, overlapping windows (n_eff ~ n/7), no confidence intervals; 3-month API view is only 8% sign disagreement but tiny; OKX formula shift around 2025-04 (0.0001 clamp appears; cause not verified) means pre-2025 OKX history describes a different funding regime; 8h-grid rounding of timestamps (OKX settlements differ by at most seconds); OKX daily candles are the swap index/last-trade series, not mark price; no borrow, slippage, ADL, margin or basis modelling; per-year sim rows carry cold-start bias (-0.35..-0.5pp) and the 2021 row (30 days) is not shown.
