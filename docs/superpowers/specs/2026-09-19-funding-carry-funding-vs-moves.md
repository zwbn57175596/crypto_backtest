# Funding rates vs big market moves (BTC/ETH perp, Binance) and implications for funding carry

**Method.** Funding: `data/carry.db` binance BTCUSDT (2019-09-10 -> 2026-09-18, 2566 days) and ETHUSDT (2019-11-28 -> 2026-09-18, 2487 days), 3 settlements/day; the 00:00 UTC settlement is assigned to the day just ended. APR = per-8h rate x 1095 x 100. Prices: Binance USDT-perp daily klines (downloaded, UTC days; partial current day dropped). "Short-perp earned" = sum of funding as % of notional (positive = short receives).
Big move = daily close-to-close return in the bottom/top 1.5% of days per symbol (BTC <= -7.2% / >= +8.5%; ETH <= -9.4% / >= +10.8%), declustered to one event per 7 days: BTC 22 crashes / 31 rallies, ETH 21 / 27. Also rolling 3-day moves beyond +-10% (BTC 79 episodes: 28 down/51 up; ETH 122: 49/73). Q3/Q4 pool BTC+ETH (5035 overlapping day-samples, ~719 non-overlapping 7-day blocks; the two are highly correlated so true n is nearer the BTC-only 365 blocks). Scripts: `.superpowers/analysis/an.py`, `ep.py`.
Unconditional baseline funding APR: BTC mean 11.6 / median 8.2; ETH 13.8 / 8.6; ~11-12% of days have negative funding.

## Q2 Event study (funding APR %, mean / median across events)
| Window | BTC crash (n=22) | ETH crash (21) | BTC rally (31) | ETH rally (27) |
|---|---|---|---|---|
| trailing 3d before | 16.3 / 10.9 | 22.8 / 7.1 | 17.4 / 10.6 | 16.5 / 7.4 |
| event day | 5.7 / 9.0 | 0.4 / 3.5 | 25.4 / 10.9 | 26.0 / 8.3 |
| next 1d | -4.2 / 10.3 | 2.4 / 6.3 | 38.4 / 18.2 | 26.9 / 10.9 |
| next 1-3d | 0.2 / 4.4 | 5.9 / 2.5 | 32.6 / 15.7 | 24.4 / 10.8 |
| next 1-7d | 2.5 / 3.4 | 7.7 / 0.8 | 29.5 / 17.2 | 21.9 / 8.9 |
| % events with negative mean, day / +1-7d | 27 / 36 | 38 / 48 | 19 / 6 | 15 / 7 |
| short-perp earned over [-3d,+7d], % notional mean / median | 0.20 / 0.19 | 0.34 / 0.06 | 0.78 / 0.48 | 0.63 / 0.25 |
| same for a normal 11-day window | 0.35 / 0.20 | 0.42 / 0.22 | | |

Crashes: funding collapses from a ~2x-baseline mean toward ~0 within a day and stays depressed for about a week (median ~1-4% APR vs 8% baseline); it goes negative in roughly 1/3-1/2 of crashes (vs ~12% of days normally), but usually only mildly. Negative funding is deep only in a few tail events (see below). Total cost of a crash window is small relative to the spot/perp leg risk: the worst 11-day window was -0.92% (BTC, 2020-03) and -1.26% (ETH) of notional; only 4/22 (BTC) and 6/21 (ETH) crash windows had negative cumulative funding. Rallies: funding roughly doubles-triples (median +60-100% vs. baseline; mean 25-38% APR) for a week, so short-perp earns ~2x normal.

**Episodes (pre-3d -> event day -> next 1-7d APR %):** 2020-03-12 BTC -40% (13 -> -8 -> -52; -161% APR trough, 8 negative days, cum -0.9%); ETH -45% (23 -> 27 -> -8). 2021-01-21 BTC -13% (83 -> 31 -> 18); ETH -19% (118 -> 45 -> 48: funding stayed high). 2021-05-19 BTC -14% (35 -> -16 -> 4); ETH -28% (31 -> -106 -> 11). 2022-05-09 BTC -12% (3 -> 11 -> 2); 2022-06-13 BTC -15% (5 -> 0 -> 0); 2022-11-09 BTC -14% (6 -> -53 -> -12; cum -0.3%, 7 negative days); ETH -17% (same, cum -0.25%). 2024-08-05 BTC -7% / ETH -10% (funding ~0-2 throughout; the 08-08 rebound did not lift it). 2025-10-10 BTC -7% / ETH -12.5% on a daily close basis (5 -> 5 -> -1; funding barely moved, intraday wick far larger than the daily close suggests). 2026-02-05 BTC -14% / ETH -15% (1 -> -2 -> -5; negative for ~a week). Rallies: 2021-01-03 ETH +26% (58 -> 198 -> 175); 2020-07 / 2021-02 BTC: funding 25-100%+ APR.

## Q3 Does trailing-3d funding predict big moves? (pooled BTC+ETH, forward from close of day t, %)
| Trailing 3d funding APR | n days (~blocks) | P(3d<=-10%) | P(3d>=+10%) | P(7d<=-15%) | P(7d>=+15%) | mean abs 7d ret | mean fwd 7d max DD | mean fwd 7d max run-up | P(7d DD<=-10%) |
|---|---|---|---|---|---|---|---|---|---|
| <0 | 550 (79) | 3.1 | 6.4 | 2.2 | 6.5 | 6.3 | -6.2 | 8.6 | 15.6 |
| 0-5 | 1168 (167) | 2.8 | 4.3 | 2.9 | 4.7 | 5.9 | -6.0 | 6.6 | 17.3 |
| 5-11 | 2191 (313) | 4.2 | 5.0 | 3.7 | 7.0 | 6.7 | -6.7 | 6.8 | 22.0 |
| 11-20 | 393 (56) | 3.1 | 8.1 | 2.3 | 10.2 | 7.2 | -6.7 | 8.6 | 18.8 |
| >20 | 733 (105) | 8.2 | 10.1 | 6.7 | 14.5 | 9.9 | -9.9 | 11.1 | 33.4 |
| all | 5035 (719) | 4.3 | 6.0 | 3.7 | 7.7 | 7.0 | -7.0 | 7.7 | 21.6 |
Very high funding (>20% APR) roughly doubles the frequency of a -10% 3-day crash (8.2% vs 4.3%) AND of a +10% rally, and raises the mean 7d drawdown from -7% to -10%: it is a volatility/leverage-regime marker more than a directional crash signal (the rally probability rises as much). The effect holds for both symbols (BTC 7.3% vs 2.9%; ETH 9.0% vs 5.6% crash) but is concentrated pre-2024-09 (>20% band: 699 of the 733 days). Since 2024-09 there are only 34 days >20% and 65 days >11%, so no evidence exists for the recent regime. Trailing 7d realised-vol terciles show the same effect more cleanly: P(3d<=-10%) 2.2 / 3.7 / 6.8% and P(7d DD<=-10%) 14 / 21 / 30% (low/mid/high vol; the high-vol tercile also has mean funding 17.8% vs 7.7%). Only 12 of 43 declustered crashes (4/22 BTC, 8/21 ETH) followed a >20% funding regime; 26 of 43 followed funding below 11%, so high funding is neither necessary nor a reliable warning.

## Q4 Persistence: trailing-3d mean funding -> next-7d mean funding (APR %)
| Period | n | corr (winsorised) | slope (intercept) | P(next7 >=2% \| t3>=6%) | P(next7 >=2% \| t3<6%) | mean next7 \| t3>=6% (median, p10) | P(next7<0 \| t3>=6%) |
|---|---|---|---|---|---|---|---|
| all | 5035 | 0.77 (0.80) | 0.72 (3.5) | 94.8% (2844/3001) | 66.1% | 19.4 (10.9, 4.2) | 2.9% |
| pre 2024-09 | 3553 | 0.75 (0.78) | 0.71 (4.7) | 94.8% | 66.2% | 21.8 (10.9, 4.4) | 3.1% |
| since 2024-09 | 1482 | 0.79 (0.80) | 0.74 (1.3) | 94.8% | 66.0% | 8.6 (7.7, 3.6) | 2.0% |
Persistence is strong and stable across regimes (slope ~0.72, corr ~0.77), so a trailing-3d threshold (6% enter / 2% exit) is a sound state variable; only the level shifted (regression intercept 4.7 -> 1.3). Within t3>=6%, funding that already fell >40% vs the prior 3 days (n=208) did NOT predict worse next-7d funding (P(<2%) 1.9% vs 5.5%), so a "fast drop" exit rule adds nothing. Regime by year (BTC / ETH mean APR; % days negative): 2021 31 / 38; 2022 4 / 1 (18% / 34% neg days); 2024 12 / 13; 2025 5 / 5; 2026 YTD 2.9 / 1.7 (26% / 30% neg days; only 30% / 19% of days above 6% APR).

## Implications
- (a) Entry/exit [well supported]: persistence is high and unchanged post-2024, so keep the 6/2 threshold on trailing-3d funding. Adding a fast-drop exit gains nothing (above). A calm-volatility filter is supported only as a size/leverage input, not as an entry gate: trailing-vol terciles separate forward-crash risk (2.2% vs 6.8%) well, but that does not change expected funding, and a gate on it would cost carry in the highest-funding weeks (which are the high-vol ones).
- (b) ADL/risk tiers [moderately supported]: crash risk (both symbols) is ~2x baseline when 3d funding >20% (and in the high-vol tercile), so the watchdog and tighter margin buffers matter most in those regimes. But 26 of 43 crashes came from low-funding regimes (<11%), so the watchdog cannot be relaxed at low funding either; treat funding regime as a modifier, not a switch. Post-2024 high-funding regimes have been too rare to test.
- (c) 1.5x leverage [supported for rallies, speculative for the rest]: high funding co-occurs with larger up-moves (P(7d>=+15%) 14.5% vs 7.7%; mean 7d max run-up +11.1% vs +7.7%) which is what stresses a levered short-perp margin. Recommend cutting or capping leverage when trailing-3d funding >20% (or realised vol in the top tercile). The funding P&L itself never justifies the extra risk: when funding is highest (2021, >100% APR) it is also when both tails fatten. (Rally margin impact was not modelled.)
- (d) Regime: carry has paid systematically in bull/high-leverage regimes (2020-21: ~17-38% APR, <=7% negative days) and bled/flattened in bear chop (2022: ~1-4% APR, 18-34% negative days; 2026 YTD ~2-3% APR, ~28% negative days). The bleed occurs in slow bear/quiet regimes rather than crash spikes; a sub-2% exit rule handles it. Crashes themselves cost little in funding (worst -0.9% to -1.3% of notional over 11 days; median crash window still earns ~0.2%), so the dominant crash risk is hedge integrity (ADL on the winning short, leg-execution), not funding cost [funding part well supported; ADL part is not measurable here].
- Also (well supported): after rallies the short leg earns ~2x baseline for a week, so an exit purely triggered by a rally forgoes the best funding weeks; the trade-off is margin stress, not funding.

## Caveats
Few crash events (22 + 21, heavily correlated, dominated by 2020-03, 2021-05, 2022-11); overlapping windows inflate n (use n/7 for effective sample size; no formal confidence intervals computed, so differences of a couple of points between adjacent bands are noise); post-2024-09 regime has almost no high-funding observations; funding is Binance-only, at 8h settlement resolution, daily close-to-close returns understate intraday wicks (e.g. 2025-10-10); no basis, spot-side, ADL, liquidation, borrow-cost or margin modelling; "big move" thresholds are sample percentiles chosen ad hoc; the 2020-03 tail drives most of the negative-funding cost numbers.
