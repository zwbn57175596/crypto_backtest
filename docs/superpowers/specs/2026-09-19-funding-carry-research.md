# Funding Carry — Research Findings and Go/No-Go (Phases 1–2)

Date: 2026-09-19 · Branch: `feat/funding-carry` · Inputs: read-only OKX probe (acctLv=3 sub-account), Binance funding history 2019→2026-09, OKX funding history (3 months), `scripts/carry_viability.py` (borrow 3.5%, fee/side 0.15% taker or 0.10% maker; enter 6% / exit 2% / lev-spread 3%). Related: `2026-09-19-funding-carry-design.md`, `2026-09-19-funding-carry-funding-vs-moves.md`.

## 1. Account
- `acctLv=3` (multi-currency margin) confirmed; `posMode=net_mode`; **`autoLoan=False`** — auto-borrow must be enabled before any leveraged/borrowed leg (account-setting change; do it only once the old strategy is stopped).
- Account is flat right now: `imr=0`, `mmr=0`, `notionalUsd=0`, `mgnRatio` empty; equity ≈ 1.1k USDT. The old strategy holds no position, so stopping it costs nothing to unwind.
- Decision (user): carry runs in this same sub-account (option B) after the old strategy is stopped. Risk noted: `mgnRatio` is account-wide, so no other strategy may run alongside.
- "Unified account" wording: OKX's unified-account migration was 2021–22; the user's account is the unified architecture in multi-currency mode.

## 2. Instruments (probe)
| | ctVal | lotSz / minSz | spot minSz |
|---|---|---|---|
| BTC-USDT-SWAP | 0.01 BTC | 0.01 contracts (=0.0001 BTC) | 0.00001 BTC |
| ETH-USDT-SWAP | 0.1 ETH | 0.01 contracts (=0.001 ETH) | 0.0001 ETH |

Minimums are irrelevant at 1000 USDT. Funding interval 8h for both. Snapshot: BTC next rate 0.0001 (10.95% APR, the base floor), ETH 6.08e-5 (6.7% APR).

## 3. Fees (real tier)
Spot maker 0.08% / taker 0.10%; swap maker 0.02% / taker 0.05% → taker 0.15% per side (0.30% round trip), maker 0.10% per side. Identical to the simulator defaults.

## 4. Borrow
- Private `interestRate` = 0.000004; public loan-quota USDT `rate` = 0.000096. Ratio is exactly 24 → hourly vs daily quoting of the same rate. **USDT borrow ≈ 3.5% APR** (0.000004 × 24 × 365). Interest accrues hourly. The earlier 8% assumption was too pessimistic; 84% was a unit error.
- USDT basic quota 5,000,000 — 500 USDT of borrowing is far below any limit.
- Remaining check: compare 3.5% with the OKX web UI once (evidence is internal consistency, not the UI).

## 5. Funding history
Binance BTC 2019-09-10→, ETH 2019-11-27→ (7698 / 7464 rows, median interval 8.00h). OKX only 2026-06-17→ (281 rows) — long-run research is Binance-based. Not checked for gaps or non-8h stretches beyond the median (Binance has used 4h intervals on some pairs under extreme funding; low risk for BTC/ETH).

## 6. Viability results (Binance, taker, borrow 3.5%; per-year rows are cold-started, bias ≈ −0.35..−0.5pp)
| Net APR | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 YTD |
|---|---|---|---|---|---|---|---|
| BTC 1.0x | +15.0 | +28.4 | −3.7 | +4.9 | +10.2 | +1.4 | +1.4 |
| BTC 1.5x | +21.3 | +41.0 | −7.3 | +3.6 | +12.9 | −0.7 | +0.2 |
| ETH 1.0x | +26.1 | +34.6 | −5.1 | +5.3 | +11.8 | −0.5 | −0.7 |
| ETH 1.5x | +36.8 | +50.4 | −9.6 | +4.1 | +15.4 | −2.5 | −2.6 |

Full period (2019→2026): BTC 1.0x +8.7% (maker +9.5%), ETH 1.0x +11.1% (maker +11.8%). 1.5x adds ≈ +1.9pp (BTC) / +3.2pp (ETH) over the full period, but only through 2020–21 and 2024; it **loses to 1.0x in 2022, 2023, 2025, 2026** (30 leverage flips in 2025 alone, each charged). Last 3 months on OKX itself (real venue): BTC +0.4% APR taker (+1.6% maker), ETH +0.1% — effectively zero at 1.0x, and 1.5x is negative.
Read: carry paid in high-leverage years (2020, 2021, 2024), bled in 2022, and is near zero in the current calm regime. The funding-vs-moves study shows this is a regime effect, not funding being eaten by crashes.

## 7. Open risks / biases
- `mgnRatio` scale and threshold calibration: unobservable while flat → Phase 3 shock simulation + demo/small live.
- Basis PnL not modelled (only 0.02% slip per traded notional); likely conservative. Accrual-only drawdown (no mark-to-market).
- Borrow-rate history unavailable; 3.5% is today's rate. 1.0x rows are borrow-independent.
- Per-year cold-start bias (−0.35..−0.5pp); table headers round 3.5% to "4%".
- Hedge integrity (ADL) is the main tail risk, not funding cost (see funding-vs-moves study).

## 8. Gate evaluation (proposed gates; user may change)
- **(a) 1.0x taker net APR ≥ 4% on the last 2 full calendar years (2024+2025), BTC and ETH — BORDERLINE.** 2-year average: BTC ≈ +5.8%, ETH ≈ +5.6% (pass), but 2025 alone is +1.4% / −0.5% (fail) and 2026 YTD ≈ +1.4% / −0.7%. The average is carried by 2024; the trend since is weak.
- **(b) 1.5x beats 1.0x by ≥ 1pp on the same window — FAIL.** BTC 2024 +2.7pp, 2025 −2.1pp (avg +0.3pp); ETH +3.6pp / −2.0pp (avg +0.8pp). The current leverage rule loses in weak years. A stricter leverage trigger might fix it but would be in-sample tuning; it needs a walk-forward test before adoption.
- **(c) Max drawdown ≤ 3% of capital — NOT EVALUABLE in Phase 2** (accrual-only drawdown; defer to Phase 3 shock simulation).
- **(d) Beats the risk-free alternative — MARGINAL: clear win over the full period and in 2024, no edge in the current regime.** Alternatives (OKX Simple Earn): USDT flexible 3.10% / 90-day 4.00%; BTC flexible 0.01%; ETH flexible 0.49%. Fair accounting: capital earns the USDT rate while the strategy is flat (parked) and is committed to the hedge while in the market, so excess over always-parked = net APR − 3.10% × time-in-market (the simulator credits 0% while flat, so comparing net APR directly with 3.10% understates the carry — an earlier draft of this section did that). Excess vs flexible 3.10%, 1.0x taker: BTC 2024 +7.6pp, 2025 −0.6pp, 2026 YTD 0.0pp, full period +6.5pp; ETH 2024 +9.1pp, 2025 −2.3pp, 2026 YTD −1.8pp, full period +8.8pp; 2024+2025 average +3.5pp (BTC) / +3.4pp (ETH) ≈ 35 USDT/yr on 1000 USDT. OKX last 3 months: BTC −1.9pp taker / −0.7pp maker. Against the 90-day 4.00% (locks funds, not like-for-like) subtract a further ≈0.6–0.8pp. Not modelled and UNVERIFIED: Simple Earn on the spot leg (BTC 0.01%, ETH 0.49% → ≤0.5pp, and probably incompatible with using the coins as margin collateral). The excess must also pay for basis, ADL and operational risk.

## 9. Recommendation
Do not build Phase 3 execution now. Expected profit at 1000 USDT is ≈ 50 USDT/yr in a good year (≈ 1.4% in the latest year, ≈ 0 over the last 3 months on OKX), against real operational and ADL risk. Drop 1.5x leverage from scope (fails gate b). Keep the option open with a lightweight, no-key funding monitor (like `edge_monitor`) that alerts when trailing-3d funding stays > ~8% APR, then revisit with 1.0x only. Rationale for ~8%: the risk-free alternative is 3.1–4.0%, round-trip fees amortise to ~1% over a month-long hold, and the remainder (~3–4pp) is the minimum premium for basis, ADL and operational risk. Until then idle USDT in Simple Earn (3.10% flexible) beats the carry in the current regime.
