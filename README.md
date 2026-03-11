# GARCH Risk Forecasting (VaR/ES) — HS vs FHS vs MC-GARCH-t

A reproducible **portfolio risk forecasting** project that compares:

- **HS (Historical Simulation)** VaR/ES  
- **FHS (Filtered Historical Simulation)** VaR/ES using **time-varying volatility** (EWMA / RiskMetrics-style “GARCH-like” volatility)  
- **MC-GARCH(1,1)-t** VaR/ES using a **Student-t** GARCH model and Monte Carlo simulation

Goal: a realistic quant workflow: **data → returns → volatility model → Monte Carlo / resampling → VaR/ES → backtesting → plots + tables**.

---

## Project structure (high level)

- `src/` — reproducible scripts you run from terminal (`python src/...`)
- `data/` — downloaded datasets (kept out of git except a `.gitkeep`)
- `notebooks/` — optional exploration (not required to reproduce results)
- `reports/` — generated outputs (ignored by git)
- `reports/figures/` — **tracked** key plots embedded in this README
- `reports/tables/` — **tracked** key result tables (CSV)

---

## What this project does

- Downloads daily price data (**SPY, TLT, GLD**) from **Stooq** (no API key)
- Builds a simple portfolio (**60% SPY, 30% TLT, 10% GLD**)
- Forecasts **1-day VaR/ES at 97.5%** using:
  - **HS:** quantile/tail mean of historical returns
  - **FHS:** standardise returns by volatility, resample shocks, scale by next-day volatility
  - **MC-GARCH-t:** fit rolling **GARCH(1,1)** with **Student-t innovations**, simulate next-day distribution
- Walk-forward backtest and produces:
  - Realised P&L vs VaR/ES thresholds
  - Rolling breach rate vs expected breach rate
  - Summary tables + formal VaR backtests

---

## Results (latest run)

Backtest setup:
- Portfolio: 60% SPY / 30% TLT / 10% GLD
- 1-day risk at **97.5%** confidence (α = 0.975)
- Walk-forward backtest sample size: **n = 4,537** days
- Expected breach rate: **1 − α = 2.50%**

Empirical calibration (VaR exceptions):
- **HS breach rate:** **2.73%** (expected 2.50%)
- **FHS (EWMA) breach rate:** **2.49%** (closest to expected)
- **MC-GARCH(1,1)-t breach rate:** **2.78%**

Full numeric summaries:
- `reports/tables/backtest_summary.csv`
- `reports/tables/var_backtests.csv`

---

## VaR backtesting (Kupiec & Christoffersen)

Formal likelihood-ratio tests (p-values in `reports/tables/var_backtests.csv`):

- **Kupiec LR_uc (unconditional coverage):** tests whether the exception rate matches the target \(1-\alpha\).
- **Christoffersen LR_ind (independence):** tests whether exceptions are independent over time (no clustering).
- **Conditional coverage LR_cc:** joint test of coverage + independence (LR_cc = LR_uc + LR_ind).

At the 5% level (**p < 0.05 ⇒ reject**):
- All models pass **unconditional coverage** (FHS is strongest).
- All models reject **independence**, indicating exception clustering.
- HS fails **conditional coverage** most strongly; FHS and MC-GARCH-t improve but still reject conditional coverage.

---

## Key figures

**Realised P&L vs VaR thresholds**
![P&L vs VaR](reports/figures/pnl_vs_var.png)

**Realised P&L vs VaR and ES thresholds**
![P&L vs VaR and ES](reports/figures/pnl_vs_var_es.png)

**Rolling VaR breach rate**
![Rolling breach](reports/figures/rolling_breach_rate.png)

---

## Reproduce locally

### 1) Create venv + install deps

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt