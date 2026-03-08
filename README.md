# GARCH Risk Forecasting (VaR/ES) — Filtered Historical Simulation

A reproducible **portfolio risk forecasting** project that compares:

- **HS (Historical Simulation)** VaR/ES  
- **FHS (Filtered Historical Simulation)** VaR/ES using **time-varying volatility** (EWMA / RiskMetrics-style “GARCH-like” volatility)

The goal is a realistic quant workflow: **data → returns → volatility model → Monte Carlo / resampling → VaR backtesting → plots**.

---

## Project structure (high level)

- `src/` — reproducible scripts you run from terminal (`python src/...`)
- `data/` — downloaded datasets (kept out of git except a `.gitkeep`)
- `notebooks/` — optional exploration (not required to reproduce results)
- `reports/` — generated outputs (ignored by git)
- `reports/figures/` — **tracked** key plots embedded in this README

---

## What this project does

- Downloads daily price data (SPY, TLT, GLD) from Stooq (no API key)
- Builds a simple portfolio (60% SPY, 30% TLT, 10% GLD)
- Forecasts **1-day VaR at 97.5%** using:
  - **HS:** quantile of historical returns
  - **FHS:** standardise returns by volatility, resample shocks, scale by next-day volatility
- Walk-forward backtest and produces:
  - Realised P&L vs VaR thresholds
  - Rolling breach rate vs expected breach rate

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

In this dataset, **FHS (EWMA)** is best calibrated to the target exception rate, while **MC-GARCH-t** produces a slightly higher exception rate.

Full numeric summary:
- `reports/tables/backtest_summary.csv`

### Key figures

![P&L vs VaR](reports/figures/pnl_vs_var.png)

![P&L vs VaR and ES](reports/figures/pnl_vs_var_es.png)

![Rolling breach](reports/figures/rolling_breach_rate.png)

### Key figures

**Realised P&L vs VaR thresholds**
![P&L vs VaR](reports/figures/pnl_vs_var.png)

**Realised P&L vs VaR and ES thresholds**
![P&L vs VaR and ES](reports/figures/pnl_vs_var_es.png)

**Rolling VaR breach rate**
![Rolling breach](reports/figures/rolling_breach_rate.png)



### Key figures

**Realised P&L vs VaR thresholds**  
![P&L vs VaR](reports/figures/pnl_vs_var.png)

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