# src/risk_forecast.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = Path("data")
FIG_DIR = Path("reports/figures")

# ---------- utilities ----------

def load_prices(ticker: str) -> pd.Series:
    path = DATA_DIR / f"{ticker}.csv"
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date")
    s = df.set_index("Date")["Close"].astype(float)
    return s


def log_returns(prices: pd.Series) -> pd.Series:
    r = np.log(prices).diff()
    return r.dropna()


def ewma_vol(returns: np.ndarray, lam: float = 0.94) -> np.ndarray:
    """
    RiskMetrics-style EWMA volatility (proxy for GARCH in a lightweight, believable student project).
    returns: daily log returns
    """
    var = np.empty_like(returns)
    var[0] = np.var(returns[:50]) if len(returns) >= 50 else np.var(returns)
    for t in range(1, len(returns)):
        var[t] = lam * var[t-1] + (1 - lam) * returns[t-1] ** 2
    return np.sqrt(np.maximum(var, 1e-18))


def var_es_from_pnl(pnl: np.ndarray, alpha: float) -> tuple[float, float]:
    """
    pnl: distribution of 1-day P&L (positive good, negative loss)
    VaR_alpha reported as a positive number (loss threshold).
    ES_alpha reported as a positive number (expected loss beyond VaR).
    """
    # loss = -pnl
    loss = -pnl
    q = np.quantile(loss, alpha)
    tail = loss[loss >= q]
    es = float(tail.mean()) if len(tail) else float(q)
    return float(q), es


@dataclass(frozen=True)
class BacktestResult:
    dates: pd.DatetimeIndex
    realised_pnl: np.ndarray
    var: np.ndarray
    es: np.ndarray


# ---------- models ----------

def historical_simulation(pnl_hist: np.ndarray, alpha: float) -> tuple[float, float]:
    return var_es_from_pnl(pnl_hist, alpha)


def filtered_historical_simulation(returns_hist: np.ndarray, vol_hist: np.ndarray, vol_next: float, alpha: float) -> tuple[float, float]:
    """
    FHS: standardise returns -> resample shocks -> scale by next-day vol.
    """
    z = returns_hist / np.maximum(vol_hist, 1e-12)
    z = z[np.isfinite(z)]
    shocks = np.random.choice(z, size=len(z), replace=True)
    r_next = vol_next * shocks
    pnl = r_next  # assume $1 notional; scaling handled later
    return var_es_from_pnl(pnl, alpha)


# ---------- pipeline ----------

def walk_forward_backtest(
    returns: pd.Series,
    alpha: float = 0.975,
    window: int = 750,
    lam: float = 0.94,
    n_mc: int = 20000,
    seed: int = 42,
) -> tuple[BacktestResult, BacktestResult]:
    """
    Compare:
      (A) Historical Simulation (HS)
      (B) Filtered Historical Simulation (FHS) using EWMA vol
    We forecast next-day VaR/ES each day, then compare against realised P&L.
    """
    rng = np.random.default_rng(seed)
    np.random.seed(seed)  # for np.random.choice in FHS

    r = returns.values.astype(float)
    dates = returns.index

    if len(r) <= window + 5:
        raise ValueError(f"Not enough data ({len(r)}) for window={window}.")

    vol = ewma_vol(r, lam=lam)

    hs_var, hs_es = [], []
    fhs_var, fhs_es = [], []
    realised = []
    out_dates = []

    for t in range(window, len(r) - 1):
        r_hist = r[t - window:t]
        vol_hist = vol[t - window:t]
        vol_next = vol[t]  # using vol at time t to forecast t+1

        # realised next-day P&L in return units
        pnl_realised = r[t + 1]

        # HS (on P&L)
        var_hs, es_hs = historical_simulation(r_hist, alpha)

        # FHS (Monte Carlo via resampling standardized shocks)
        # Use rng.choice to avoid global state; implement directly here for speed
        z = r_hist / np.maximum(vol_hist, 1e-12)
        z = z[np.isfinite(z)]
        shocks = rng.choice(z, size=n_mc, replace=True)
        pnl_mc = vol_next * shocks
        var_fhs, es_fhs = var_es_from_pnl(pnl_mc, alpha)

        hs_var.append(var_hs)
        hs_es.append(es_hs)
        fhs_var.append(var_fhs)
        fhs_es.append(es_fhs)
        realised.append(pnl_realised)
        out_dates.append(dates[t + 1])

    idx = pd.DatetimeIndex(out_dates)
    hs = BacktestResult(idx, np.array(realised), np.array(hs_var), np.array(hs_es))
    fhs = BacktestResult(idx, np.array(realised), np.array(fhs_var), np.array(fhs_es))
    return hs, fhs


def breach_rate(pnl: np.ndarray, var: np.ndarray) -> float:
    # breach if loss > VaR  <=> -pnl > var  <=> pnl < -var
    return float(np.mean(pnl < -var))


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Simple portfolio: 60% SPY, 30% TLT, 10% GLD
    weights = {"SPY": 0.60, "TLT": 0.30, "GLD": 0.10}

    rets = []
    for tkr, w in weights.items():
        p = load_prices(tkr)
        r = log_returns(p).rename(tkr)
        rets.append(r)

    df = pd.concat(rets, axis=1, join="inner").dropna()
    port_r = (df * pd.Series(weights)).sum(axis=1)
    port_r.name = "Portfolio"

    alpha = 0.975
    hs, fhs = walk_forward_backtest(port_r, alpha=alpha, window=750, lam=0.94, n_mc=20000, seed=42)

    hs_br = breach_rate(hs.realised_pnl, hs.var)
    fhs_br = breach_rate(fhs.realised_pnl, fhs.var)

    print("Backtest summary (1-day, return units, $1 notional):")
    print(f"alpha={alpha:.3f}  window=750  n_obs={len(hs.dates)}")
    print(f"HS  breach_rate = {hs_br:.4f} (expected ~ {1-alpha:.4f})")
    print(f"FHS breach_rate = {fhs_br:.4f} (expected ~ {1-alpha:.4f})")

    # ---------- Figure 1: time series of realised P&L vs -VaR ----------
    plt.figure()
    plt.plot(hs.dates, hs.realised_pnl, label="Realised P&L (return)")
    plt.plot(hs.dates, -hs.var, label="HS VaR threshold (-VaR)")
    plt.plot(fhs.dates, -fhs.var, label="FHS VaR threshold (-VaR)")
    plt.title("1-day Portfolio P&L vs VaR Threshold (HS vs FHS)")
    plt.xlabel("Date")
    plt.ylabel("Return")
    plt.legend()
    plt.tight_layout()
    out1 = FIG_DIR / "pnl_vs_var.png"
    plt.savefig(out1, dpi=160)
    plt.close()
    print(f"Saved {out1}")

    # ---------- Figure 2: rolling breach rate ----------
    roll = 250
    hs_b = (pd.Series(hs.realised_pnl, index=hs.dates) < -pd.Series(hs.var, index=hs.dates)).astype(int)
    fhs_b = (pd.Series(fhs.realised_pnl, index=fhs.dates) < -pd.Series(fhs.var, index=fhs.dates)).astype(int)
    hs_roll = hs_b.rolling(roll).mean()
    fhs_roll = fhs_b.rolling(roll).mean()

    plt.figure()
    plt.plot(hs_roll.index, hs_roll.values, label=f"HS rolling breach rate ({roll}d)")
    plt.plot(fhs_roll.index, fhs_roll.values, label=f"FHS rolling breach rate ({roll}d)")
    plt.axhline(1 - alpha, linestyle="--", label="Expected")
    plt.title("Rolling VaR Breach Rate")
    plt.xlabel("Date")
    plt.ylabel("Breach rate")
    plt.legend()
    plt.tight_layout()
    out2 = FIG_DIR / "rolling_breach_rate.png"
    plt.savefig(out2, dpi=160)
    plt.close()
    print(f"Saved {out2}")


if __name__ == "__main__":
    main()