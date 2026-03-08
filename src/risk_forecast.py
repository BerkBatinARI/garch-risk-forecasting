# src/risk_forecast.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from arch import arch_model


DATA_DIR = Path("data")
FIG_DIR = Path("reports/figures")
TABLE_DIR = Path("reports/tables")


# ---------- utilities ----------
def load_prices(ticker: str) -> pd.Series:
    path = DATA_DIR / f"{ticker}.csv"
    df = pd.read_csv(path, parse_dates=["Date"]).sort_values("Date")
    return df.set_index("Date")["Close"].astype(float)


def log_returns(prices: pd.Series) -> pd.Series:
    return np.log(prices).diff().dropna()


def ewma_vol(returns: np.ndarray, lam: float = 0.94) -> np.ndarray:
    """RiskMetrics-style EWMA volatility."""
    var = np.empty_like(returns)
    if len(returns) >= 50:
        var[0] = np.var(returns[:50])
    else:
        var[0] = np.var(returns)

    for t in range(1, len(returns)):
        var[t] = lam * var[t - 1] + (1 - lam) * returns[t - 1] ** 2
    return np.sqrt(np.maximum(var, 1e-18))


def var_es_from_pnl(pnl: np.ndarray, alpha: float) -> tuple[float, float]:
    """
    pnl: distribution of 1-day P&L (positive good, negative loss)
    VaR_alpha returned as a positive number (loss threshold).
    ES_alpha returned as a positive number (expected loss beyond VaR).
    """
    loss = -pnl
    q = np.quantile(loss, alpha)
    tail = loss[loss >= q]
    es = float(tail.mean()) if len(tail) else float(q)
    return float(q), es


def breach_rate(pnl: np.ndarray, var: np.ndarray) -> float:
    # breach if loss > VaR  <=>  -pnl > var  <=>  pnl < -var
    return float(np.mean(pnl < -var))


def draw_std_student_t(rng: np.random.Generator, nu: float, size: int) -> np.ndarray:
    """
    Draw standardized Student-t shocks with E[z]=0, Var[z]=1.
    For t ~ t_nu, Var[t] = nu/(nu-2) for nu>2.
    """
    nu = float(nu)
    if not np.isfinite(nu) or nu <= 2.05:
        # Safety fallback to avoid infinite/huge variance; use a conservative df.
        nu = 8.0
    t = rng.standard_t(df=nu, size=size)
    return t / np.sqrt(nu / (nu - 2.0))


# ---------- results container ----------
@dataclass(frozen=True)
class BacktestResult:
    dates: pd.DatetimeIndex
    realised_pnl: np.ndarray
    var: np.ndarray
    es: np.ndarray


# ---------- models ----------
def historical_simulation(pnl_hist: np.ndarray, alpha: float) -> tuple[float, float]:
    return var_es_from_pnl(pnl_hist, alpha)


def filtered_historical_simulation_mc(
    rng: np.random.Generator,
    returns_hist: np.ndarray,
    vol_hist: np.ndarray,
    vol_next: float,
    alpha: float,
    n_mc: int,
) -> tuple[float, float]:
    """
    FHS: standardise returns -> resample shocks -> scale by next-day vol.
    """
    z = returns_hist / np.maximum(vol_hist, 1e-12)
    z = z[np.isfinite(z)]
    shocks = rng.choice(z, size=n_mc, replace=True)
    pnl_mc = vol_next * shocks
    return var_es_from_pnl(pnl_mc, alpha)


def garch_one_step_forecast_sigma_and_nu(
    returns_hist: np.ndarray,
) -> tuple[float, float]:
    """
    Fit GARCH(1,1) with Student-t innovations on returns_hist (mean=0),
    return (sigma_next, nu).
    """
    # Mean=Zero keeps it simple and stable for daily returns.
    am = arch_model(
        returns_hist,
        mean="Zero",
        vol="GARCH",
        p=1,
        q=1,
        dist="StudentsT",
        rescale=False,
    )
    res = am.fit(disp="off", options={"maxiter": 2000})
    # 1-step ahead forecast variance
    f = res.forecast(horizon=1, reindex=False)
    var_next = float(f.variance.values[-1, 0])
    sigma_next = float(np.sqrt(max(var_next, 1e-18)))

    nu = float(res.params.get("nu", np.nan))
    return sigma_next, nu


def monte_carlo_garch_t(
    rng: np.random.Generator,
    sigma_next: float,
    nu: float,
    alpha: float,
    n_mc: int,
) -> tuple[float, float]:
    """
    MC VaR/ES for next-day returns under Student-t innovations:
      r_{t+1} = sigma_{t+1} * z,   z ~ standardized t_nu
    """
    z = draw_std_student_t(rng, nu=nu, size=n_mc)
    pnl_mc = sigma_next * z
    return var_es_from_pnl(pnl_mc, alpha)


# ---------- pipeline ----------
def walk_forward_backtest(
    returns: pd.Series,
    alpha: float = 0.975,
    window: int = 750,
    lam: float = 0.94,
    n_mc_fhs: int = 20000,
    n_mc_garch: int = 20000,
    seed: int = 42,
    garch_refit_every: int = 5,
) -> tuple[BacktestResult, BacktestResult, BacktestResult]:
    """
    Compare:
      (A) HS
      (B) FHS using EWMA vol (resampling standardized shocks)
      (C) MC-GARCH(1,1) with Student-t innovations

    garch_refit_every:
      Fit GARCH every k steps to avoid extremely slow runtimes.
      Between refits, reuse last (sigma_next, nu). (Good practical compromise.)
    """
    rng = np.random.default_rng(seed)

    r = returns.values.astype(float)
    dates = returns.index

    if len(r) <= window + 5:
        raise ValueError(f"Not enough data ({len(r)}) for window={window}.")

    # Precompute EWMA vol for FHS
    vol = ewma_vol(r, lam=lam)

    hs_var, hs_es = [], []
    fhs_var, fhs_es = [], []
    g_var, g_es = [], []
    realised = []
    out_dates = []

    last_sigma_next = None
    last_nu = None

    for step, t in enumerate(range(window, len(r) - 1)):
        r_hist = r[t - window : t]
        vol_hist = vol[t - window : t]
        vol_next = vol[t]  # EWMA sigma at time t for forecasting t+1

        pnl_realised = r[t + 1]  # return units

        # HS
        var_hs, es_hs = historical_simulation(r_hist, alpha)

        # FHS (EWMA-filtered, MC via resampling)
        var_fhs, es_fhs = filtered_historical_simulation_mc(
            rng=rng,
            returns_hist=r_hist,
            vol_hist=vol_hist,
            vol_next=vol_next,
            alpha=alpha,
            n_mc=n_mc_fhs,
        )

        # MC-GARCH-t
        do_refit = (step % max(int(garch_refit_every), 1) == 0) or (last_sigma_next is None)
        if do_refit:
            sigma_next, nu = garch_one_step_forecast_sigma_and_nu(r_hist)
            last_sigma_next, last_nu = sigma_next, nu

        var_g, es_g = monte_carlo_garch_t(
            rng=rng,
            sigma_next=float(last_sigma_next),
            nu=float(last_nu),
            alpha=alpha,
            n_mc=n_mc_garch,
        )

        hs_var.append(var_hs)
        hs_es.append(es_hs)
        fhs_var.append(var_fhs)
        fhs_es.append(es_fhs)
        g_var.append(var_g)
        g_es.append(es_g)

        realised.append(pnl_realised)
        out_dates.append(dates[t + 1])

    idx = pd.DatetimeIndex(out_dates)
    realised_arr = np.array(realised, dtype=float)

    hs = BacktestResult(idx, realised_arr, np.array(hs_var), np.array(hs_es))
    fhs = BacktestResult(idx, realised_arr, np.array(fhs_var), np.array(fhs_es))
    garch_t = BacktestResult(idx, realised_arr, np.array(g_var), np.array(g_es))
    return hs, fhs, garch_t


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    # Portfolio: 60% SPY, 30% TLT, 10% GLD
    weights = {"SPY": 0.60, "TLT": 0.30, "GLD": 0.10}

    rets = []
    for tkr in weights:
        p = load_prices(tkr)
        rets.append(log_returns(p).rename(tkr))

    df = pd.concat(rets, axis=1, join="inner").dropna()
    port_r = (df * pd.Series(weights)).sum(axis=1)
    port_r.name = "Portfolio"

    alpha = 0.975
    window = 750

    hs, fhs, garch_t = walk_forward_backtest(
        port_r,
        alpha=alpha,
        window=window,
        lam=0.94,
        n_mc_fhs=20000,
        n_mc_garch=20000,
        seed=42,
        garch_refit_every=5,  # speed-friendly; set to 1 for strict refit-daily
    )

    hs_br = breach_rate(hs.realised_pnl, hs.var)
    fhs_br = breach_rate(fhs.realised_pnl, fhs.var)
    g_br = breach_rate(garch_t.realised_pnl, garch_t.var)

    n_obs = len(hs.dates)
    expected = (1 - alpha) * n_obs

    print("Backtest summary (1-day, return units, $1 notional):")
    print(f"alpha={alpha:.3f} window={window} n_obs={n_obs}")
    print(f"Expected breach rate ~ {1 - alpha:.4f}")
    print(f"HS         breach_rate = {hs_br:.4f}")
    print(f"FHS (EWMA)  breach_rate = {fhs_br:.4f}")
    print(f"MC_GARCH_T  breach_rate = {g_br:.4f}")

    # ---------- Table: summary CSV ----------
    summary = pd.DataFrame(
        [
            {
                "model": "HS",
                "alpha": alpha,
                "n_obs": n_obs,
                "breaches": int((hs.realised_pnl < -hs.var).sum()),
                "breach_rate": float(hs_br),
                "expected_breaches": float(expected),
                "avg_var": float(hs.var.mean()),
                "avg_es": float(hs.es.mean()),
            },
            {
                "model": "FHS",
                "alpha": alpha,
                "n_obs": n_obs,
                "breaches": int((fhs.realised_pnl < -fhs.var).sum()),
                "breach_rate": float(fhs_br),
                "expected_breaches": float(expected),
                "avg_var": float(fhs.var.mean()),
                "avg_es": float(fhs.es.mean()),
            },
            {
                "model": "MC_GARCH_T",
                "alpha": alpha,
                "n_obs": n_obs,
                "breaches": int((garch_t.realised_pnl < -garch_t.var).sum()),
                "breach_rate": float(g_br),
                "expected_breaches": float(expected),
                "avg_var": float(garch_t.var.mean()),
                "avg_es": float(garch_t.es.mean()),
            },
        ]
    )

    out_csv = TABLE_DIR / "backtest_summary.csv"
    summary.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")

    # ---------- Figure 1: realised P&L vs -VaR ----------
    plt.figure()
    plt.plot(hs.dates, hs.realised_pnl, label="Realised P&L (return)")
    plt.plot(hs.dates, -hs.var, label="HS (-VaR)")
    plt.plot(fhs.dates, -fhs.var, label="FHS (-VaR)")
    plt.plot(garch_t.dates, -garch_t.var, label="MC_GARCH_T (-VaR)")
    plt.title("1-day Portfolio P&L vs VaR Threshold (HS vs FHS vs MC-GARCH-t)")
    plt.xlabel("Date")
    plt.ylabel("Return")
    plt.legend()
    plt.tight_layout()
    out1 = FIG_DIR / "pnl_vs_var.png"
    plt.savefig(out1, dpi=160)
    plt.close()
    print(f"Saved {out1}")

    # ---------- Figure 1b: realised P&L vs -VaR and -ES ----------
    plt.figure()
    plt.plot(hs.dates, hs.realised_pnl, label="Realised P&L (return)")
    plt.plot(hs.dates, -hs.var, label="HS (-VaR)")
    plt.plot(hs.dates, -hs.es, label="HS (-ES)")
    plt.plot(fhs.dates, -fhs.var, label="FHS (-VaR)")
    plt.plot(fhs.dates, -fhs.es, label="FHS (-ES)")
    plt.plot(garch_t.dates, -garch_t.var, label="MC_GARCH_T (-VaR)")
    plt.plot(garch_t.dates, -garch_t.es, label="MC_GARCH_T (-ES)")
    plt.title("1-day Portfolio P&L vs VaR/ES Thresholds")
    plt.xlabel("Date")
    plt.ylabel("Return")
    plt.legend()
    plt.tight_layout()
    out_es = FIG_DIR / "pnl_vs_var_es.png"
    plt.savefig(out_es, dpi=160)
    plt.close()
    print(f"Saved {out_es}")

    # ---------- Figure 2: rolling breach rate ----------
    roll = 250

    hs_b = (
        (pd.Series(hs.realised_pnl, index=hs.dates) < -pd.Series(hs.var, index=hs.dates))
        .astype(int)
        .rename("HS_breach")
    )
    fhs_b = (
        (
            pd.Series(fhs.realised_pnl, index=fhs.dates)
            < -pd.Series(fhs.var, index=fhs.dates)
        )
        .astype(int)
        .rename("FHS_breach")
    )
    g_b = (
        (
            pd.Series(garch_t.realised_pnl, index=garch_t.dates)
            < -pd.Series(garch_t.var, index=garch_t.dates)
        )
        .astype(int)
        .rename("MC_GARCH_T_breach")
    )

    hs_roll = hs_b.rolling(roll).mean()
    fhs_roll = fhs_b.rolling(roll).mean()
    g_roll = g_b.rolling(roll).mean()

    plt.figure()
    plt.plot(hs_roll.index, hs_roll.values, label=f"HS rolling breach rate ({roll}d)")
    plt.plot(fhs_roll.index, fhs_roll.values, label=f"FHS rolling breach rate ({roll}d)")
    plt.plot(g_roll.index, g_roll.values, label=f"MC_GARCH_T rolling breach rate ({roll}d)")
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