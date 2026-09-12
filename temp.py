"""
Zhang_and_Xiang.py

Implements the quadratic-in-moneyness implied volatility (IV) curve
of Zhang and Xiang (2008). For each (date, expiry) pair the model fits:

    IV(k) = γ₀ + γ₁·k + γ₂·k²

where k = log(K/F) is log-moneyness (≈0 at-the-money, negative for
OTM puts, positive for OTM calls).

The three fitted factors carry the following economic content
(Zhang & Xiang, 2008, Proposition 2):

    γ₀  Level     ≈ ATM implied volatility (risk-neutral std deviation)
    γ₁  Slope     ≈ (1/6)  × risk-neutral skewness  (risk-reversal)
    γ₂  Curvature ≈ (1/24) × risk-neutral excess kurtosis  (butterfly)

Pipeline:
    options (OTM+ATM filtered) + forward_df
        → implied_volatility_curve()      fit γ₀/γ₁/γ₂ per (date, expiry)
        → interpolate_constant_maturity() align to 30/60/90/120/180/360-day grid
        → compute_term_spreads()          long-minus-short factor differences
        → compute_hump_factor()           medium-term excess over linear TS

Usage:
    from forward_price import estimate_forward_curve
    from Zhang_and_Xiang import (implied_volatility_curve,
                                  interpolate_constant_maturity,
                                  compute_term_spreads,
                                  compute_hump_factor)

    fwd      = estimate_forward_curve(cleaned_options)
    iv_panel = implied_volatility_curve(filtered_options, fwd)
    cm       = interpolate_constant_maturity(iv_panel)
    spreads  = compute_term_spreads(cm)
    hump     = compute_hump_factor(cm)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats.mstats import winsorize

from constant_maturity import bracket_expirations

# ── Configuration constants ───────────────────────────────────────────────────

# Must match the maturity grid in variance_risk_premium_data.py so that
# factor maturities and return maturities are on the same constant-maturity grid
TARGET_MATURITIES: dict[str, int] = {
    "1m":  30,
    "2m":  60,
    "3m":  90,
    "4m":  120,
    "6m":  180,
    "12m": 360,
}

# Winsorise term spread and hump factors at 0.5% of the whole sample,
# in line with Guo et al. (2023) and Cao and Han (2013).
_WINSORISE_LIMITS: tuple[float, float] = (0.005, 0.005)

# Minimum number of OTM/ATM options needed for a reliable 3-parameter OLS fit.
# Set to 5 (slightly above the 3 minimum for forward price) because the
# quadratic requires enough curvature information across strikes.
_MIN_OBS: int = 5

# Factor column names - used throughout for consistency
_FACTORS: list[str] = ["gamma0", "gamma1", "gamma2"]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _add_log_moneyness(
    option_slice: pd.DataFrame,
    forward: float,
    vix: float,
    dte: int,
) -> pd.DataFrame:
    """
    Adds standardised log-moneyness k = log(K/F) / (σ√τ) following
    Carr and Wu (2003) with VIX as the constant benchmark volatility.

    k < 0 : OTM put territory
    k = 0 : exactly at-the-money
    k > 0 : OTM call territory

    Parameters
    ----------
    forward : forward price F for this (date, expiry)
    vix     : VIX level on this date (expressed as a percentage, e.g. 20.0)
    dte     : days to expiry for this expiry
    """
    s = option_slice.copy()
    sigma = vix / 100.0              # annualised volatility as a decimal
    tau   = dte / 365.0              # time to maturity in years
    s["log_moneyness"] = np.log(s["strike_price"] / forward) / (sigma * np.sqrt(tau))
    return s


def _linear_interp(
    val_short: float,
    val_long: float,
    dte_short: int,
    dte_long: int,
    target: int,
) -> float:
    """
    Linearly interpolates a scalar value between two bracketing maturities.

    Uses the same time-weighted scheme as bracket_expirations() so that
    IV curve factor maturities are constructed identically to VRP return
    maturities (variance_swap_returns.py, straddle_returns.py).

        interpolated = w_short * val(dte_short) + w_long * val(dte_long)
        w_short = (dte_long - target) / (dte_long - dte_short)
        w_long  = (target - dte_short) / (dte_long - dte_short)
    """
    if dte_short == dte_long:
        # Target exactly matches an available expiry; no interpolation needed
        return float(val_short)

    # Weight for the shorter expiry decreases as target moves toward dte_long
    w_short = (dte_long - target) / (dte_long - dte_short)
    # Weight for the longer expiry increases as target moves toward dte_long
    w_long  = (target - dte_short) / (dte_long - dte_short)

    return float(w_short * val_short + w_long * val_long)


# ── Core IV curve fitting functions ──────────────────────────────────────────

def fit_iv_curve(
    option_slice: pd.DataFrame,
    forward: float,
    vix: float,
    dte: int,
    min_obs: int = _MIN_OBS,
) -> dict | None:
    """
    Fits IV = γ₀ + γ₁·k + γ₂·k² via OLS for one (date, expiry),
    where k = log(K/F) / (σ√τ) is the Carr and Wu (2003) standardised
    moneyness scaled by the VIX benchmark volatility.

    Parameters
    ----------
    option_slice : options data for a single (date, expiry).
                   Required columns: strike_price, impl_volatility, best_bid.
    forward      : forward price F for this (date, expiry).
    vix          : VIX level on this date (percentage, e.g. 20.0).
    dte          : days to expiry for this (date, expiry).
    min_obs      : minimum number of valid options required (default 5).

    Returns
    -------
    dict with: gamma0, gamma1, gamma2, r_squared, n_obs
    or None if the fit is rejected (too few observations, non-positive ATM IV).
    """
    # Keep only options with a valid implied volatility AND a live bid.
    # Options with impl_volatility = NaN were rejected by OptionMetrics' solver
    # (typically very deep OTM options where IV is poorly identified).
    # Options with best_bid = 0 are stale quotes (already caught in cleaning
    # but good to double-check here for robustness).
    s = option_slice[
        option_slice["impl_volatility"].notna() &
        (option_slice["best_bid"] > 0)
    ].copy()

    if len(s) < min_obs:
        # Fewer than min_obs valid strikes → quadratic fit would be unreliable
        return None

    # Add log-moneyness column
    s = _add_log_moneyness(s, forward, vix, dte)

    # Build the OLS design matrix [1, k, k²]
    # This corresponds to the model: IV = γ₀·1 + γ₁·k + γ₂·k²
    k = s["log_moneyness"].to_numpy()          # log-moneyness vector
    y = s["impl_volatility"].to_numpy()        # implied volatility vector
    X = np.column_stack([
        np.ones(len(k)),    # constant term → γ₀ (ATM IV when k=0)
        k,                  # linear term   → γ₁ (slope / risk-reversal)
        k ** 2,             # quadratic term → γ₂ (curvature / butterfly)
    ])

    try:
        res = sm.OLS(y, X).fit()
    except Exception:
        # Catch any singular matrix or numerical error
        return None

    gamma0, gamma1, gamma2 = res.params

    # γ₀ must be strictly positive: it represents ATM implied volatility.
    # A non-positive γ₀ indicates a degenerate or data-quality-driven fit
    # that should not be passed downstream to factor construction.
    if gamma0 <= 0:
        return None

    return {
        "gamma0":    float(gamma0),      # Level (ATM implied vol)
        "gamma1":    float(gamma1),      # Slope (risk-reversal proxy)
        "gamma2":    float(gamma2),      # Curvature (butterfly proxy)
        "r_squared": float(res.rsquared),  # Goodness-of-fit of the quadratic
        "n_obs":     int(len(s)),        # Number of options used in this fit
    }


def implied_volatility_curve(
    df: pd.DataFrame,
    forward_df: pd.DataFrame,
    vix_series: pd.Series,
    min_obs: int = _MIN_OBS,
    min_r2:  float = 0.0,
) -> pd.DataFrame:
    """
    Fits the Zhang-Xiang IV curve for every (date, expiry) in the options data.

    !Apply add_columns from initial_clean.py first!
    !Apply filter_otm_atm after estimating forward prices!

    Parameters
    ----------
    df         : cleaned OTM+ATM options data.
                 Required columns: date, exdate, expiry, strike_price,
                 impl_volatility, cp_flag, best_bid.
    forward_df : forward price estimates from estimate_forward_curve().
                 Required columns: date, exdate, expiry, forward_price.
    vix_series : daily VIX levels indexed by date (percentage, e.g. 20.0).
                 Used to scale moneyness following Carr and Wu (2003).
    min_obs    : minimum options per (date, expiry) to attempt a fit.
    min_r2     : minimum acceptable R² for the quadratic fit (default 0.0).

    Returns
    -------
    DataFrame with one row per (date, expiry) where a valid fit was obtained.
    Columns: date, exdate, expiry, gamma0, gamma1, gamma2, r_squared, n_obs.
    """
    merged = df.merge(
        forward_df[["date", "exdate", "expiry", "forward_price"]],
        on=["date", "exdate", "expiry"],
        how="inner",
    )

    records = []

    for (dt, exd, dte), group in merged.groupby(["date", "exdate", "expiry"]):

        fwd = float(group["forward_price"].iloc[0])

        # Look up VIX for this date. Skip if unavailable — standardised
        # moneyness k = log(K/F) / (σ√τ) cannot be computed without it.
        if dt not in vix_series.index or np.isnan(vix_series[dt]):
            continue
        vix = float(vix_series[dt])

        result = fit_iv_curve(group, fwd, vix, int(dte), min_obs=min_obs)  # winsorization applied to factors downstream

        if result is None:
            # Fit rejected: too few options, non-positive ATM IV, or numerical error
            continue

        if result["r_squared"] < min_r2:
            # Fit rejected: R² below threshold (only active if min_r2 > 0)
            continue

        records.append({
            "date":      dt,
            "exdate":    exd,
            "expiry":    dte,              # days to expiry
            "gamma0":    result["gamma0"],
            "gamma1":    result["gamma1"],
            "gamma2":    result["gamma2"],
            "r_squared": result["r_squared"],
            "n_obs":     result["n_obs"],
        })

    # Sort chronologically then by maturity for easy downstream use
    out = (
        pd.DataFrame(records)
        .sort_values(["date", "expiry"])
        .reset_index(drop=True)
    )
    return out


# ── Constant-maturity interpolation ──────────────────────────────────────────

def interpolate_constant_maturity(
    iv_params: pd.DataFrame,
    target_maturities: dict[str, int] = TARGET_MATURITIES,
) -> pd.DataFrame:
    """
    Interpolates IV curve factors (γ₀, γ₁, γ₂) to constant maturities
    by linearly interpolating between the two bracketing available expiries.

    This mirrors exactly the interpolation used in variance_swap_returns.py
    and straddle_returns.py, ensuring that IV curve factor maturities and
    VRP return maturities are aligned on the same 30/60/90/120/180/360-day
    grid so that they can be matched for the regression analysis.

    Parameters
    ----------
    iv_params         : output of implied_volatility_curve(); one row per
                        (date, expiry) with columns gamma0, gamma1, gamma2.
    target_maturities : mapping from maturity label to days, e.g. {"1m": 30}.
                        Default matches TARGET_MATURITIES in this module.

    Returns
    -------
    DataFrame with columns: date, maturity, gamma0, gamma1, gamma2.
    One row per (date, maturity_label) where interpolation succeeds.
    """
    records = []

    # Process each trading day independently
    for dt, day_group in iv_params.groupby("date"):

        # Sort available expiries in ascending order for bracket search
        day_group    = day_group.sort_values("expiry").reset_index(drop=True)
        available_dtes = day_group["expiry"].to_numpy()  # array of available DTE values

        for label, target in target_maturities.items():

            # Find the two available expiries that bracket the target maturity.
            # bracket_expirations returns (dte_short, dte_long) where
            # dte_short <= target <= dte_long, or (target, target) for exact matches.
            bracket = bracket_expirations(available_dtes, target)

            if bracket is None:
                # Target falls outside the range of available expiries this day.
                # Drop this (date, maturity) rather than extrapolating.
                continue

            dte_short, dte_long = bracket

            # Retrieve the factor values at each bracketing expiry
            row_short = day_group[day_group["expiry"] == dte_short]
            row_long  = day_group[day_group["expiry"] == dte_long]

            if row_short.empty or row_long.empty:
                # Should not happen after groupby, but guard defensively
                continue

            row = {"date": dt, "maturity": label}

            # Interpolate each factor independently between the two brackets.
            # Uses the same time-weighted linear weights as the VRP return
            # construction to preserve maturity alignment.
            for f in _FACTORS:
                row[f] = _linear_interp(
                    val_short = float(row_short[f].iloc[0]),
                    val_long  = float(row_long[f].iloc[0]),
                    dte_short = int(dte_short),
                    dte_long  = int(dte_long),
                    target    = target,
                )

            records.append(row)

    out = (
        pd.DataFrame(records)
        .sort_values(["date", "maturity"])
        .reset_index(drop=True)
    )
    return out


# ── Term spread and hump factor construction ──────────────────────────────────

def compute_term_spreads(
    cm_factors: pd.DataFrame,
    spread_pairs: list[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """
    Computes term spreads for each IV curve factor across maturity pairs.

    A term spread is defined as:
        spread(long, short) = factor(long maturity) - factor(short maturity)

    For the level factor γ₀:
        spread > 0 → contango   (long-dated IV > short-dated IV; upward slope)
        spread < 0 → backwardation (short-dated IV > long-dated IV; inverted)

    This sign convention defines the regime indicator D_t in the regression:
        D_t = 1  if spread < 0  (backwardation)
        D_t = 0  if spread >= 0 (contango)

    Parameters
    ----------
    cm_factors   : output of interpolate_constant_maturity().
                   Columns: date, maturity, gamma0, gamma1, gamma2.
    spread_pairs : list of (long_label, short_label) tuples.
                   Default includes the key spreads from Guo et al. (2023)
                   plus the hump-region 180-60 spread from section 3.3.

    Returns
    -------
    Long-format DataFrame with columns:
        date, factor, long_mat, short_mat, spread, spread_label
    where spread_label is a unique string identifier for each combination,
    e.g. "gamma0_6m_1m" for the 180-30 day level spread.
    """
    if spread_pairs is None:
        # Default spread pairs - all standard combinations for the regression
        # "6m_1m"  = 180-day vs 30-day  (primary predictor in Guo et al.)
        # "12m_1m" = 360-day vs 30-day  (full term structure spread)
        # "12m_6m" = 360-day vs 180-day (long-dated spread)
        # "6m_2m"  = 180-day vs 60-day  (hump-region spread, section 3.3)
        # "4m_1m"  = 120-day vs 30-day  (medium-short spread)
        spread_pairs = [
            ("6m",  "1m"),
            ("12m", "1m"),
            ("12m", "6m"),
            ("6m",  "2m"),
            ("4m",  "1m"),
        ]

    # Pivot to wide format so each (factor, maturity) becomes one column.
    # wide.columns is a MultiIndex: (factor_name, maturity_label).
    wide = cm_factors.pivot(
        index="date",
        columns="maturity",
        values=_FACTORS,
    )

    records = []

    for factor in _FACTORS:
        for long_mat, short_mat in spread_pairs:

            # Check that both maturities exist in the pivoted data
            if (factor, long_mat) not in wide.columns:
                continue
            if (factor, short_mat) not in wide.columns:
                continue

            # Compute the full time series of this spread then winsorise
            # at 0.5% of the whole sample (Guo et al., 2023; Cao and Han, 2013).
            spread_series = (
                wide[(factor, long_mat)] - wide[(factor, short_mat)]
            ).dropna()
            spread_values = winsorize(spread_series.to_numpy(), limits=_WINSORISE_LIMITS)

            for dt, val in zip(spread_series.index, spread_values):
                records.append({
                    "date":         dt,
                    "factor":       factor,                   # e.g. "gamma0"
                    "long_mat":     long_mat,                  # e.g. "6m"
                    "short_mat":    short_mat,                 # e.g. "1m"
                    "spread":       float(val),
                    # Unique identifier for this (factor, pair) combination
                    "spread_label": f"{factor}_{long_mat}_{short_mat}",
                })

    out = (
        pd.DataFrame(records)
        .sort_values(["factor", "date"])
        .reset_index(drop=True)
    )
    return out


def compute_hump_factor(
    cm_factors: pd.DataFrame,
    mid_mat:   str = "6m",    # 180-day medium-term maturity
    short_mat: str = "1m",    # 30-day  short anchor
    long_mat:  str = "12m",   # 360-day long anchor
) -> pd.DataFrame:
    """
    Computes the hump factor: excess of the medium-term factor over the value
    predicted by linear interpolation between the short and long maturities.

        H_t = F_t(mid) − [w_short × F_t(short) + w_long × F_t(long)]

    Interpolation weights:
        w_short = (DTE_long − DTE_mid)  / (DTE_long − DTE_short)
        w_long  = (DTE_mid  − DTE_short) / (DTE_long − DTE_short)

    These weights ensure that the linear benchmark at the mid maturity is
    exact: w_short × DTE_short + w_long × DTE_long = DTE_mid ✓

    Economic interpretation:
        H_t > 0 : medium-term factor EXCEEDS linear interpolation
                  → the term structure is humped at the medium horizon
        H_t < 0 : medium-term factor BELOW the linearly interpolated value
                  → the term structure is concave at the medium horizon
        H_t = 0 : the term structure is exactly linear (no hump)

    The hump captures non-monotonic variance dynamics that level or slope
    spreads miss, and is the key novelty of the paper's section 3.3 analysis.

    Parameters
    ----------
    cm_factors : output of interpolate_constant_maturity().
    mid_mat    : maturity label for the medium-term point (default "6m" = 180 days).
    short_mat  : maturity label for the short anchor (default "1m" = 30 days).
    long_mat   : maturity label for the long anchor (default "12m" = 360 days).

    Returns
    -------
    DataFrame with columns: date, factor, hump_value, mid_mat, short_mat, long_mat.
    """
    # Retrieve the actual days-to-expiry for each maturity label to compute
    # the correct linear interpolation weights
    mid_days   = TARGET_MATURITIES[mid_mat]    # 180
    short_days = TARGET_MATURITIES[short_mat]  # 30
    long_days  = TARGET_MATURITIES[long_mat]   # 360

    # Linear interpolation weight on the short anchor.
    # As mid approaches long, w_short → 0 (less weight on short end).
    w_short = (long_days - mid_days) / (long_days - short_days)   # = 180/330 ≈ 0.545

    # Linear interpolation weight on the long anchor.
    # As mid approaches short, w_long → 0 (less weight on long end).
    w_long  = (mid_days - short_days) / (long_days - short_days)  # = 150/330 ≈ 0.455

    # Pivot cm_factors to wide format for easy column arithmetic
    wide = cm_factors.pivot(
        index="date",
        columns="maturity",
        values=_FACTORS,
    )

    records = []

    for factor in _FACTORS:

        # Check all three required maturities are present
        required = [mid_mat, short_mat, long_mat]
        if not all((factor, m) in wide.columns for m in required):
            continue

        # Linearly interpolated "benchmark" at the mid maturity under a
        # monotone term structure assumption
        interpolated = (
            w_short * wide[(factor, short_mat)] +
            w_long  * wide[(factor, long_mat)]
        )

        # Hump = actual mid-maturity factor minus the linearly interpolated value.
        # A positive value means the medium-term factor is elevated relative to
        # what pure linearity between short and long would predict.
        # Winsorise at 0.5% of the whole sample (Guo et al., 2023; Cao and Han, 2013).
        hump_series  = (wide[(factor, mid_mat)] - interpolated).dropna()
        hump_values  = winsorize(hump_series.to_numpy(), limits=_WINSORISE_LIMITS)

        for dt, val in zip(hump_series.index, hump_values):
            records.append({
                "date":       dt,
                "factor":     factor,
                "hump_value": float(val),
                "mid_mat":    mid_mat,         # metadata: which maturities were used
                "short_mat":  short_mat,
                "long_mat":   long_mat,
            })

    out = (
        pd.DataFrame(records)
        .sort_values(["factor", "date"])
        .reset_index(drop=True)
    )
    return out