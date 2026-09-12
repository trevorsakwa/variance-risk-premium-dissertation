from __future__ import annotations
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats.mstats import winsorize
from constant_maturity import bracket_expirations

"""
this script is intended to impliment the second-order polynomial equation of Zhang and Xiang (2008).
for each (date,expiry) pair the model fits:
IV(k) = y0 (1+ y1.moneyness + y2.moneyness^2)

where moneyness is the logarithm of the strike price over the implied forward price scaled by the standard
deviation of the return on maturity.
"""

target_maturities = {
    "1m": 30,
    "2m": 60,
    "3m": 90,
    "4m": 120,
    "6m": 180,
    "12m": 360,
}

#this matches the variance_risk_premium_data.py target_maturities. critical !

winsorise_limits = (0.005, 0.005)
# applied to the term spreads and hump factors

factors = ["gamma0","gamma1","gamma2"]
#for implied volatility curve calculations and term spreads

def log_moneyness(
        option_slice: pd.DataFrame,
        implied_forward: float,
        vix: float,
        expiry: int,
) -> pd.DataFrame:
    """
    given moneyness is defined as:
    defined as the logarithm of the strike price over the 
    implied forward price scaled by the standard deviation 
    of the return on maturity following the industry notation 
    by Carr and Wu (2003) when using a constant benchmark volatility,

    recall: moneyness = log(K/F)/sigma*sqrt(tau)
    """
    slic = option_slice.copy()
    sigma = vix/ 100.0 # as vix is saved as a percentage
    tau = expiry / 365.0 #annualised time to maturity 
    slic["log_moneyness"] = np.log(slic["strike_price"] / implied_forward) / (sigma * np.sqrt(tau))

    return slic

def fit_implied_vol_curve(
        option_slice: pd.DataFrame,
        forward: float,
        vix: float,
        dte: int,
        min_obs: int = 5,
) -> dict | None:
    """
    fits implied volatility curve via ols for one (date, expiry)
    recall: IV(k) = y0 (1+ y1.moneyness + y2.moneyness^2)

    returns: gamma0, gamma1, gamma2, r_squared, n_obs
    (n_obs) = number of observations
    or none: if the fit is rejected (too few observations - set to 5 instead of 3)
    """
    slic = option_slice[option_slice["impl_volatility"].notna() &
                         (option_slice["best_bid"] > 0)].copy()
    
    if len(slic) < min_obs:

        return None
    
    slic = log_moneyness(slic,forward,vix,dte)

    moneyness = slic["log_moneyness"].to_numpy()
    y = slic["impl_volatility"].to_numpy()
    x = np.column_stack([
        np.ones(len(moneyness)),
        moneyness,
        moneyness **2,
    ])
    """
    constant term: y0
    linear term: y1
    quadratic term: y2
    """
    try:
        result = sm.OLS(y,x).fit()
    except Exception:
        return None

    gamma0, gamma1, gamma2 = result.params

    if gamma0 <=0:
        return None
    
    return{
        "gamma0": float(gamma0), #level (ATM implied volatility)
        "gamma1": float(gamma1), #slope (risk-reversal)
        "gamma2": float(gamma2), # curvature (butterfly)
        "r_squared": float(result.rsquared), # goodness-of-fit of the quadratic
        "n_obs": int(len(slic)), # number of options used
    }

def implied_volatility_curve(
        df: pd.DataFrame,
        forward_df: pd.DataFrame,
        vix_series: pd.Series,
        min_obs: int =5,
        min_r2: float = 0.0,
)-> pd.DataFrame:
    """
    fits the implied volatility curve for every (date,expriy ) in the options data.
    !apply add columns from inital_clean.py first!
    accounts for datetime
    df = pd.dateframe after fitler and add_columns and cleaning
    forward_df = pd.dataframe from forward_price.py

    returns: 
    dataframe with one row per(date, expiry)
    columns: date, exdate, expiry, gamma0, gamma1, gamma2m r_squared, n_obs
    """
    merged = df.merge(
        forward_df[["date","exdate","expiry","forward_price"]],
        on=["date","exdate","expiry"],
        how="inner",)
    
    merged["date"] = pd.to_datetime(merged["date"])
    merged["exdate"] = pd.to_datetime(merged["exdate"])
    
    records = []

    for (dt, exd, dte), group in merged.groupby(["date","exdate","expiry"]):

        fwd = float(group["forward_price"].iloc[0])

        if dt not in vix_series.index or np.isnan(vix_series[dt]):
            continue

        vix = float(vix_series[dt])

        result = fit_implied_vol_curve(group,fwd,vix,int(dte),min_obs=min_obs)

        if result is None:
            continue

        if result["r_squared"] < min_r2:
            continue

        records.append({
            "date": dt,
            "exdate": exd,
            "expiry": dte,
            "gamma0": result["gamma0"],
            "gamma1": result["gamma1"],
            "gamma2": result["gamma2"],
            "r_squared": result["r_squared"],
            "n_obs": result["n_obs"],

        })
    out = (pd.DataFrame(records).sort_values(["date","expiry"]).reset_index(drop=True))

    return out


def linear_interpolation(
        val_short: float,
        val_long : float,
        dte_short: int,
        dte_long : int,
        target : int,

):
    """
    linearly interpolates a value between the two bracket maturities

    applies the same time-weighted method as in bracket_expirations for
    consistency with VRP return data. 

    interpolated = w_short * val(dte_short) + w_long*val(dte_long)
    """

    if dte_short == dte_long:
        return float(val_short) #if the target exactly matches an available expiry then no interpolation required
    
    w_short = (dte_long - target) / (dte_long - dte_short)
    #weight for the shorter expiry options
    w_long = (target - dte_short) / (dte_long - dte_short)
    #weight for longer expiry increases as target moves towards dte_long

    return float(w_short * val_short + w_long * val_long)

def interpolate_constant_maturity(
        iv_params: pd.DataFrame,
        target_mat: dict[str,int] = target_maturities, 
)-> pd.DataFrame:
    """
    interpolates IV curve factors to constant maturities by linearly interpolating between available expiries
    mirrors interpolation used in variance_swap_returns.py and straddle_returns.py for consistency

    iv_params: output from implied_volatility_curve()
    target_maturities: set at 1-12 months
    returns Dateframe: date, maturitiy, gamma0,gamma1,gamma2
    """
    records = []

    for dt, day in iv_params.groupby("date"):

        day = day.sort_values("expiry").reset_index(drop=True)
        available_dte = day["expiry"].to_numpy()

        for label,target in target_mat.items():
            #find the two available expiries that bracket the target maturity
            bracket = bracket_expirations(available_dte, target)

            if bracket is None:
                continue

            dte_short, dte_long = bracket

            row_s = day[day["expiry"] == dte_short]
            row_l = day[day["expiry"] == dte_long]

            if row_s.empty or row_l.empty:
                continue

            row = {"date": dt, "maturity": label}

            #interpolate each factor independently
            for f in factors:
                row[f] = linear_interpolation(
                    val_short=float(row_s[f].iloc[0]),
                    val_long= float(row_l[f].iloc[0]),
                    dte_short= int(dte_short),
                    dte_long= int(dte_long),
                    target=target,
                )
            records.append(row)

    out = (pd.DataFrame(records).sort_values(["date","maturity"]).reset_index(drop=True))

    return out

def term_spreads(
        cm_factors: pd.DataFrame,
)-> pd.DataFrame:
    """
    computes the term spreads for each IV curve factor given maturity pairs

    factors: output of interpolate_constant_maturity()
    ("6m",  "1m"),    180 - 30: additional spread
    ("12m", "1m"),    360 - 30: full term structure spread
    ("12m", "6m"),    360 - 180: long-dated spread

    returns: date, factor, long_mat, short_mat, spread, spread_label
    """
    spread_pairs = [
        ("6m",  "1m"),   
        ("12m", "1m"),   
        ("12m", "6m"),   
    ]

    wide = cm_factors.pivot(index = "date", columns="maturity", values=factors)

    records = []

    for fact in factors:
        for long_maturity, short_maturity in spread_pairs:
            # quick check
            if (fact, long_maturity) not in wide.columns:
                continue
            if (fact, short_maturity) not in wide.columns:
                continue

            spread_series = (
                wide[(fact, long_maturity)] - wide[(fact, short_maturity)]
            ).dropna()

            spread_values = winsorize(spread_series.to_numpy(), limits=winsorise_limits)

            for dt, val in zip(spread_series.index, spread_values):
                records.append({
                    "date": dt,
                    "factor": fact,
                    "long_maturity": long_maturity,
                    "short_maturity": short_maturity,
                    "spread": float(val),
                    "spread_label": f"{fact}__{long_maturity}__{short_maturity}",

                })
    out = (pd.DataFrame(records).sort_values(["factor","date"]).reset_index(drop=True))

    return out

def hump_factor(
        cm_factors: pd.DataFrame,
        mid_mat: str = "6m",
        short_mat: str = "1m",
        long_mat: str = "12m",
)-> pd.DataFrame:
    """
    computes the rate-normaised hump factor
    H = [y(360) - y(180)]/ 180 - [y(180)-y(30)]/150

    """
    mid_days = target_maturities[mid_mat]
    short_days = target_maturities[short_mat]
    long_days = target_maturities[long_mat]

    span_long = long_days - mid_days
    span_short = mid_days - short_days

    wide = cm_factors.pivot(index="date", columns = "maturity", values=factors)

    records = []

    for fact in factors:
        required = [mid_mat,short_mat, long_mat]

        if not all((fact, m) in wide.columns for m in required):
            continue

        # per-day slope over the long segment
        slope_long = (wide[(fact, long_mat)] - wide[(fact, mid_mat)])/span_long
        
        #per-day slope over the short segment
        slope_short = (wide[(fact, mid_mat)] - wide[(fact, short_mat)])/ span_short

        # discrete second difference
        hump_series = (slope_long - slope_short).dropna()

        if hump_series.empty:
            continue

        hump_values = winsorize(hump_series.to_numpy(), limits=winsorise_limits)

        for dt, val in zip(hump_series.index, hump_values):
            records.append({
                "date": dt,
                "factor": fact,
                "hump_value": float(val),
                "mid_maturity": mid_mat,
                "short_maturity": short_mat,
                "long_maturity": long_mat,
            })
    out = (pd.DataFrame(records).sort_values(["factor","date"]).reset_index(drop=True))

    return out