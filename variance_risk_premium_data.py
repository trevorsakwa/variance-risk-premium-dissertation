from __future__ import annotations
import numpy as np
import glob
import pandas as pd
from forward_price import estimate_forward_curve
from scipy.stats.mstats import winsorize 
import straddle_returns as st
import variance_swap_returns as vs

# following the methodology as in Johnson (2017a)
# generates variance risk premium data
target_maturity_days = {
    "1m": 30,
    "2m": 60,
    "3m": 90,
    "4m": 120,
    "6m": 180,
    "12m": 360,
}
def load_cached(data_dir: str = "data") -> pd.DataFrame:
    files = sorted(glob.glob(f"{data_dir}/options_*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

def build_vrp(options: pd.DataFrame) -> pd.DataFrame:
    """
    """


    trading_days = sorted(options["date"].unique())
    daily_rows = []

    for i in range(len(trading_days) -1):
        t, t1 = trading_days[i], trading_days[i +1]
        opt_t = options[options["date"] ==t]
        opt_t1 = options[options["date"] ==t1]

        if opt_t.empty or opt_t1.empty:
            continue

        fwd_t = estimate_forward_curve(opt_t)
        if fwd_t.empty:
            continue

        for label, target_days in target_maturity_days.items():
            variance_swap_results = vs.constant_maturity_return(opt_t,opt_t1,fwd_t,target_days)
            straddle_results = st.constant_maturity_return(opt_t,opt_t1,fwd_t,target_days)

            daily_rows.append({
                "date": t,
                "next_date": t1,
                "maturity": label,
                "variance_swap_return_daily": variance_swap_results["ret"],
                "straddle_return_daily": straddle_results["ret"]
            })
    
    panel = pd.DataFrame(daily_rows).sort_values(["maturity","date"]).reset_index(drop=True)
    return panel

def compounded_returns(
        panel: pd.DataFrame, 
        weekly_window: int = 5,
        monthly_window: int = 21) -> pd.DataFrame:
    
    """
    compounds daily log-returns forward to produce next-week (5-days)
      and next-month (21-days) horizon returns.

      includes winsorizing daily returns at 0.5% similar to in term spreads
      to prevent extreme outliers from contaminating weekly/monthly series
      specifically values that are not nan.

    """
    out = []
    
    for mat, g in panel.groupby("maturity"):
        g = g.sort_values("date").reset_index(drop=True)

        for colum in ["variance_swap_return_daily","straddle_return_daily"]:
            apply = g[colum].notna()
            if apply.sum() > 10:
                g.loc[apply, colum] = winsorize(g.loc[apply,colum].to_numpy(), limits=(0.005,0.005))

            
        vs_log = np.log1p(g["variance_swap_return_daily"])
        st_log = np.log1p(g["straddle_return_daily"])

        g["variance_swap_return_weekly"] = np.expm1(
            vs_log.rolling(weekly_window).sum().shift(-(weekly_window - 1))
        )
        g["straddle_return_weekly"] = np.expm1(
            st_log.rolling(weekly_window).sum().shift(-(weekly_window -1))
        )

        g["variance_swap_return_monthly"] = np.expm1(
            vs_log.rolling(monthly_window).sum().shift(-(monthly_window -1))
        )
        g["straddle_return_monthly"] = np.expm1(
            st_log.rolling(monthly_window).sum().shift(-(monthly_window - 1))
        )
        out.append(g)
    return pd.concat(out, ignore_index=True)

