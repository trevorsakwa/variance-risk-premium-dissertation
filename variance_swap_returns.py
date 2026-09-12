import numpy as np
import pandas as pd
from constant_maturity import bracket_expirations

#method based on the work of Carr and Wu(2008)
#implementing the methodology as employed in Johnson(2017a) and Dew-Becker et al.(2017)
"""
given the variance swap payoff is replicated by holding a static,
continuously weighted portfolio of out-of-the-money (OTM) options comprising
of both put and call options across a continuum of strike prices expiring on the 
swaps's maturiy date
"""

def otm_strip_weights(option_slice: pd.DataFrame, forward: float)-> pd.DataFrame:
    """
    option_slice: options for a single (date, expiry)
    forward: forward price for this (date, expiry) derived from forward_price.py
    
    builds the portolio of out-of-the-money puts and calls

    using puts below the first strike under the forward price, calls above it and
    the average of put-call at that boundary strike. 
    returns: a dataframe as "optionid, strike_price, cp_flag_used, weight, and mid_price
    """

    s = option_slice[option_slice["best_bid"] > 0].copy()
    if s.empty:
        return s
    
    strikes = np.sort(s["strike_price"].unique())
    k0_candidates = strikes[strikes <= forward]

    if len(k0_candidates) ==0:
        k0 = strikes.min()
    else:
        k0 = k0_candidates.max()
    
    rows = []
    for i, k in enumerate(strikes):
        if k < k0:
            leg = s[(s["strike_price"] ==k) & (s["cp_flag"] == "P")]
            price = leg["mid_price"].mean() if not leg.empty else np.nan
        elif k > k0:
            leg = s[(s["strike_price"] ==k) & (s["cp_flag"] == "C")]
            price = leg["mid_price"].mean() if not leg.empty else np.nan
        else:
            leg_p = s[(s["strike_price"] ==k) & (s["cp_flag"] == "P")]["mid_price"]
            leg_c = s[(s["strike_price"] ==k) & (s["cp_flag"] == "C")]["mid_price"]
            vals = pd.concat([leg_p, leg_c])
            price = vals.mean() if not vals.empty else np.nan

        if np.isnan(price):
            continue

        if i ==0:
            dk = strikes[1] - strikes[0] if len(strikes) > 1 else 1.0
        elif i == len(strikes) -1:
            dk = strikes[-1] - strikes[-2]
        else:
            dk = (strikes[i + 1] - strikes[i -1])/ 2.0
        
        weight = dk / (k**2)

        side = "P" if k < k0 else ("C" if k > k0 else "PC")
        if side == "PC":
            for cp in ("P", "C"):
                leg = s[(s["strike_price"] == k) & (s["cp_flag"] == cp)]
                if not leg.empty:
                    rows.append((leg["optionid"].iloc[0], k, cp, weight/ 2.0, leg["mid_price"].iloc[0]))
        else:
            leg = s[(s["strike_price"] == k) & (s["cp_flag"] == side)]
            rows.append((leg["optionid"].iloc[0], k, side, weight, leg["mid_price"].iloc[0]))
    return pd.DataFrame(rows, columns=["optionid", "strike_price", "cp_flag_used", "weight", "mid_price"])

def portfolio_value(strip: pd.DataFrame, price_column: str = "mid_price")-> float:
    """
    provides the dollar value of the out-of-the-money strip portfolio
    """
    if strip.empty:
        return np.nan
    return float((strip["weight"] * strip[price_column]).sum())

def revalue_next_day(
        strip: pd.DataFrame, 
        options_t1: pd.DataFrame,
        minimum_coverage: float = 0.95)-> float:
    """
    revalues the same stripe using day t+1 quotes (next day)
    """

    merged = strip.merge(options_t1[["optionid", "mid_price","best_bid"]].rename(columns={"mid_price": "mid_price_t1","best_bid": "best_bid_t1"}),
                         on="optionid", how="left")
    valid = merged["mid_price_t1"].notna() & (merged["best_bid_t1"] >0)
    covered_weight = merged.loc[valid, "weight"].abs().sum()#############
    total_weight = merged["weight"].abs().sum()

    if total_weight ==0 or covered_weight/total_weight < minimum_coverage:
        return np.nan
    
    merged["mid_price_t1"] = merged["mid_price_t1"].fillna(0.0)
    merged.loc[~valid, "mid_price_t1"]=0.0

    
    return float((merged["weight"] * merged["mid_price_t1"]).sum())


def combo_weights_variance(s1: int, s2: int, target: int)-> tuple[float,float]:
    """given the relative weights, this generates the synthetic variance swap"""
    if s1 == s2:
        return 1.0, 0.0
    w1 = (s2 - target)/ (s2 -s1)
    w2 = (target - s1)/ (s2 - s1)
    return w1, w2

def constant_maturity_return(
        date_t_options: pd.DataFrame,
        date_t1_options: pd.DataFrame,
        forward: pd.DataFrame,
        target_days: int
) -> dict:
    available = date_t_options["expiry"].unique()
    bracket = bracket_expirations(available, target_days)

    if bracket is None:
        return {"ret": np.nan, "reason": "no_bracket"}
    s1, s2 = bracket

    legs = {}
    for s in set([s1,s2]):
        expiry_data = forward[forward["expiry"] == s]
        if expiry_data.empty:
            return {"ret": np.nan, "reason": f"no_forward {s}"}
        
        fwd = expiry_data["forward_price"].iloc[0]
        exdate = expiry_data["exdate"].iloc[0]

        slice_t = date_t_options[date_t_options["expiry"] == s]
        strip = otm_strip_weights(slice_t, fwd)

        if strip.empty:
            return {"ret": np.nan, "reason": f"empty strip {s}"}
        
        v_t = portfolio_value(strip)
        
        slice_t1 = date_t1_options[date_t1_options["exdate"] == exdate]
        v_t1  = revalue_next_day(strip, slice_t1)

        legs[s] = {"v_t": v_t, "v_t1": v_t1}

    w1, w2 = combo_weights_variance(s1, s2, target_days)
    v_t_combined = w1 * legs[s1]["v_t"] + w2 * legs[s2]["v_t"]
    v_t1_combined = w1 * legs[s1]["v_t1"] + w2 * legs[s2]["v_t1"]

    if np.isnan(v_t_combined) or np.isnan(v_t1_combined) or v_t_combined == 0:
        return {"ret": np.nan, "reason": "missing leg values"}
    ret = v_t1_combined / v_t_combined - 1.0

    return {"ret": ret, "s1": s1, "s2": s2, "w1": w1, "w2": w2}