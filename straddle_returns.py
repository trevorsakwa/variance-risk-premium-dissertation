import numpy as np
import pandas as pd
from constant_maturity import bracket_expirations

#following the methodology as in Coval and Shumway (2001)
#builds constant-maturity, at-the-money (ATM) S&P 500 straddle returns.

def select_atm_straddle(option_data: pd.DataFrame, forward: float) -> dict | None:
    """
    option_data: this is the options data for a single date and expiration
    forward: forward price for this (date, expiration) sourced from forward_price.py resutls

    picks the single strike closest to the forward price that has both a call and a put quoted, 
    and returns their prices. the function returns a dictionary with the strike price, both option
    IDs, and both midprices. 
    
    returns: NONE - if no strike has both legs available.
    """
    strikes = option_data["strike_price"].unique()
    if len(strikes) ==0:
        return None
    strikes = strikes[np.argsort(np.abs(strikes-forward))]
    """we reorder the strike prices such that those closest to the forward price
    come first"""

    for k in strikes:
        call = option_data[(option_data["strike_price"] == k) & (option_data["cp_flag"]=="C")]
        put = option_data[(option_data["strike_price"] == k) & (option_data["cp_flag"]=="P")]

        if not call.empty and not put.empty:
            return{
                "strike": k,
                "call_optionid": call["optionid"].iloc[0],
                "put_optionid": put["optionid"].iloc[0],
                "call_mid": call["mid_price"].iloc[0],
                "put_mid": put["mid_price"].iloc[0],
            }
    return None

def straddle_value(call_price: float, put_price: float) ->float:
    """
    returns the total value of a straddle position by simply adding the call price
    and the put price together as a single float. 
    as a straddle consists of buying or selling one call and one put at the same strike price
    and expiration.
    
    """
    return call_price + put_price

def revalue_next_day(position: dict, option_t1: pd.DataFrame) -> float:
    """
    revalue the given call and put contracts using day t+1 quotes. 
    this derives the next day value of the position. returns NaN if either 
    led has stopped quoting (the end of the sample data)
    """
    call_row = option_t1[(option_t1["optionid"] == position["call_optionid"]) & 
                         (option_t1["best_bid"] > 0)]
    put_row = option_t1[(option_t1["optionid"] == position["put_optionid"]) & 
                        (option_t1["best_bid"] > 0)]
    if call_row.empty or put_row.empty:
        return np.nan
    return float(call_row["mid_price"].iloc[0] + put_row["mid_price"].iloc[0])

def combo_weights_straddle(s1: int, s2: int, target: int)-> tuple[float, float]:
    """
    linear interpolation weights between the two bracketing straddle maturities

    """ 
    if s1 == s2:
        return 1.0, 0.0
    w1 = (s2 - target) / (s2 - s1)
    w2 = (target - s1) / (s2 - s1)
    return w1, w2

def constant_maturity_return(
        date_t_options: pd.DataFrame,
        date_t1_options: pd.DataFrame,
        forwards: pd.DataFrame,
        target_days: int,
) -> dict:
    
    available = date_t_options["expiry"].unique()
    bracket = bracket_expirations(available, target_days)
    
    if bracket is None:
        return { "ret": np.nan, "reason": "no_bracket"}
    s1, s2 = bracket

    legs = {}
    for s in set([s1, s2]):
        exp_row = forwards[forwards["expiry"] == s]
        if exp_row.empty:
            return {"ret": np.nan, "reason": f"no_forward :{s}"}
        fwd = exp_row["forward_price"].iloc[0]
        exdate = exp_row["exdate"].iloc[0]

        slice_t = date_t_options[date_t_options["expiry"] ==s]
        position = select_atm_straddle(slice_t, fwd)
        if position is None:
            return {"ret": np.nan, "reason": f"no_atm_pair {s}"}
        
        v_t = straddle_value(position["call_mid"], position["put_mid"])

        slice_t1 = date_t1_options[date_t1_options["exdate"] == exdate]
        v_t1 = revalue_next_day(position, slice_t1)

        legs[s] = {"v_t": v_t, "v_t1": v_t1}

    w1, w2 = combo_weights_straddle(s1, s2, target_days)
    v_t_combined = w1 * legs[s1]["v_t"] + w2 * legs[s2]["v_t"]
    v_t1_combined = w1 * legs[s1]["v_t1"] + w2 * legs[s2]["v_t1"]

    if np.isnan(v_t_combined) or np.isnan(v_t1_combined) or v_t_combined ==0:
        return {"ret": np.nan, "reason": "missing_leg_value"}
    
    ret = v_t1_combined / v_t_combined - 1.0
    return {"ret": ret, "s1": s1, "s2": s2, "w1": w1, "w2": w2}

    



