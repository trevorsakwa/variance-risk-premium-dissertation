import numpy as np
import pandas as pd

# applies the forward_price equation as in Zhang and Xiang (2008) with OLS regression to derive the discount factor.

"""first the linear regression function"""
def linear_regression(x: np.ndarray, y: np.ndarray) ->tuple[float, float]:
    """used to calculate the linear regression that returns the 
    intercept and gradient as a and b respectively"""

    x_mean = x.mean()
    y_mean = y.mean()

    """recall slope coefficient: the change in y for one unit change in x
    cov(x,y)/var(x)"""
    b = np.sum((x - x_mean) * (y - y_mean))/ np.sum((x - x_mean) **2)
    """recall intercept term: the line's intersect with y-axis at x = 0"""
    a = y_mean - (b * x_mean)
    return b,a

def estimate_forward_curve(data: pd.DataFrame, min_pairs: int = 3) -> pd.DataFrame:

    """
    returns: DataFrame with columns:
    data, exdate, expiry, forward_price (following Zhang and Xiang (2008),
    forward_price_ols (forward_price derived through linear regression),
    discount_factor (from linear regression), and r_annualized (which is the implied continuous risk-free rate)
    data:!apply add_columns from initial_clean function first!

    assumes we have applied the initial_clean function to add the 
    "expiry" and the "mid_price" columns 
    it also assumes the date, and exdate data was converted to datetime and columns for cp_flag and strike_price are present.
    
    """
    calls = data[data["cp_flag"]=="C"][["date","exdate", "expiry","strike_price","mid_price"]]
    calls = calls.rename(columns={"mid_price" : "call_mid"})

    puts = data[data["cp_flag"] =="P"][["date","exdate", "expiry","strike_price","mid_price"]]
    puts = puts.rename(columns={"mid_price" : "put_mid"})

    #this is to divide the dataframe into call and put options respectively.
    combined = calls.merge(puts, on=["date","exdate","strike_price","expiry"], how="inner")

    # now we merge the calls and puts together when they have the same date, exdate, and expiry.
    combined["cp_diff"] = combined["call_mid"] - combined["put_mid"]

    records = []
    #create an empty dataframe
    for (dt,exd, dte), g in combined.groupby(["date","exdate","expiry"]):
        """we created a sub-dateset with the grouped options named g"""
        if len(g) < min_pairs:
            """recall the minimum number of option contracts to use is 3,
            with the same date, expiry date and maturity"""
            continue
        x = g["strike_price"].to_numpy()
        y = g["cp_diff"].to_numpy()

        b,a = linear_regression(x,y)

        discount_factor = -b
        # for situations when results are invalid
        if discount_factor<=0:
            continue #to skip invalid forward data

        forward_ols = a/discount_factor
        t_years = dte/ 365.0 #convert days to expiry to years
        r_annualized = -np.log(discount_factor)/ t_years if t_years> 0 else np.nan

        argument = np.argmin(np.abs(y))
        k_star = x[argument]
        cp_diff_star = y[argument]
        forward = k_star + cp_diff_star / discount_factor


        records.append((dt, exd, dte, forward, forward_ols, discount_factor, r_annualized))

    out = pd.DataFrame(records, columns=["date", "exdate", "expiry", 
                                         "forward_price", "forward_price_ols","discount_factor","r_annualized"])
    return out





