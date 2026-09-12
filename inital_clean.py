import pandas as pd

#following Neumann and Skiadopoulos (2013) cleaning methods
def add_columns(
        df: pd.DataFrame
) -> pd.DataFrame:
    """adds expiry (time to maturity, mid price and adjusts strike price)
    accounts for am_settlement/pm_settlement options"""
    df = df.copy()
    df["date"] = pd.to_datetime(df['date'])
    df["exdate"] = pd.to_datetime(df['exdate'])
    
    df["expiry"] = (df["exdate"] - df["date"]).dt.days - df["am_settlement"]
    df["strike_price"] = df["strike_price"]/1000
    df["mid_price"] = (df["best_offer"] + df["best_bid"]) /2

    return df

def cleaning(
        df: pd.DataFrame,
        min_mid: float = 0.20,
        min_expiry: int =7,
        min_strike: int = 3,

)-> pd.DataFrame:
    """follows the cleaning methods based on Neuman and SkiaDopoulos (2013)
    1. discard observations without an Implied volatility (ie rows with nan)
    2. exclude options with a maturity of fewer than 7 days
    3. discard observations with zero bid price and those that have a mid price less than 0.2 $
    4. remove contracts with less than three strike prices on a given day"""
    n0 = len(df)
    df = df[df["impl_volatility"].notna()] 
    n1 = len(df)
    df = df[df["expiry"]>= min_expiry]
    n2 = len(df)
    df = df[df["best_bid"] > 0]
    n3 = len(df)
    df = df[df["mid_price"]>= min_mid]
    n4 = len(df)

    #additional cleaning step for open interest and volume
    df = df[~((df["open_interest"] == 0) & (df["volume"] == 0))]
    n6 = len(df)

    strike = df.groupby(["date", "exdate"]) ["strike_price"].transform("nunique")
    df = df[strike >= min_strike]
    n5 = len(df)
    
    """dropping unnecessary columns"""
    df = df.drop(columns=['issuer','index_flag','exercise_style'])

    """for number of rows removed"""
    print("cleaning summary")
    print(f"initial rows: {n0}")
    print(f" after IV filter: removed:  {n0 - n1} rows")
    print(f" after maturity filter: removed: {n1 - n2} rows")
    print(f" after best bid filter : removed {n2 - n3} rows")
    print(f" after midprice filter: removed {n3-n4}")
    print(f" after strike price filter: removed {n4 - n6} rows")
    print(f"after zero OI/volume filter: removed {n6 - n5} rows")
    print(f"final test rows: {n6}")
    return df

def filter_otm_atm(df: pd.DataFrame,
                   forward: str = "forward_price"
                   ) -> pd.DataFrame:
    """
    filters for out-of-the-money and in-the-money options data
    first merge forward price data!
    """
    """
    given: 
    call options - ITM = strike < underlying (forward price)
    put options - ITM = strike > underlying (forward price) 

    keep only OTM and ATM options (drop in-the-money options)
    """
    is_call = df["cp_flag"] == "C"
    is_put = df["cp_flag"] == "P"
    n0 = len(df)



    otm_atm_call = is_call & (df["strike_price"] >= df[forward])
    otm_atm_put = is_put & (df["strike_price"] <= df[forward])

    out = df[otm_atm_put | otm_atm_call]

    print("filter summary")
    print(f"inital rows = {n0} rows")
    print(f"after filtering call option data = {otm_atm_call.sum()} rows")
    print(f"after filerting put option data = {otm_atm_put.sum()} rows")
    print(f"number of dropped rows = {n0 - len(out)}")
    print(f"total kept rows = {len(out)} rows")

    return out


    
