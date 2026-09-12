import pandas as pd
import numpy as np

def table1(
        df: pd.DataFrame
):
    """generates table activity data from cleaned options data
    seperated into maturities 0,30,90,180,360 and greater days to maturity
    """
    bins = [0, 30, 90, 180, 360, np.inf]
    labels = ['<30', '30-90', '90-180', '180-360', '>360']
    # this generates the labels and categorises the data

    df = df.copy()
    df['maturity_bucket'] = pd.cut(df['expiry'], bins=bins, labels=labels, right=False)
    #his creates a dataframe with the title maturity_bucket
    #pd.cut(data['expiration']),bins = , labels = , right= ) the cut function groups the data into pairs 
    #this creates a column and groups time to expiration according to the bins and labels

    def summarize(sub):
        daily = sub.groupby('date').agg(
            daily_oi=('open_interest','sum'),
            daily_volume=('volume','sum'),
            daily_strikes = ('strike_price','nunique'),
        )
        mean_oi = daily['daily_oi'].mean()
        mean_volume = daily['daily_volume'].mean()
        mean_strikes = daily['daily_strikes'].mean()

        std_oi = daily['daily_oi'].std()
        std_volume = daily['daily_volume'].std()
        std_strikes = daily['daily_strikes'].std()

        return pd.Series({
            'Mean open interest': mean_oi,
            'Stand.Dev open interest': std_oi,
            'CV open interest': (std_oi/mean_oi) * 100,
            'Mean trading volume': mean_volume,
            'Stand.Dev trading volume': std_volume,
            'CV trading volume': (std_volume/mean_volume) * 100,
            'Mean strike numbers': mean_strikes,
            'Stand.Dev strike numbers': std_strikes,
            'CV strike numbers': (std_strikes/mean_strikes) * 100,
    })
    overall = summarize(df).rename('Overall')
    by_bucket = df.groupby('maturity_bucket', observed=True).apply(summarize).T
    table_1 = pd.concat([overall, by_bucket], axis=1)[['Overall'] + labels]

    return table_1

