from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import binomtest

#newey-west lag selection

def nw_lag(
        horizon: str
) -> int:
    """
    returns the Newey-west lag appropriate for each return horizon
    """
    mapping = {"daily": 1, "weekly": 4,"monthly":20}
    return mapping[horizon]

# regime indicator

def regime_indicator(
        x: pd.Series
):
    """
    constructs binary regime indicator Dt
    Dt = 1 if Xt <0 (backwardation: short-dated IV > long-dated IV)
    Dt = 0 if Xt>=0 (contango: long-dated IV >= short-dated IV)
    """
    Dt = (x<0).astype(float)
    Dt.name = "Dt"
    return Dt

#in-sample regression
def in_sample_regression(
        y: pd.Series,
        x: pd.Series,
        Dt: pd.Series,
        horizon: str="daily",
        lags: int | None=None,
)-> dict:
    """
    estimates the regime-conditional regression
    Y: return series (variance swap or straddle)
    x: term spread or hump factor
    Dt:regime indicator
    horizon: daily, then compounded for weekly and monthly
    lags: to override NW lag count if necessary

    returns: 
    dict: coefficient estimators, NW t-statistics, p-values, R^2, n_obs, 
    and regime-specific intercepts and slopes
    """
    if lags is None:
        lags = nw_lag(horizon)
    
    data = pd.DataFrame({"y":y,"x":x,"Dt":Dt}).dropna()

    if len(data) < 30:
        #too few observations
        return {"error": "insufficient_no._observations","n_obs": len(data)}
    
    y_arr = data["y"].to_numpy()
    x_arr = data["x"].to_numpy()
    Dt_arr= data["Dt"].to_numpy()

    reg_x = np.column_stack([
        np.ones(len(y_arr)), #constant
        x_arr,   # x_t
        Dt_arr,  # Dt
        x_arr * Dt_arr, #X_t.Dt 
    ])

    #OLS with Newey-West HAC covariance matrix.
    model = sm.OLS(y_arr, reg_x)
    results = model.fit(cov_type="HAC",
                        cov_kwds={"maxlags": lags,"use_correction":True})
    
    #unpack the estimated parameters
    alpha, beta1, beta2, beta3 = results.params
    t_alpha, t_beta1, t_beta2, t_beta3 = results.tvalues
    p_alpha, p_beta1, p_beta2, p_beta3 = results.pvalues

    return{
        #coefficient estimates
        "alpha": float(alpha),
        "beta1": float(beta1),
        "beta2": float(beta2),
        "beta3": float(beta3),
        #newey-west t-statistics
        "t_alpha": float(t_alpha),
        "t_beta1": float(t_beta1),
        "t_beta2": float(t_beta2),
        "t_beta3": float(t_beta3),
        #tw0-tailed p-values
        "p_alpha": float(p_alpha),
        "p_beta1": float(p_beta1),
        "p_beta2": float(p_beta2),
        "p_beta3": float(p_beta3),
        # goodness of fit
        "r_squared": float(results.rsquared),
        "n_obs": int(len(y_arr)),
        #regime specific decomposition-contango
        "contango_intercept": float(alpha),
        "contango_slope": float(beta1),
        #backwardation:
        "backwardation_intercept": float(alpha + beta2),
        "backwardation_slope": float(beta1+ beta3),
    }

#evaluation metrics
def r2_out_of_sample(
        y_oss: np.ndarray,
        y_hat: np.ndarray,
        y_bar_Insample: float,
)->float:
    """
    campbell-thompson (2008) out-of-sample R^2:
    R^2_os = 1 - model squared errors/ historical mean benchmark

    R^2_os > 0: model beats the historical mean (positive predictive content)
    R^2_os < 0: model is worse than the historical mean

    y_oss: realised out-of-sample returns
    y_hat: model out-of-sample forecasts
    y_bar_insample: in-sample historical mean
    """
    SS_model = np.sum((y_oss-y_hat)**2) #model sum of squared errors
    SS_benchmark = np.sum((y_oss - y_bar_Insample)**2) #historical mean benchmark sse

    if SS_benchmark ==0:
        return np.nan
    
    return float(1.0 - SS_model/SS_benchmark)

def clark_west_test(
        y_oss: np.ndarray,
        y_hat: np.ndarray,
        y_bar_Insample: float,
        lags: int,
)-> tuple:
    """
    clark-west (2007) Mean squared predicted error-adjusted statistic (MSPE) 
    review thresholds:

    > 1.282 -> p<0.10
    > 1.645 -> p<0.05
    > 2.326 -> p<0.01

    returns: (cw_statistic, one_tailed_p_value)

    Fix: accounts for error whereby empty out of sample arrays crashed the sm.OLS with zero-size array error. 
    """
    if len(y_oss) == 0 or len(y_hat) == 0:
        return np.nan, np.nan
    

    benckmark_error = y_oss - y_bar_Insample
    model_error = y_oss - y_hat

    adjustment = (y_bar_Insample - y_hat) **2

    F = benckmark_error**2 - (model_error**2 - adjustment)

    #test whether mean(F) > 0 by regressing F on a constant and reading off the
    # HAC-robust t-statistic

    ols = sm.OLS(F, np.ones(len(F)))
    results = ols.fit(cov_type="HAC",cov_kwds={"maxlags": lags,"use_correction": True})

    cw_statistic = float(results.tvalues[0])

    p_twotail = float(results.pvalues[0])
    p_onetail = (p_twotail/2.0) if cw_statistic > 0 else (1.0 - p_twotail/ 2.0)
    
    return cw_statistic, p_onetail

def directional_accuracy(
        y_oss: np.ndarray,
        y_hat: np.ndarray,
        horizon: str = "daily",
)-> dict:
    """
    directional accuracy (DA): fraction of out-of-sample periods where the model
    correctly predicted the sign (negative/positive) of the variance-sensitive asset

    DA = (1/N) x summation(1[sign(R-outofsample)] = sign(r-insample))

    significance is assessed using a one-tailed binomial test:
    hypothesis-0: DA = 0.5 (no directional skill, coin toss)
    hypothesis-1: DA > 0.5 (genuine directional skill)

    returns: Directional_Accuracy (hit-rate), n_correct, n_total, binomial_p_value
    """
    y_arr = np.asarray(y_oss, dtype=float)
    yh_arr = np.asarray(y_hat,dtype=float)

    step = {"daily": 1,"weekly": 5, "monthly": 21}[horizon]
    y_arr = y_arr[::step]
    yh_arr = yh_arr[::step]
    #remove nan pairs
    mask = ~(np.isnan(y_arr) | np.isnan(yh_arr))
    y_arr = y_arr[mask]
    yh_arr = yh_arr[mask]

    n_total = len(y_arr)
    #check
    if n_total == 0:
        return {"Directional_Accuracy": np.nan,"n_correct": 0,"n_total": 0,"binomial_p_value": np.nan}
    
    correct = (np.sign(yh_arr) == np.sign(y_arr)) & (np.sign(yh_arr) != 0)
    n_correct = int(correct.sum())
    Directional_Accuracy = float(n_correct/n_total)

    #binomial test: 
    result = binomtest(n_correct, n_total, p=0.5, alternative="greater")

    return {
        "Directional_Accuracy": Directional_Accuracy,
        "n_correct": n_correct,
        "n_total": n_total,
        "binomial_p_value": float(result.pvalue),
    }

def out_of_sample_test(
        y: pd.Series,
        x: pd.Series,
        Dt: pd.Series,
        train_end: str="2015-12-31",
        horizon: str = "daily",
        lags: int| None = None,
)-> dict:
    """
    fixed split out-of-sample evaluation 

    fixed CW OLS error
    
    recall: 
        in-sample: Jan 2000 - Dec 2015
        out-of-sample: Jan 2016 - July 2025
        **can change cut date. 
    """

    if lags is None:
        lags = nw_lag(horizon)
    
    #drop possible nan rows
    data = pd.DataFrame({"y":y, "x":x,"Dt":Dt}).dropna()
    data.index = pd.to_datetime(data.index)

    #split at the fixed cut date
    split = pd.Timestamp(train_end)
    is_data = data[data.index <= split]  #in-sample: 2000-2015
    out_of_sample_data = data[data.index > split] #out-of-sample: 2016 to end of dataset

    if len(is_data)<30:
        return {"error present": "insufficient insample data","N_is": len(is_data),"N_obs": len(out_of_sample_data)}
    if len(out_of_sample_data) < 10:
        return {"error present": "insufficient out of sample data","N_is": len(is_data),"N_obs": len(out_of_sample_data)}


    #estimate in-sample regression
    in_sample_results = in_sample_regression(
        y = is_data["y"],
        x = is_data["x"],
        Dt= is_data["Dt"],
        horizon = horizon,
        lags = lags
    )

    if "error" in in_sample_results:
        return in_sample_results

    #in-sample historical mean
    y_bar_in_sample = float(is_data["y"].mean())

    #out-of-sample forecasts using in-sample coefficients
    alpha = in_sample_results["alpha"]
    beta1 = in_sample_results["beta1"]
    beta2 = in_sample_results["beta2"]
    beta3 = in_sample_results["beta3"]

    #predictor, regime indicator and realised returns in out-of-sample period
    x_out_of_sample = out_of_sample_data["x"].to_numpy()
    Dt_out_of_sample = out_of_sample_data["Dt"].to_numpy()
    y_out_of_sample = out_of_sample_data["y"].to_numpy()

    #out of sample forecast:
    y_hat_out_of_sample = (
        alpha + beta1 * x_out_of_sample 
        + beta2 * Dt_out_of_sample 
        + beta3 * x_out_of_sample * Dt_out_of_sample

    )

    # out-of-sample evaluation metrics
    r2_oss = r2_out_of_sample(y_out_of_sample, y_hat_out_of_sample, y_bar_in_sample)
    cw_statistic, cw_p_value = clark_west_test(y_out_of_sample, y_hat_out_of_sample,y_bar_in_sample,lags)
    direct_accuracy = directional_accuracy(y_out_of_sample,y_hat_out_of_sample,horizon)

    return {
        #insample coefficient results
        **{f"is_{k}": v for k, v in in_sample_results.items()},
        #benchmark
        "y_benchmark_in_sample": y_bar_in_sample,
        #r2_out_of_sample
        "n_out_of_sample": int(len(y_out_of_sample)),
        "r2_out_of_sample": r2_oss,
        #ew Statistic and one-tailed p_value_metrics
        "clark-west_statistic": cw_statistic,
        "clark-west_p_value": cw_p_value,
        #directional accuracy and binomial p-value
        "directional_accuracy": direct_accuracy["Directional_Accuracy"],
        "directional_accuracy_n_correct": direct_accuracy["n_correct"],
        "directional_accuracy_n_total": direct_accuracy["n_total"],
        "directional_accuracy_p_value": direct_accuracy["binomial_p_value"],
    }

def run_full_analysis(
        vrp_panel: pd.DataFrame,
        term_spreads: pd.DataFrame,
        hump_factors: pd.DataFrame,
        train_end: str="2015-12-31",
        assets: list |None=None,
        horizons: list |None=None,
):
    """
    runs the entire analysis at once. to be used to verify consistency
    """
    if assets  is None:
        assets = ["variance_swap","straddle"]
    if horizons is None:
        horizons = ["daily","weekly","monthly"]
    
    asset_column = {
        "variance_swap":"variance_swap_return",
        "straddle":"straddle_return",
    }

    maturities = vrp_panel["maturity"].unique()

    spread_predictors = {}
    for label, group in term_spreads.groupby("spread_label"):
        spread_series = group.set_index("date")["spread"]
        spread_predictors[str(label)] = spread_series

    hump_predictors = {}
    for factor, group in hump_factors.groupby("factor"):
        hump_series = group.set_index("date")["hump_value"]
        hump_predictors[f"hump_{factor}"] = hump_series
    
    #combines all predictors into one dictionary

    all_predictors = {**spread_predictors, **hump_predictors}

    all_results = []

    for asset in assets:
        column_prefix = asset_column[asset]

        for mat in maturities:
            maturity_subset = vrp_panel[vrp_panel["maturity"] == mat].set_index("date")

            for horizon in horizons:
                ret_column = f"{column_prefix}_{horizon}"

                if ret_column not in maturity_subset.columns:
                    continue

                #return series
                y = maturity_subset[ret_column].dropna()
                if len(y) < 30:
                    continue

                for predictor_label, x in all_predictors.items():
                    Dt = regime_indicator(x)

                    results = out_of_sample_test(
                        y=y,
                        x=x,
                        Dt=Dt,
                        train_end=train_end,
                        horizon=horizon,

                    )
                    row = {
                        "asset": asset,
                        "maturity": mat,
                        "horizon": horizon,
                        "predictor": predictor_label,
                    }
                    row.update(results)
                    all_results.append(row)

    results_data = pd.DataFrame(all_results)
    return results_data

