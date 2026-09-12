# Variance Risk Premium and the Implied Volatility Term Structure
### Regime-Conditional Dynamics and Hump Effects

**MSc Finance and Investment Management Dissertation**
University of Liverpool Management School

**Author:** Trevor Sakwa Timothy
**Student ID:** 201843465
**Sample period:** January 2000 – July 2025

---

## Overview

This repository contains the full codebase for the dissertation *"Variance Risk Premium and the Implied Volatility Term Structure: Regime-Conditional Dynamics and Hump Effects"*, which investigates whether the shape of the S&P 500 implied volatility term structure predicts the variance risk premium across maturities and market regimes.

The study extends Guo et al. (2023) in three dimensions:

- **Extended sample** — January 2000 to July 2025, incorporating Volmageddon (2018), COVID-19 (2020), and the 2022 Federal Reserve tightening cycle
- **Regime-conditional framework** — a binary indicator distinguishing contango from backwardation, allowing the predictive slope to vary across market states
- **Hump factor** — a rate-normalised discrete second difference capturing non-linear medium-term curvature dynamics that endpoint-based spreads cannot detect

---

## Repository Structure

```
├── data/                          # Data files (not tracked — see Data section)
├── figures/                       # Output figures (PDF/PNG)
│
├── inital_clean.py                # Data cleaning and OTM/ATM filtering
├── forward_price.py               # Forward price estimation via put-call parity (Zhang & Xiang K* method)
├── constant_maturity.py           # Bracket expirations for constant-maturity interpolation
├── variance_swap_returns.py       # Synthetic variance swap return construction
├── straddle_returns.py            # ATM straddle return construction
├── variance_risk_premium_data.py  # VRP panel builder and return compounding
├── Zhang_and_Xiang.py             # IV curve fitting, term spreads, and hump factor
├── regression_analysis.py         # In-sample and out-of-sample regression framework
│
├── sourcing.ipynb                 # Data sourcing, cleaning, and validation
├── full_study.ipynb               # Main analysis notebook
├── review_johnson.ipynb           # Johnson (2017) benchmark comparison
├── vrp_data.ipynb                 # VRP return construction and diagnostics
│
└── README.md
```

---

## Data Requirements

This codebase requires three external data sources. **None are included in this repository** — they must be sourced independently.

### 1. S&P 500 Options Data — OptionMetrics IvyDB via WRDS
- **Source:** Wharton Research Data Services (WRDS) → OptionMetrics → IvyDB US
- **File:** `cleaned_research_data.csv` — cleaned and filtered SPX option data
- **Variables required:** `date`, `exdate`, `cp_flag`, `strike_price`, `best_bid`, `best_offer`, `volume`, `open_interest`, `impl_volatility`, `optionid`, `am_settlement`
- **Sample:** January 2000 – July 2025

### 2. CBOE VIX Daily Levels — WRDS CBOE Index Data
- **Source:** WRDS → CBOE Indexes → `cboe.cboe` table, variable `vix` (closing level)
- **File:** `vix-2000-2025-prepped.csv` — deduplicated daily VIX series
- **Format required:** Two columns (`date`, `vix`), date as index, no NaN values

### 3. Johnson (2017) Benchmark Data — Author's Website
- **Source:** Travis Johnson's faculty page — *Risk Premia and the VIX Term Structure*
- **Files:** `Johnson_straddle_returns.csv`, `Johnson_variance_swap_return.csv`
- **Used for:** Validation of VRP return construction against the published benchmark

---

## Installation

This project requires Python 3.12+.

```bash
# Clone the repository
git clone https://github.com/trevorsakwa/variance-risk-premium-dissertation.git
cd variance-risk-premium-dissertation

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
.venv\Scripts\activate           # Windows

# Install dependencies
pip install pandas numpy scipy statsmodels matplotlib seaborn pyarrow jupyter
```

---

## Pipeline

The analysis runs in five sequential stages. Each stage saves its output so subsequent stages can be run independently without recomputing earlier steps.

### Stage 1 — Data Cleaning (`sourcing.ipynb`)

```python
from inital_clean import add_columns, cleaning, filter_otm_atm
from forward_price import estimate_forward_curve

data     = pd.read_csv("data/raw_options.csv")
data     = add_columns(data)
cleaned  = cleaning(data)

# Save pre-filter data for VRP construction
cleaned.to_csv("data/cleaned_research_data.csv", index=False)

# Estimate forward prices on full cleaned data
forward_df = estimate_forward_curve(cleaned)

# Apply OTM/ATM filter for IV curve construction
filtered = filter_otm_atm(cleaned.merge(forward_df[...], on=[...]))
filtered.drop(columns=["forward_price"]).to_csv("data/filtered_options.csv", index=False)
```

### Stage 2 — VRP Returns (`vrp_data.ipynb`)

```python
from variance_risk_premium_data import build_vrp, compounded_returns

data      = pd.read_csv("data/cleaned_research_data.csv", parse_dates=["date","exdate"])
vrp_daily = build_vrp(data)                    # ~3.5 hours for full 25-year sample
vrp_panel = compounded_returns(vrp_daily)      # adds weekly and monthly horizons
vrp_panel.to_csv("data/vrp_panel_data.csv", index=False)
```

### Stage 3 — IV Curve and Factors (`full_study.ipynb`)

```python
from Zhang_and_Xiang import (implied_volatility_curve,
                              interpolate_constant_maturity,
                              term_spreads, hump_factor)

vix_series    = pd.read_csv("data/vix-2000-2025-prepped.csv",
                             index_col=0, parse_dates=True)["vix"]
iv_panel      = implied_volatility_curve(filtered, forward_df, vix_series)
cm_factors    = interpolate_constant_maturity(iv_panel)
spreads       = term_spreads(cm_factors)
humps         = hump_factor(cm_factors)
```

### Stage 4 — Regression Analysis (`full_study.ipynb`)

```python
from regression_analysis import run_full_analysis

results = run_full_analysis(
    vrp_panel    = vrp_panel,
    term_spreads = spreads,
    hump_factors = humps,
    train_end    = "2015-12-31",      # in-sample: 2000–2015, OOS: 2016–2025
)
results.to_csv("data/overall_study_results.csv", index=False)
```

### Stage 5 — Results Tables and Figures (`full_study.ipynb`)

All tables and figures are generated from `overall_study_results.csv` and the intermediate data files. See `full_study.ipynb` for the full output.

---

## Key Results

| Finding | Detail |
|---|---|
| **In-sample** | Short-end level spread $\gamma_0^{180-30}$ significantly predicts monthly straddle returns at 3m–6m maturities (p < 0.05) |
| **Regime sign reversal** | Long-end spread $\gamma_0^{360-180}$ produces a complete slope sign reversal at 12m between contango (+1.894) and backwardation (−3.674) — only detectable via the regime-conditional framework |
| **Hump factor** | Level hump $H_0$ significant in-sample at 2m, 3m, 6m, and 12m straddle maturities; fails OOS |
| **OOS result** | $\gamma_0^{360-180}$ at 12m achieves $R^2_{OS} = 0.051$, Clark-West = 2.707 (p = 0.003) — the study's strongest OOS result |
| **Directional accuracy** | DA = 70.5% (p = 0.001) at 1m straddle OOS — model correctly identifies return sign despite negative $R^2_{OS}$ |

---

## Methodology Notes

**Forward price estimation** follows the Zhang and Xiang (2008) / CBOE convention: $K^* = \arg\min_K |C(K) - P(K)|$, then $F = K^* + (C(K^*) - P(K^*))/\text{discount factor}$. The discount factor is estimated from the OLS slope across all matched call-put pairs, avoiding the need for an external risk-free rate series.

**Moneyness standardisation** uses the VIX as the constant benchmark volatility $\sigma$ in $\xi = \ln(K/F_0)/(\sigma\sqrt{\tau})$, following Carr and Wu (2003).

**Regime indicator** $D_t \in \{0,1\}$ is defined from the sign of the level spread $\gamma_0^{180-30}$. Slope and curvature spreads are negative on average on 67–90% of days (structural skew steepening) and are used only as continuous predictors $X_t$, not as regime signals.

**Newey-West lags:** 1 lag (daily), 4 lags (weekly), 20 lags (monthly) — matching the compounding window length.

**Winsorisation:** All term spread and hump factor predictors are winsorised at the 0.5% whole-sample level following Guo et al. (2023) and Cao and Han (2013). VRP returns are not winsorised.

---

## References

- Guo, H., Legerstee, R., and Vivian, A. (2023). *Predictability of variance risk premia by the implied volatility smirk.* Journal of Empirical Finance.
- Johnson, T. (2017). *Risk premia and the VIX term structure.* Journal of Financial and Quantitative Analysis, 52(6), 2461–2490.
- Zhang, J. E., and Xiang, Y. (2008). *The implied volatility smirk.* Quantitative Finance, 8(3), 263–284.
- Carr, P., and Wu, L. (2009). *Variance risk premiums.* Review of Financial Studies, 22(3), 1311–1341.
- Campbell, J. Y., and Thompson, S. B. (2008). *Predicting excess stock returns out of sample.* Review of Financial Studies, 21(4), 1509–1531.
- Clark, T. E., and West, K. D. (2007). *Approximately normal tests for equal predictive accuracy.* Journal of Econometrics, 138(1), 291–311.
- Dew-Becker, I., Giglio, S., Le, A., and Rodriguez, M. (2017). *The price of variance risk.* Journal of Financial Economics, 123(2), 225–250.

---

## License

This repository is shared for academic reference purposes. The code is freely reusable with attribution. The underlying data (OptionMetrics, CBOE) is subject to the respective vendors' terms of use and is not redistributed here.

---

*For questions about the methodology or codebase, contact via GitHub.*
