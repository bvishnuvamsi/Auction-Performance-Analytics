# Auction Performance Analytics

> **Goal:** Analyze and model art auction performance using historical lot-level data (≈41K records) — replicating Auctions style analytics to understand value drivers, pricing patterns, and visual feature influence.

## Project Overview

This project explores real auction data to identify factors that drive artwork prices — spanning artist reputation, material type, artwork size, color composition, and temporal patterns.

The goal is to build a foundation for a predictive valuation model that mimics real-world auction house analytics.

## Project Structure

```bash
Auction-Performance-Analytics/
│
├── data.txt                     # Original data (data.txt)             
├── auction_cleaned.csv          # Cleaned and feature-engineered datasets
├── auction_model_readt.csv      # feature-engineered datasets for modeling
│ 
├── EDA.ipynb                    # Data cleaning + exploratory analysis
├── Feature_Engineering.ipynb    # Extracted Feartures for Modeling
├── Modeling.ipynb               # Applied different ML Models
├── dashboard.py                 # To view the basic Business processes
│
├── README.md
└── requirements.txt
```

## Environment Setup 
```bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install required packages
pip install -U pip pandas numpy matplotlib seaborn scikit-learn plotly streamlit jupyter
pip install tensorflow-macos tensorflow-metal
```
## Data Cleaning Summary

* **Raw Data:** 41,253 auction records, 23 columns
* **Key Columns:** `artist`, `country`, `year`, `price`, `material`, `height`, `width`, `dominantColor`, `brightness`, `soldtime`, etc.
* **Major Cleaning Steps:**
    * Dropped redundant column `Unnamed: 0`
    * Standardized column names (snake\_case)
    * Converted numeric columns (`height`, `width`, `area`, `price`, `brightness`, etc.)
    * Extracted `sold_year` from `soldtime`
    * Created missing value flags (`year_missing`, `sold_year_missing`)
    * Avoided median imputation for sparse `year` fields (40–56% missing) to preserve data integrity
    * Handled missing `height` and `width` using median imputation (6% missing)
    * Normalized artist names (`.title()` casing)
    * Cleaned categorical text (`material`, `country`, etc.)

## Exploratory Data Analysis (EDA)

### Price Distribution

* Prices follow a log-normal pattern — a few ultra-high sales skew the mean.
* Majority of lots fall between $2K–$50K.
* Log transformation normalized skew and revealed market center.

### Price by Material

* Medium type drives valuation volatility.
* Oil on canvas dominates volume; mixed media and works on paper show higher price variability.

### Size vs Price

* Weak–moderate correlation — larger area tends to fetch more, but artist brand > size in predicting price.

### Top Artists by Average Sale Price

* Mark Rothko, Francis Bacon, Vincent van Gogh dominate the high-value spectrum ($1M+).
* Confirms that a handful of blue-chip artists anchor auction house revenue.

### Average Price by Year

* Market peaks in late 1980s–early 1990s (historical boom period).
* Sparse post-2000 data due to missing `soldtime`.
* Suggests potential for forecasting if more temporal data becomes available.

### Feature Correlations

* Strong correlations: `height`–`width`–`area`.
* Weak correlation between `brightness` and `price`.
* Suggests simplifying redundant physical features for future models.

### Country-wise Average Price

* European & American artists lead high-value markets.
* Asian and Middle Eastern artists show undervalued growth potential.
* Detected anomalies (15thC, 16thC) — flagged for data quality improvement.

### Brightness vs Price

* No observable relationship; visual brightness doesn’t influence price.
* Market value determined more by artist, medium, and provenance than image luminance.

## Key Analytical Takeaways

* **Market Distribution:** Mid-tier sales dominate count; a few high-value lots drive profit.
* **Material Sensitivity:** Mixed media and works-on-paper show volatile, high-margin behavior.
* **Size vs Value:** Size affects price weakly; artist reputation dominates valuation.
* **Artist Concentration:** A few blue-chip artists drive the top-end; emerging artists may offer untapped upside.
* **Geographic Pattern:** Western art dominates; Asian markets emerging as undervalued.
* **Temporal Trend:** Historical cycles visible; more complete timestamps would strengthen trend forecasting.
* **Feature Correlations:** Redundant metrics (`height`/`width`/`area`) and weak visual correlations streamline model design.

## One-Line Summary (Quick Glance)

* Prices are log-normally distributed with high skew — most art sells mid-tier.
* Medium type drives price variance; mixed media has unpredictable highs.
* Size correlates weakly with price; reputation dominates.
* Few artists command most of the market; long tail sustains volume.
* Western markets dominate; Asian art undervalued.
* Brightness shows no predictive power — metadata trumps aesthetics.
* Feature redundancy identified for efficient model setup.

## Feature Engineering & Modeling

### **Feature Engineering Overview**

- **Parsed year columns:** Converted string-based `yearofbirth`, `yearofdeath` into numeric years using pattern extraction (`est`, `thC`, etc.).
- **Dropped `yearofdeath`:** High missingness (~22%) made imputation unreliable; removal did not affect performance.
- **Derived Features:**
  - `area` = `height × width`
  - `log_price` = `log(price + 1)` for normalized target distribution
  - `artist_score` = frequency encoding of artists to capture reputation
  - Binary flags: `year_missing`, `sold_year_missing`
- **Encoded Categoricals:** Applied label encoding to `country` and `material` for model compatibility.
- **Scaled Numeric Variables:** Normalized continuous features for models sensitive to magnitude (Linear Regression).

---

### **Modeling Approach**

Three regression models were compared to predict artwork sale price (log-transformed):

|       Model       |  RMSE ↓ |  R² ↑ |
|-------------------|---------|-------|
| Linear Regression |  1.588  | 0.433 |
|    Random Forest  |  1.229  | 0.661 |
|      XGBoost      |**1.092**| **0.732** |

**Interpretation:**
- **Linear Regression** underfits — can’t handle non-linear effects between artist, year, and size.
- **Random Forest** provides solid baseline performance with interpretable feature importances.
- **XGBoost** captures intricate feature interactions and dominates in predictive power.

---

### **Hyperparameter Tuning**

Performed `RandomizedSearchCV` (10 iterations, 3-fold CV) for `XGBRegressor` using parameters:
```python
{
    'n_estimators': [300, 500, 700],
    'max_depth': [6, 8, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.9, 1.0],
    'min_child_weight': [1, 3, 5]
}
```
### Tuning Results:

|       Phase       |  RMSE ↓ |  R² ↑ |
|-------------------|---------|-------|
|   Before Tuning   |  1.091  | 0.732 |
|After Tuning (Val) |  1.115  | 0.709 |
|After Tuning (Test)|  1.094  | 0.731 |

**Interpretation**:
- Hyperparameter tuning yielded marginal improvement which confirms that the default model was already well-optimized and robust.

### Model Diagnostics

- **Residuals**: Symmetric, centered at zero → no systematic bias.
- **Prediction vs Actual**: Strong diagonal alignment indicates good generalization.
- **Feature Importance**:
    - sold_year – auction recency major driver of price.
    - artist_score – reputation and visibility matter significantly.
    - material, area – intrinsic artwork characteristics contribute.
    - Dropping sold_year_missing had negligible effect, confirming redundancy.

---
### Final Model Selection

- Selected Model: Tuned XGBoost
- R²: ~0.73
- RMSE: ~1.09
- Reason: Balanced trade-off between interpretability and performance.


### Dashboard (Streamlit)

```bash
streamlit run app/dashboard.py
```

Filters: Year range (by sold_year), Artist (top 50), Material
KPI Cards:
- Total Sales (sum of price)
- Lots (count)
- Average Price
- Median Price
- Revenue Concentration – Top Artists’ Share (donut or horizontal bar)
- Country × Material – Average Price Heatmap (where specific media outperform)

Core Visuals (tabs):
- Price by Material (toggle: Total Sales / Average Price)
- Area vs Price (scatter; log-y)
- Top Artists by Average Sale Price (bar)
- Country-wise Average Price (bar)
- Price vs Brightness (scatter; log-y)

> Data assumptions: expects auction_cleaned.csv with price, artist, material, country, brightness, and sold_year (NaN/−1 allowed).

### Business Takeaways
- **Temporal Sensitivity**: Newer auctions fetch higher prices; market dynamics evolve rapidly.
- **Artist Reputation**: Strong positive price correlation; frequency encoding captures this well.
- **Physical Attributes**: Size and medium influence price but are secondary to provenance and artist.
- **Model Utility**: The trained model can forecast auction prices for new artworks, aiding valuation, bidding, and portfolio strategy.

### How to Reproduce

1. Run 01_EDA.ipynb → explores data / insights.
2. Run 02_Feature_Engineering.ipynb → saves auction_cleaned.csv, auction_model_ready.csv.
3. Run 03_Modeling.ipynb → trains models, prints metrics, saves model + features (optional).
4. streamlit run app/dashboard.py → interact with KPIs & visuals.

## References
---
Dataset is provided by github.com/ahmedhosny/theGreenCanvas 