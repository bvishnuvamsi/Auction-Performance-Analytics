# Auction Performance Analytics

> **Goal:** Analyze and model art auction performance using historical lot-level data (≈41K records) — replicating Auctions style analytics to understand value drivers, pricing patterns, and visual feature influence.

## Project Overview

This project explores real auction data to identify factors that drive artwork prices — spanning artist reputation, material type, artwork size, color composition, and temporal patterns.

The goal is to build a foundation for a predictive valuation model that mimics real-world auction house analytics.

## Project Structure

```bash
Auction-Performance-Analytics/
│
├── data.txt            # Original data (data.txt)             
├── processed/          # Cleaned and feature-engineered datasets
│ 
├── 01_EDA.ipynb        # Data cleaning + exploratory analysis
│
├── app/
│   └── dashboard.py        # (To be added in later phase)
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

## Next Steps (Feature Engineering & Modeling)

The next phase will focus on:

* Creating log-transformed target variable (`log_price`) for stable modeling.
* Encoding categorical features (`artist`, `material`, `country`).
* Building a baseline price prediction model using Linear Regression, Random Forest, and XGBoost.
* Visualizing feature importance and SHAP explanations to interpret valuation drivers.
* Deploying a Streamlit dashboard for interactive exploration (optional).