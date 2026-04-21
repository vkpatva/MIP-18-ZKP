---
license: mit
path: winning_gbc_model
---
# 🏦 Loan Default Prediction — Credit Risk EDA & Modeling

**Author:** Uri Sivan  
**Assignment:** Assignment #2 — Classification, Regression, Clustering & Evaluation  
**Dataset:** [Loan Default Dataset](https://www.kaggle.com/datasets/yasserh/loan-default-dataset) — Kaggle  
**Repository:** `Uris001/credit-risk-eda`

---

## 🎥 Presentation

> 📺 **Video link:** *(add your video link here after recording)*

---

## 📌 Project Overview

This project builds a full end-to-end machine learning pipeline to predict
loan default risk using a real-world mortgage dataset of approximately 147,000
loans. The pipeline progresses from raw data through exploratory analysis,
feature engineering, unsupervised clustering, regression modeling, and
multi-class classification — ending with a production-ready XGBoost classifier
exported for deployment.

**Research Question:**
Given loan application data available at origination time, can we accurately
predict which loans will default — and assign each loan to a meaningful risk
tier (Low, Medium, High)?

---

## 🗺️ Project Workflow

```
Raw Dataset
    ↓
EDA (cleaning, statistics, visualizations)
    ↓
Train Baseline Linear Regression Model
    ↓
Show Feature Importance (coefficients)
    ↓
Feature Engineering
(transformations, encoding, clustering, PCA, synthetic features)
    ↓
Train 3 Regression Models → Evaluate All → Upload Best to HuggingFace
    ↓
Convert Regression → Classification Problem
    ↓
Train 3 Classification Models → Upload Best to HuggingFace
    ↓
Upload Colab Notebook to HF
    ↓
Write README on HF
    ↓
Record Presentation + Add Link to README
```

---

## 📂 Repository Contents

| File | Description |
|---|---|
| `Uri_Sivan_Assignment_2.ipynb` | Full notebook — all parts |
| `best_model_xgboost.pkl` | Winning classification model (XGBoost) |
| `best_regression_model.pkl` | Winning regression model (Gradient Boosting) |
| `README.md` | This file |

---

## 📊 Dataset Description

**Source:** Kaggle — Loan Default Dataset  
**Size:** ~147,000 rows × 28 features  
**After cleaning:** 146,829 rows × 27 features  
**Target:** `Status` — binary (0 = Repaid, 1 = Defaulted)  
**Class distribution:** 75.66% repaid / 24.34% defaulted

---

## 🔍 Part 2: Exploratory Data Analysis

### 2.1 Initial Column Cleanup

- Renamed all 28 columns to readable snake_case names
- **Dropped 5 zero-variance / identifier columns:**
  - `loan_id` — unique identifier, zero predictive value
  - `year` — single value (2019), zero variance
  - `construction_type` — single value (`sb`)
  - `secured_by` — single value (`home`)
  - `security_type` — near 100% `direct`
- **Dropped `loan_purpose`** — undocumented codes (p1–p4) with no codebook
- Relabeled all categorical values from codes to readable strings
  (e.g. `cf` → `conforming`, `pre` → `yes`, `pr` → `primary_residence`)

### 2.2 Missing Value Analysis

Before imputing anything, a **missingness co-occurrence heatmap** was computed
to understand whether missing values cluster on the same rows. This revealed
that `interest_rate`, `interest_rate_spread`, `upfront_charges`, and
`debt_to_income_ratio` are missing on the same rows — corresponding to
applications that did not reach final funding.

**Missing value decisions:**

| Column | Missing | Strategy | Justification |
|---|---|---|---|
| `term_months` | 41 (0.03%) | Drop rows | Random clerical gaps, no segment signal |
| `negative_amortization` | 121 (0.08%) | Drop rows | Independent missingness pattern |
| `age_group` + `submission_channel` | 200 (0.13%) | Drop rows | Co-occurring on same 200 rows |
| `approved_in_advance` | 908 (0.6%) | Drop rows | Independent missingness |
| `loan_limit` | 3,344 (2.25%) | Mode imputation | Too many to drop; 91% conforming |
| `income` | 10,410 (7%) | 2D binning: loan decile × credit score band | Preserves income-leverage relationship |
| `property_value` + `LTV` | 15,131 | Back-derive from median LTV by loan decile | Keeps both columns mechanically consistent |
| `debt_to_income_ratio` | 24,121 (16%) | 2D binning: credit band × income decile | Uses strongest predictors, no leakage |
| `interest_rate` | 36,439 (25%) | 2D binning: credit band × LTV band | Structural missingness (non-funded apps) |
| `interest_rate_spread` | Partial | Binned on interest_rate deciles | Fallback imputation |

### 2.3 Invalid Value Conversion

- `income == 0` → converted to NaN (then imputed)
- `loan_to_value_ratio > 150` → converted to NaN (mechanically impossible)
- `interest_rate == 0` → converted to NaN (unfunded applications)
- `income < $1,000/month` → rows dropped (unrealistically low, likely errors)

### 2.4 Duplicate Removal

- Identified and removed exact duplicate rows
- Final dataset: **146,829 rows**

### 2.5 Outlier Detection and Handling

- IQR analysis performed on all numeric columns
- **Strategy: log transformation** for right-skewed monetary columns
  (`loan_amount`, `property_value`, `income`)
- `term_months` excluded from IQR analysis — effectively categorical (98% = 360 months)
- Converted `term_months` to product buckets: 30yr, 15yr, 20yr, 25yr, other

### 2.6 Feature Exclusions (Data Quality)

| Feature | Reason |
|---|---|
| `credit_score` | Pearson r = 0.003 with target; near-uniform distribution |
| `interest_rate` | Leakage — lender's post-approval pricing decision |
| `interest_rate_spread` | Derived from excluded leakage column |
| `credit_worthiness` | Leakage — lender's internal risk classification |
| `credit_bureau` | Cramér's V = 0.5929 — anomalously high, probable leakage |
| `coapplicant_credit_bureau` | Same leakage concern as credit_bureau |
| `upfront_charges` | 0% default rate in no-fee segment is a data artifact |
| `open_credit_flag` | Cramér's V < 0.01 — no detectable signal |

### 2.7 Descriptive Statistics

- Styled summary table: mean, median, std, quartiles for all numeric features
- Categorical summary table: top category, share, number of distinct values
- **Correlation heatmap** — Pearson correlations across numeric features and target
- **Feature-target correlation bar chart** — ranked by absolute correlation

### 2.8 Univariate Analysis

For each key numeric feature, two-panel plots were produced:
- Left: distribution histogram + KDE with mean/median lines
- Right: default rate by quintile/tier with confidence intervals or counts

**Features analyzed:**
- `loan_amount` — right-skewed; higher loans → lower default (inverse relationship)
- `property_value` — stronger inverse relationship than loan amount (−12.4pp spread)
- `loan_to_value_ratio` — LTV risk zones annotated; peak risk at 75–90% band
- `income` — strongest individual numeric predictor; 36.8% → 19.5% across deciles
- `debt_to_income_ratio` — non-linear; 28–43% band highest risk (selection bias at extremes)

### 2.9 Categorical Statistical Tests

Chi-square tests + Cramér's V computed for all 17 categorical features vs target:

**Top predictors by Cramér's V:**

| Feature | Cramér's V | Significant |
|---|---|---|
| `credit_bureau` | 0.5929 | Yes (excluded — leakage) |
| `lump_sum_payment_flag` | 0.1894 | Yes |
| `negative_amortization` | 0.1523 | Yes |
| `coapplicant_credit_bureau` | 0.1446 | Yes (excluded — leakage) |
| `submission_channel` | 0.1198 | Yes |
| `loan_type` | 0.0885 | Yes |
| `business_or_commercial` | 0.0871 | Yes |
| `open_credit_flag` | 0.0096 | Yes (excluded — no signal) |

### 2.10 Bivariate and Interaction Analysis

**LTV × DTI Compound Risk Heatmap**

![LTV DTI Heatmap](ltv_dti_heatmap.png)

- LTV Q3 (75–90%) combined with DTI Q2–Q4 reaches 68.9% default — 3× dataset average
- Peak risk is not at maximum LTV but at mid-band leverage + mid-band debt burden
- LTV Q5 flattens due to mortgage insurance selection effect

**Monthly Income vs Loan Amount**
- Defaulters earn ~32% less but borrow only ~10% less → LTI ratio is the key driver
- No clean linear decision boundary → motivates tree-based models

**Default Rate by Decile: Income vs Loan Amount vs LTI**
- Income: strong monotonic decline (36.8% → 19.5%)
- LTI: near-flat for deciles 0–6, spikes to 35.2% at decile 9 → tail-risk flag

**Age Group × Region Heatmap**
- North-East under-25: 50.0% default
- North-East over-74: 44.7% default
- North region: lowest and most stable across all age groups

**Default Rate by Age Group and Gender**
- Joint applicants: lowest default (17.5%–24.4%) across every age group
- Male: steepest age-related increase (30.6% at <25 → 34.5% at >74)

### 2.11 Term Category Analysis

- 25-year loans: **56.4% default** — more than double any other term
- 30yr (82%) and 15yr (9%) statistically indistinguishable at ~24%

---

## ⚙️ Part 4: Feature Engineering

### 4.1 New Features Created

| Feature | Type | EDA Motivation |
|---|---|---|
| `lti_ratio_log` | Continuous | Q2: risk runs along LTI diagonal |
| `loan_to_property` | Continuous | Alternative leverage measure |
| `monthly_debt_est` | Continuous | Absolute monthly debt burden |
| `is_extreme_lti` | Binary | Q3: top decile spikes to 35.2% default |
| `is_compound_risk` | Binary | Q1: LTV mid-band AND DTI mid-band → 68.9% |
| `is_25yr_term` | Binary | 56.4% default rate |
| `is_northeast_under25` | Binary | Q4: 50.0% default cell |
| `is_northeast_over74` | Binary | Q4: 44.7% default cell |
| `is_joint_prime_age` | Binary | Q5: lowest default rate segment |
| `is_exotic_product` | Binary | Consolidates neg_amort + interest_only + lump_sum |

### 4.2 Scikit-Learn Pipeline

```
ColumnTransformer
├── StandardScaler     → 8 numeric features
├── passthrough        → 7 binary flags
└── OneHotEncoder      → 14 categorical features → 30 columns
```

All transformers fit on training set only. Zero test set leakage.

### 4.3 PCA

- 9 numeric features → **5 principal components** (98.6% variance explained)
- PC1 (34.7%): wealth — loan amount, property value, income
- PC2 (24.7%): affordability stress — LTI, loan-to-property, monthly debt
- PC3 (19.1%): leverage — LTV, DTI

### 4.4 K-Means Clustering (K=4)

![Cluster Profiles](cluster_profiles.png)

| Cluster | N | Default Rate | Label |
|---|---|---|---|
| 2 | 19,498 | 13.8% | Conservative borrowers |
| 3 | 18,959 | 19.5% | Stable mid-tier |
| 0 | 44,039 | 25.2% | Standard borrowers |
| 1 | 34,967 | 31.8% | Stressed borrowers |

18-point spread confirms financially meaningful segmentation.
Clusters validated with both PCA and t-SNE projections.

**Cluster features added:**
- `cluster_id` (one-hot) — discrete segment membership
- `cluster_dist` — distance to centroid (atypicality signal)
- `cluster_default_rate` — **excluded** (target encoding = leakage)

### 4.5 Final Feature Matrix

| Category | Count |
|---|---|
| Numeric (scaled) | 8 |
| Binary flags | 7 |
| One-hot encoded | 30 |
| PCA components | 5 |
| Cluster features | 4 |
| **Total** | **54** |

---

## 📈 Part 3 & 5: Regression Models

### Part 3 — Baseline Linear Regression (34 raw features)

| Metric | Train | Test |
|---|---|---|
| MAE | 0.3223 | 0.3227 |
| RMSE | 0.3939 | 0.3944 |
| R² | 0.1575 | 0.1555 |
| ROC-AUC | — | 0.693 |

**Top coefficient features:**
- `lump_sum_payment_flag_yes` (+0.5251) — strongest risk-increasing
- `negative_amortization_yes` (+0.1836)
- `term_category_25yr` (+0.1784)
- `occupancy_type_primary_residence` (−0.1125) — strongest protective

### Part 5 — Three Improved Models (54 engineered features)

![ROC Curves](roc_curves.png)

| Model | ROC-AUC | F1 (Default) | Accuracy | R² |
|---|---|---|---|---|
| Linear Regression (Baseline) | 0.693 | 0.244 | 77.8% | 0.109 |
| Linear Regression (Engineered) | 0.809 | 0.519 | 80.6% | 0.254 |
| Logistic Regression | 0.812 | 0.589 | 76.5% | — |
| **Gradient Boosting (Winner)** | **0.882** | **0.726** | **88.7%** | — |

**Key finding:** Feature engineering alone improved R² from 0.109 → 0.254
and AUC from 0.693 → 0.809. The largest single improvement in the pipeline.

**Confusion matrix comparison:**

| Model | FNR (Defaults missed) | FPR (False alarms) |
|---|---|---|
| Linear Reg (Engineered) | 57.1% | 7.2% |
| Logistic Regression | 30.8% | 21.2% |
| **Gradient Boosting** | **38.5%** | **2.6%** |

**Winner: Gradient Boosting Classifier** — `best_regression_model.pkl`

---

## 🏷️ Part 7: Regression → Classification

**Strategy: Business Rule Threshold (3-Class)**

| Class | Label | Threshold | Rationale |
|---|---|---|---|
| 0 | Low Risk | score < 0.20 | Below dataset default rate with margin |
| 1 | Medium Risk | 0.20 ≤ score < 0.40 | Spans the dataset average |
| 2 | High Risk | score ≥ 0.40 | Materially above dataset average |

**Why recall > precision:**
Missing a high-risk loan = realized financial loss.
Flagging a safe loan = opportunity cost only.
False negatives cost 5–10× more than false positives in lending.

---

## 🧠 Part 8: Classification Models

### Models Trained

| Model | Architecture | Key Parameters |
|---|---|---|
| Random Forest | Ensemble of independent trees | 300 trees, max_depth=12, balanced |
| XGBoost | Sequential gradient boosted trees | 300 rounds, lr=0.05, max_depth=5 |
| K-Nearest Neighbors | Distance-based instance learner | K=15, distance weights |

### Feature Importance

![Feature Importance](feature_importance.png)

**Top XGBoost features:**
1. `pca_1` — wealth composite
2. `is_25yr_term` — 56.4% default flag
3. `cluster_dist` — atypicality within segment
4. `approved_in_advance_yes`
5. `business_or_commercial_yes`

**Winner: XGBoost** — `best_model_xgboost.pkl`

---

## 🧹 Data Cleaning Summary

![Cleaning Summary](cleaning_summary.png)

| Step | Rows After | Action |
|---|---|---|
| Raw dataset | 148,670 | Initial load |
| Drop zero-variance columns | 148,670 | 5 columns removed |
| Drop loan_purpose | 148,670 | Undocumented codes |
| Drop term_months nulls | 148,629 | 41 rows |
| Drop negative_amortization nulls | 148,508 | 121 rows |
| Drop age_group / submission nulls | 148,308 | 200 rows |
| Drop approved_in_advance nulls | 147,400 | 908 rows |
| Impute loan_limit | 147,400 | Mode fill |
| Impute income | 147,400 | 2D binning |
| Drop income < $1,000 | ~147,300 | Data errors |
| Impute property_value / LTV | ~147,300 | Back-derivation |
| Drop LTV > 150 | 147,267 | 33 artifacts |
| Impute DTI | 147,267 | 2D binning |
| Impute interest_rate | 147,267 | 2D binning |
| Remove duplicates | **146,829** | ~438 rows |

**Total rows dropped:** 1,841 (1.2%) — **Retained: 146,829 (98.8%)**

---

## 🚀 How to Load and Use the Models

```python
import pickle
import numpy as np

# Load classification model
with open("best_model_xgboost.pkl", "rb") as f:
    clf_model = pickle.load(f)

# Load regression model
with open("best_regression_model.pkl", "rb") as f:
    reg_model = pickle.load(f)

# Both expect the 54-feature engineered matrix from Part 4
y_class  = clf_model.predict(X_new)
y_score  = np.clip(reg_model.predict(X_new), 0, 1)

class_map = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
risk_labels = [class_map[c] for c in y_class]
```

---

## 📦 Requirements

```
pandas>=1.3
numpy>=1.21
scikit-learn>=1.0
xgboost>=1.5
matplotlib>=3.4
seaborn>=0.11
plotly>=5.0
scipy>=1.7
statsmodels>=0.13
```

---

## 📝 Key Design Decisions

| Decision | Rationale |
|---|---|
| Exclude `upfront_charges` | 0% default in no-fee segment is a data artifact |
| Exclude `credit_score` | Near-zero linear correlation, near-uniform distribution |
| Exclude `interest_rate` | Post-approval pricing decision — leakage |
| Exclude `credit_bureau` | Cramér's V = 0.59 — probable structural leakage |
| Remove `cluster_default_rate` | Target encoding = indirect leakage |
| Stratified train/test split | Preserves 24.34% class rate in both sets |
| Fit transformers on train only | Prevents test set influencing training |
| K=4 for clustering | Elbow flattens at K=4; 4 financially interpretable segments |
| PCA on numeric block only | Categoricals already orthogonal after one-hot encoding |

---

## 📋 Assignment Structure

| Part | Description | Key Output |
|---|---|---|
| Part 1 | Dataset selection and description | Research question |
| Part 2 | EDA — cleaning, statistics, visualizations | 11 subsections, 20+ plots |
| Part 3 | Baseline Linear Regression | R²=0.1555, AUC=0.693 |
| Part 4 | Feature engineering — transformations, PCA, clustering | 54-feature matrix |
| Part 5 | Three improved regression models | GBC winner AUC=0.882 |
| Part 7 | Regression-to-Classification | 3 risk classes |
| Part 8 | Three classification models | XGBoost winner |

---

*Assignment #2 — Data Science Program | April 2026*
