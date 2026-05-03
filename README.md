---
license: mit
language:
  - en
tags:
  - loan-default
  - credit-risk
  - classification
  - regression
  - xgboost
  - gradient-boosting
  - tabular
  - finance
  - supervised-learning
  - feature-engineering
datasets:
  - yasserh/loan-default-dataset
metrics:
  - f1
  - roc_auc
  - accuracy
library_name: sklearn
pipeline_tag: tabular-classification
---

# 🏦 Loan Default Prediction — Credit Risk EDA & Modeling

**Author:** Uri Sivan  
**Assignment:** Assignment #2 — Classification, Regression, Clustering & Evaluation  
**Dataset:** [Loan Default Dataset](https://www.kaggle.com/datasets/yasserh/loan-default-dataset) — Kaggle  
**Repository:** `Uris001/loan-default-risk-predictor`

---

## 🎥 Presentation

<video src="https://huggingface.co/Uris001/loan-default-risk-predictor/resolve/main/presentation.mp4" controls="controls" style="max-width: 720px;"></video>

---

## 📌 Project Overview

This project builds a full end-to-end machine learning pipeline to predict loan default risk
using a real-world mortgage dataset of approximately 148,000 loans. The pipeline progresses
from raw data through exploratory analysis, feature engineering, unsupervised clustering,
regression modeling, and multi-class classification — ending with two production-ready models
exported for deployment.

**Research Question:**
> Given loan application data available at origination time, can we accurately predict which
> loans will default — and assign each loan to a meaningful risk tier (Low / Medium / High)?

**Why this matters:**
In mortgage lending, a single missed default costs the lender the full outstanding principal
plus legal, servicing, and provisioning costs. A model that correctly flags high-risk loans
at origination time can prevent billions in portfolio losses — but only if it is built without
data leakage and is grounded in real financial logic.

---

## 🗺️ Full Project Workflow

```
Raw Dataset (148,670 rows × 28 features)
    ↓
Part 2: EDA
  ├── Column cleanup and renaming
  ├── Missingness co-occurrence analysis (heatmap before imputation)
  ├── Domain-grounded imputation — 8 columns, 4 different strategies
  ├── Invalid value detection and removal
  ├── Outlier detection — IQR analysis + log transforms on 4 monetary cols
  ├── Duplicate removal
  ├── Descriptive statistics + correlation heatmap
  ├── Univariate analysis — 7 numeric + 12 categorical features
  ├── Chi-square + Cramér's V — all 17 categorical features
  └── 5 bivariate research questions with dedicated visualizations
    ↓
Part 3: Baseline Linear Regression
  ├── 34 model features (after one-hot encoding of 27 cleaned columns)
  ├── 80/20 stratified split (SEED=42)
  ├── MAE=0.3227, RMSE=0.3944, R²=0.1555, AUC=0.693
  └── Feature importance via coefficients
    ↓
Part 4: Feature Engineering
  ├── 10 new features (3 ratio + 7 binary interaction flags)
  ├── ColumnTransformer pipeline (StandardScaler + OneHotEncoder + passthrough)
  ├── PCA: 9 numeric → 5 orthogonal components (98.6% variance)
  ├── K-Means K=4 (elbow method) + t-SNE + PCA visualization
  └── Final: 54-feature matrix, zero leakage
    ↓
Part 5: Three Improved Regression Models
  ├── Linear Regression (engineered) — AUC 0.809 (+16.7% over baseline)
  ├── Logistic Regression — AUC 0.812
  ├── Gradient Boosting — AUC 0.882 ← WINNER
  ├── ROC + Precision-Recall curves, confusion matrices, feature importance
  └── best_regression_model.pkl → HuggingFace
    ↓
Part 7: Regression → Classification
  ├── Business rule thresholds: 0.20 / 0.40
  ├── 3 classes: Low Risk (9.4% DR) / Medium Risk (26.7% DR) / High Risk (61.8% DR)
  └── 52.4pp spread validates threshold quality
    ↓
Part 8: Three Classification Models
  ├── Random Forest (300 trees, class_weight=balanced)
  ├── XGBoost (tuned via RandomizedSearchCV) ← WINNER
  ├── K-Nearest Neighbors (K=15, distance weights)
  ├── Classification reports + confusion matrices + threshold analysis
  └── best_model_xgboost.pkl → HuggingFace
    ↓
Upload notebook + README + models → HuggingFace
Record presentation → Add link to README
```

---

## 📂 Repository Contents

| File | Description |
|---|---|
| `Uri_Sivan_Assignment_2.ipynb` | Full notebook — all parts with outputs |
| `best_model_xgboost.pkl` | Winning classification model (XGBoost tuned) |
| `best_regression_model.pkl` | Winning regression model (Gradient Boosting) |
| `README.md` | This file |
| `plots/cleaning_summary.png` | Data cleaning waterfall chart |
| `plots/ltv_dti_heatmap.png` | Q1 — LTV × DTI compound risk |
| `plots/income_loan_scatter.png` | Q2 — Income vs loan amount scatter |
| `plots/default_by_decile.png` | Q2 — Default rate by decile |
| `plots/age_region_heatmap.png` | Q3 — Age × region interaction |
| `plots/credit_bureau_vs_loan_type.png` | Q4 — Credit bureau leakage proof |
| `plots/age_gender_default.png` | Q5 — Age × gender default rates |
| `plots/cluster_profiles.png` | K-Means cluster default rates |
| `plots/roc_curves.png` | ROC + PR curves — all regression models |
| `plots/confusion_matrices_part5.png` | Confusion matrices — Part 5 |
| `plots/feature_imprtance_part_5.png` | Feature importance — Part 5 models |
| `plots/feature_engineering_impact.png` | Before/after engineering comparison |
| `plots/three_models_evaluation.png` | Classification model comparison |
| `plots/confusion_matrices_part8.png` | Confusion matrices — Part 8 |
| `plots/feature_importance_part_8.png` | Feature importance — Part 8 models |
| `plots/threshold_analysis.png` | Threshold analysis — XGBoost |

---

## 📊 Dataset Description

| Property | Value |
|---|---|
| Source | Kaggle — Loan Default Dataset |
| Raw size | 148,670 rows × 28 features |
| After cleaning | 146,829 rows × 27 features |
| Baseline model input | 34 features (post one-hot encoding) |
| Engineered model input | 54 features |
| Target | `Status` — binary (0 = Repaid, 1 = Defaulted) |
| Class distribution | 75.66% repaid / 24.34% defaulted |
| Geography | US mortgage market |
| Time period | 2019 |

---

## 📋 Raw Feature Dictionary — All 28 Original Columns

The raw dataset contains 28 columns across four categories.
Eight were excluded before modeling (see Section 2.4 for justification).

### Loan Characteristics

| Feature | Type | Description | Kept |
|---|---|---|---|
| `loan_amount` | Numeric | Total loan disbursed in USD | ✅ (log-transformed) |
| `loan_to_value_ratio` | Numeric | Loan amount ÷ property value × 100 | ✅ |
| `term_months` | Numeric | Loan duration in months (360=30yr, 180=15yr, etc.) | ✅ (bucketed) |
| `loan_type` | Categorical | Type 1 / Type 2 / Type 3 — mortgage product category | ✅ |
| `loan_limit` | Categorical | Conforming / Non-conforming (agency limit compliance) | ✅ |
| `loan_purpose` | Categorical | p1 / p2 / p3 / p4 — undocumented codes | ❌ (no codebook) |
| `negative_amortization` | Binary | Yes/No — balance can grow over time | ✅ |
| `interest_only_flag` | Binary | Yes/No — interest-only payment period | ✅ |
| `lump_sum_payment_flag` | Binary | Yes/No — large irregular payment option | ✅ |
| `business_or_commercial` | Binary | Yes/No — loan for business purpose | ✅ |

### Property Characteristics

| Feature | Type | Description | Kept |
|---|---|---|---|
| `property_value` | Numeric | Appraised value of collateral property in USD | ✅ (log-transformed) |
| `occupancy_type` | Categorical | Primary residence / Investment / Secondary home | ✅ |
| `total_units` | Categorical | 1U / 2U / 3U / 4U — number of units in property | ✅ |
| `construction_type` | Categorical | `sb` (site-built) — 100% single value | ❌ (zero variance) |
| `secured_by` | Categorical | `home` — 99.9% single value | ❌ (zero variance) |
| `security_type` | Categorical | `direct` — 99.9% single value | ❌ (zero variance) |

### Borrower Characteristics

| Feature | Type | Description | Kept |
|---|---|---|---|
| `income` | Numeric | Monthly gross income of primary applicant in USD | ✅ (log-transformed) |
| `debt_to_income_ratio` | Numeric | Total monthly debt payments ÷ gross monthly income | ✅ |
| `credit_score` | Numeric | Credit score 500–900 — creditworthiness measure | ❌ (r=0.003 with target) |
| `age_group` | Categorical | <25 / 25-34 / 35-44 / 45-54 / 55-64 / 65-74 / >74 | ✅ |
| `gender` | Categorical | Male / Female / Joint / Sex Not Available | ✅ |
| `credit_bureau` | Categorical | CIB / CRIF / EQUI / EXP — bureau used for credit check | ❌ (leakage — EQUI=100% default) |
| `coapplicant_credit_bureau` | Categorical | Same as credit_bureau for co-applicant | ❌ (same leakage) |
| `credit_worthiness` | Categorical | l1 / l2 — lender's internal risk classification | ❌ (leakage — set post-underwriting) |

### Loan Pricing and Process

| Feature | Type | Description | Kept |
|---|---|---|---|
| `interest_rate` | Numeric | Loan interest rate in % — set by lender post-approval | ❌ (leakage) |
| `interest_rate_spread` | Numeric | Rate minus benchmark rate — derived from interest_rate | ❌ (inherits leakage) |
| `upfront_charges` | Numeric | Origination fee charged at closing in USD | ❌ (data artifact — 0% default in no-fee segment) |
| `approved_in_advance` | Binary | Yes/No — pre-approval before property selection | ✅ |
| `submission_channel` | Categorical | Retail / Broker / Direct — how application was submitted | ✅ |
| `region` | Categorical | North / North-East / Central / South | ✅ |
| `open_credit_flag` | Binary | Yes/No — open credit line exists | ❌ (Cramér's V < 0.01) |

### Identifiers (always excluded)

| Feature | Type | Description |
|---|---|---|
| `loan_id` | ID | Unique loan identifier — no predictive value |
| `year` | Numeric | Single value (2019) — zero variance |

---

### Feature Count Summary

| Stage | Features | Notes |
|---|---|---|
| Raw dataset | 28 | All original columns including IDs |
| After dropping IDs + zero-variance | 21 | loan_id, year, construction_type, secured_by, security_type, loan_purpose removed |
| After excluding leakage + no-signal | 13 raw columns | credit_score, interest_rate, interest_rate_spread, credit_worthiness, credit_bureau, coapplicant_credit_bureau, upfront_charges, open_credit_flag removed |
| After one-hot encoding (baseline) | **34 model features** | Categorical expansion for Part 3 |
| After feature engineering | **54 model features** | 10 new features + PCA + cluster features |

---

## 🔍 Part 2: Exploratory Data Analysis

### 2.1 Initial Column Audit and Cleanup

Before any analysis, every column was audited for informativeness:

- **Renamed** all 28 columns to readable `snake_case` names
- **Dropped 5 zero-variance / identifier columns** — these carry zero predictive value:
  `loan_id` (unique ID), `year` (single value: 2019), `construction_type` (99.9% `sb`),
  `secured_by` (99.9% `home`), `security_type` (99.9% `direct`)
- **Dropped `loan_purpose`** — codes p1–p4 with no codebook available. Including
  undocumented codes as features would embed unknown biases into the model.
- **Relabeled** all categorical codes to readable strings (`cf` → `conforming`,
  `pr` → `primary_residence`, `pre` → `yes`, etc.)

---

### 2.2 Missingness Analysis — Co-occurrence Heatmap First

**Before imputing a single value**, a missingness co-occurrence heatmap was computed
across all columns. This revealed that `interest_rate`, `interest_rate_spread`,
`upfront_charges`, and `debt_to_income_ratio` are missing on the **same rows** —
corresponding to applications that did not reach final funding. This is structural
missingness, not random. Understanding this pattern drove the imputation strategy.

| Column | Missing N | % | Strategy | Justification |
|---|---|---|---|---|
| `term_months` | 41 | 0.03% | Drop rows | Random clerical gaps |
| `negative_amortization` | 121 | 0.08% | Drop rows | Independent missingness |
| `age_group` + `submission_channel` | 200 | 0.13% | Drop rows | Co-occurring on same 200 rows |
| `approved_in_advance` | 908 | 0.61% | Drop rows | Independent missingness |
| `loan_limit` | 3,344 | 2.25% | Mode imputation | 91% conforming — safe to fill |
| `income` | 10,410 | 7.0% | **2D binning**: loan decile × credit score band | Preserves income-leverage relationship |
| `property_value` + `LTV` | 15,131 | 10.2% | **Back-derivation** from median LTV by loan decile | Keeps both columns mechanically consistent |
| `debt_to_income_ratio` | 24,121 | 16.2% | **2D binning**: credit band × income decile | Uses strongest predictors, no leakage |
| `interest_rate` | 36,439 | 24.5% | **2D binning**: credit band × LTV band | Structural missingness on non-funded applications |

---

### 2.3 Invalid Values and Outlier Treatment

**Invalid values converted to NaN (then imputed):**
- `income == 0` — mechanically impossible for a funded mortgage
- `loan_to_value_ratio > 150` — division artifact, not a real loan
- `interest_rate == 0` — unfunded applications

**Rows dropped:**
- `income < $1,000/month` — 538 rows. Below $1,000 cannot sustain any mortgage payment.
- `LTV > 150` — 33 rows after NaN conversion.

**Outlier detection — IQR analysis on all numeric columns:**

| Column | Skewness (raw) | Treatment | Skewness (after) |
|---|---|---|---|
| `loan_amount` | 1.8 | `log1p` transform | 0.12 |
| `property_value` | 4.6 | `log1p` transform | −0.04 |
| `income` | 18.0 | `log1p` transform | 0.16 |
| `upfront_charges` | 2.1 | `log1p` transform | 0.09 |
| `term_months` | — | Bucketed: 30yr (82%), 15yr (9%), 20yr (4%), 25yr (2%), other (4%) | — |
| `loan_to_value_ratio` | 0.3 | Retained — near-normal | — |
| `debt_to_income_ratio` | 0.8 | Retained — acceptable | — |

Log transforms reduced skewness by 90%+ across all four monetary columns.

---

### 2.4 Feature Exclusions — Eight Columns Removed With Evidence

| Feature | Evidence | Reason |
|---|---|---|
| `credit_score` | Pearson r = 0.003; near-uniform 500–900 distribution | Pre-publication filtering removed the predictive range |
| `interest_rate` | Set by lender post-approval | Post-origination leakage — encodes the outcome |
| `interest_rate_spread` | Mechanically derived from `interest_rate` | Inherits leakage from parent column |
| `credit_worthiness` | Lender's internal risk tier (l1/l2) | Set after underwriting — leakage |
| `credit_bureau` | Cramér's V = 0.5929; EQUI = 100% default across ALL loan types | Q4 proves this is post-default label assignment |
| `coapplicant_credit_bureau` | Same mechanism as `credit_bureau` | Same leakage concern |
| `upfront_charges` | No-fee segment: exactly 0.0% default (N=20,582) | Structurally impossible — data artifact |
| `open_credit_flag` | Cramér's V = 0.0096 | Below noise threshold |

---

### 2.5 Data Cleaning Summary

![Cleaning Summary](plots/cleaning_summary.png)

| Step | Rows After | Change |
|---|---|---|
| Raw dataset | 148,670 | — |
| Drop zero-variance + loan_purpose | 148,670 | Columns only |
| Drop term_months nulls | 148,629 | −41 |
| Drop negative_amortization nulls | 148,508 | −121 |
| Drop age_group / submission nulls | 148,308 | −200 |
| Drop approved_in_advance nulls | 147,400 | −908 |
| Impute loan_limit (mode) | 147,400 | 0 rows |
| Impute income (2D binning) | 147,400 | 0 rows |
| Drop income < $1,000 | 146,862 | −538 |
| Impute property_value / LTV (back-derive) | 146,862 | 0 rows |
| Drop LTV > 150 | 146,829 | −33 |
| Impute DTI + interest_rate (2D binning) | 146,829 | 0 rows |
| **Final clean dataset** | **146,829** | **−1,841 total (1.2%)** |

---

### 2.6 Descriptive Statistics and Correlation Analysis

Key findings from the correlation analysis:

- `loan_amount_log` ↔ `property_value_log`: r = 0.85 — strongest multicollinearity pair
- `loan_amount_log` ↔ `income_log`: r = 0.66 — second strongest
- `loan_to_value_ratio` ↔ `Status`: r = +0.12 — strongest raw numeric predictor
- `income_log` ↔ `Status`: r = −0.18 — strongest protective numeric predictor
- `credit_score` ↔ `Status`: r = +0.003 — confirms exclusion decision

---

### 2.7 Univariate Analysis — Key Findings

**Numeric features — default rate by quintile:**

| Feature | Bottom Quintile DR | Top Quintile DR | Direction |
|---|---|---|---|
| `loan_amount` | 29.8% | 22.4% | Inverse (larger = safer) |
| `property_value` | 31.5% | 19.1% | Strong inverse |
| `income` | 36.8% | 19.5% | Strongest inverse |
| `loan_to_value_ratio` | 13.6% (<60%) | 22.5% (90%+) | Non-linear peak at 75–90% |
| `debt_to_income_ratio` | lower | 28–43% band = peak | Non-linear |

**Categorical features — Cramér's V ranking (all 17 features tested):**

| Feature | Cramér's V | Status |
|---|---|---|
| `credit_bureau` | 0.5929 | **Excluded — leakage** |
| `lump_sum_payment_flag` | 0.1894 | Retained |
| `negative_amortization` | 0.1523 | Retained |
| `coapplicant_credit_bureau` | 0.1446 | **Excluded — leakage** |
| `submission_channel` | 0.1198 | Retained |
| `loan_type` | 0.0885 | Retained |
| `open_credit_flag` | 0.0096 | **Excluded — no signal** |

---

### 2.8 Five Bivariate Research Questions

---

#### Q1 — Does leverage (LTV) × debt burden (DTI) create compound risk?

> *"Is the combination of high LTV and high DTI more dangerous than either alone?"*

![LTV DTI Heatmap](plots/ltv_dti_heatmap.png)

**Finding:**
The peak default rate (68.9%) occurs at LTV Q3 (75–90%) × DTI Q2–Q4 — not at the maximum
values of either variable. LTV Q5 (highest leverage) defaults less than Q3 because very high
LTV loans required mortgage insurance and stricter underwriting that pre-screened the worst
borrowers. The compound risk interaction is non-linear and invisible to any model that treats
LTV and DTI as independent additive predictors.

**Modeling implication:**
Created `is_compound_risk` — a binary flag for the 75–90% LTV AND DTI mid-band zone.
This single cell reaches 3× the dataset average default rate.

---

#### Q2 — Does loan-to-income ratio outperform absolute income or loan amount?

> *"Is affordability stress — not income or loan size — the real driver?"*

![Income VS Loan amount](plots/income_loan_scatter.png)
![Default Rate by Decile](plots/default_by_decile.png)

**Finding:**
Defaulters earn ~32% less but borrow only ~10% less than repaid borrowers. The risk gradient
runs diagonally along the loan-to-income ratio, not along either axis independently. The
decile plot confirms: income has the strongest and most consistent monotonic gradient
(36.8% → 19.5%), loan amount is weaker and shallower, and LTI is near-flat for deciles 0–6
but spikes to 35.2% at decile 9 — a tail-risk feature.

**Modeling implication:**
Engineered `lti_ratio_log`. Created `is_extreme_lti` binary flag for the top LTI decile only.
Raw LTI as continuous predictor discarded — its signal concentrates entirely in the tail.

---

#### Q3 — Do age and geography interact to create localized hotspots?

> *"Are young or elderly borrowers in specific regions disproportionately risky?"*

![Age Region Heatmap](plots/age_region_heatmap.png)

**Finding:**
North-East region contains two structural extreme cells: under-25 at 50.0% default and
over-74 at 44.7% default — both more than double the dataset average. The North region
is consistently the safest across all age groups (19.7%–28.1%). The individual Cramér's V
for age (0.049) and region (0.048) are modest, but their interaction creates cells with
2× the dataset average — signal invisible to any model using only main effects.

**Modeling implication:**
Created `is_northeast_under25` and `is_northeast_over74` binary flags.
Used North as the reference (lowest-risk) category in one-hot encoding.

---

#### Q4 — Which credit bureau × loan type combinations are most dangerous?

> *"Do specific credit bureau and loan type combinations reveal leakage?"*

![Credit Bureau Loan Type Heatmap](plots/credit_bureau_vs_loan_type.png)

**Finding:**
Three bureaus (CIB, CRIF, EXP) show realistic moderate default rates (13%–26%) across all
loan types. The EQUI bureau shows **100.0% default rate across every single loan type without
exception**. A perfect 100% default rate uniform across all product types and borrower profiles
cannot be a risk signal — it is forensic evidence of post-default label assignment. This
explains the anomalous Cramér's V of 0.5929 for `credit_bureau` — it was not signal, it
was leakage.

**Modeling implication:**
`credit_bureau` and `coapplicant_credit_bureau` excluded from all models.
`loan_type` retained — the 12-point spread (13% vs 25%) is real product-driven variation.

---

#### Q5 — Does gender and applicant type affect default risk across the life cycle?

> *"Do joint applicants systematically outperform individual borrowers at every age?"*

![Age Gender Default](plots/age_gender_default.png)

**Finding:**
Joint applicants have the lowest default rate at every single age group (17.5%–24.4%).
Male applicants show the steepest age-related increase (30.6% at <25 → 34.5% at >74).
All four groups follow a U-shaped age pattern with the trough at 35–44 — peak earning years.
The joint-male gap widens with age: ~7 points at <25 → ~10 points at >74.

**Modeling implication:**
Created `is_joint_prime_age` flag for joint applicants aged 35–54 — the safest identifiable
demographic segment.

---

## 📉 Part 3: Baseline Linear Regression

**Goal:** Establish a reproducible, leakage-free performance floor before any feature
engineering. Every subsequent model must beat this benchmark.

**Feature count:** After cleaning and one-hot encoding, the 27 remaining raw columns
expand to **34 model features** (categorical columns encode into multiple binary columns).

**Design decisions:**
- **34 model features**: log-transformed monetary + bounded numeric + one-hot categoricals
- **80/20 stratified split**: preserves the 24.34% default rate in both sets
- **`random_state=42`**: all results are fully reproducible
- **`StandardScaler` fit on train only**: zero test set leakage
- **`LinearRegression()` with default parameters**: no regularization, no tuning

**Results:**

| Metric | Train | Test | Gap |
|---|---|---|---|
| MAE | 0.3223 | 0.3227 | 0.0004 |
| MSE | 0.1552 | 0.1555 | 0.0003 |
| RMSE | 0.3939 | 0.3944 | 0.0005 |
| R² | 0.1575 | 0.1555 | 0.0020 |
| ROC-AUC | — | 0.693 | — |
| F1 (Default) | — | 0.244 | — |
| Accuracy | — | 77.8% | — |
| FNR | — | 57.1% | — |

**Key observations:**
- **No overfitting** — train/test gap < 0.002 across all metrics
- **R² = 0.1555** — explains 15.6% of default variance. Real signal exists, 84% unexplained
- **FNR = 57.1%** — misses more than half of actual defaults. Not deployable.
- **Score distributions overlap heavily** — both classes peak at ~0.25

**Top coefficient features:**

| Feature | Coefficient | Direction |
|---|---|---|
| `lump_sum_payment_flag_yes` | +0.5251 | Risk-increasing |
| `negative_amortization_yes` | +0.1836 | Risk-increasing |
| `term_category_25yr` | +0.1784 | Risk-increasing |
| `loan_limit_non_conforming` | +0.1027 | Risk-increasing |
| `occupancy_type_primary_residence` | −0.1125 | Protective |
| `property_value_log` | −0.08 | Protective |
| `income_log` | −0.07 | Protective |

**Key finding:** Loan product type features dominate — not borrower financial metrics.
The type of mortgage selected predicts default more strongly than income, LTV, or DTI.
This directly shaped Part 4 feature engineering.

---

## ⚙️ Part 4: Feature Engineering

Feature engineering was the single most impactful step in the entire pipeline —
more impactful than any model choice. Every feature below is directly traceable
to a specific EDA finding.

### 4.1 Ten New Features

| Feature | Type | EDA Source | Default Rate Signal |
|---|---|---|---|
| `lti_ratio_log` | Continuous | Q2: risk runs along LTI diagonal | Tail spikes to 35.2% at decile 9 |
| `loan_to_property` | Continuous | Alternative leverage, independent of LTV imputation | Complements LTV |
| `monthly_debt_est` | Continuous | DTI × income / 100 — absolute debt burden | Magnitude, not just ratio |
| `is_extreme_lti` | Binary flag | Q2: top decile spike | 35.2% vs 23% baseline |
| `is_compound_risk` | Binary flag | Q1: LTV 75–90% AND DTI mid-band | Up to 68.9% default |
| `is_25yr_term` | Binary flag | Term analysis: 56.4% default | 2× any other term category |
| `is_northeast_under25` | Binary flag | Q3: North-East × under-25 | 50.0% default |
| `is_northeast_over74` | Binary flag | Q3: North-East × over-74 | 44.7% default |
| `is_joint_prime_age` | Binary flag | Q5: joint applicants aged 35–54 | 17.5–19.7% default |
| `is_exotic_product` | Binary flag | Baseline top-3 coefficients | Consolidates neg_amort + interest_only + lump_sum |

### 4.2 Scikit-Learn ColumnTransformer Pipeline

All transformations fit on train only, applied to test. Zero leakage.

```
ColumnTransformer
├── StandardScaler      → 8 numeric features (mean=0, std=1)
├── OneHotEncoder       → 14 categorical features (drop_first=True) → 30 columns
└── passthrough         → 7 binary flags (already 0/1, no scaling needed)
```

### 4.3 PCA — Compressing Correlated Numeric Features

`loan_amount_log`, `property_value_log`, and `income_log` correlate at
0.66–0.85 — severe enough to inflate coefficient variance in linear models.
PCA compresses the 9 numeric features into 5 orthogonal components that
carry 98.6% of the original variance while eliminating multicollinearity.

| Component | Variance | Cumulative | What it captures |
|---|---|---|---|
| PC1 | 34.7% | 34.7% | Wealth — loan amount, property value, income move together |
| PC2 | 24.7% | 59.4% | Affordability stress — LTI, loan-to-property, monthly debt |
| PC3 | 19.1% | 78.5% | Leverage — LTV and DTI capture collateral and debt burden |
| PC4 | 10.1% | 88.6% | Residual orthogonal variation |
| PC5 | 9.4% | 98.0% | Residual orthogonal variation |

---

### 4.4 K-Means Clustering — Borrower Segmentation

K=4 selected via elbow method — the rate of inertia reduction flattens
most noticeably between K=4 and K=5, and four clusters produce four
financially interpretable borrower segments.

Clusters were validated with two dimensionality reduction methods: PCA
projects the global variance structure and confirms the segments occupy
different regions; t-SNE reveals local neighborhood coherence and confirms
the clusters are not arbitrary partitions.

![Cluster Profiles](plots/cluster_profiles.png)

| Cluster | N (Train) | Default Rate | Mean Dist | Financial Profile |
|---|---|---|---|---|
| 2 | 19,498 | **13.8%** | 2.244 | Low LTV, high income, low LTI — conservative borrowers with strong repayment capacity |
| 3 | 18,959 | 19.5% | 2.390 | Below-average risk achieved through diverse financial profiles — the most internally varied cluster |
| 0 | 44,039 | 25.2% | 1.539 | Typical mortgage borrower — standard product, moderate leverage, closest to the portfolio average |
| 1 | 34,967 | **31.8%** | 1.896 | High LTI and LTV combined with exotic product flags — the primary target for risk intervention |

The 18-point spread (13.8% → 31.8%) confirms the segmentation captures
real financial structure, not statistical noise. Cluster 1 represents 30%
of the training portfolio at 31.8% default.

**Cluster features added to the model:**
- `cluster_id` (one-hot, 3 columns) — discrete segment membership; which
  of the four borrower archetypes this loan most closely resembles
- `cluster_dist` — Euclidean distance to centroid; a high distance signals
  an atypical loan within its segment, which carries different risk than
  a central member
- `cluster_default_rate` — **excluded**: this encodes the average `Status`
  value of each cluster computed from training labels — indirect target
  leakage that would inflate all downstream metrics

---

### 4.5 Feature Engineering Impact — Isolated Proof

![Feature Engineering Impact](plots/feature_engineering_impact.png)

The same Linear Regression model, same hyperparameters, same stratified
split — only the feature matrix changed:

| Stage | Features | ROC-AUC | R² | F1 (Default) |
|---|---|---|---|---|
| Raw features (Part 3) | 34 | 0.693 | 0.1555 | 0.244 |
| Engineered features (Part 4) | 54 | 0.809 | 0.2539 | 0.519 |
| **Gain** | +20 | **+0.116** | **+0.098 (+63%)** | **+0.275** |

AUC improved by 16.7%, R² by 63%, and F1 on the default class more than
doubled — all with zero model change. This is the strongest possible
evidence that feature engineering drove performance, not model selection.

---
### 4.6 Final Feature Matrix

| Category | Count | Source |
|---|---|---|
| Numeric (scaled) | 8 | StandardScaler |
| Binary flags | 7 | EDA interaction flags |
| One-hot encoded | 30 | OneHotEncoder (14 categoricals) |
| PCA components | 5 | Numeric compression |
| Cluster features | 4 | K-Means (id × 3 + dist) |
| **Total** | **54** | All fit on train only |

---

## 📈 Part 5: Three Improved Regression Models

All three are genuine regression models outputting continuous default probability
scores in [0, 1]. R² is the primary regression metric. Classification metrics
(AUC, F1, Accuracy) are derived by applying a 0.5 threshold to the scores.
All trained on the same 54-feature matrix, same split, same seed.

---

### Model 1 — Linear Regression (Engineered Features)

Same OLS architecture as Part 3 baseline, retrained on 54-feature matrix.
Any improvement over Part 3 isolates the contribution of feature engineering alone.

| Metric | Train | Test |
|---|---|---|
| MAE | 0.2808 | **0.2781** |
| MSE | 0.1385 | **0.1374** |
| RMSE | 0.3721 | **0.3707** |
| R² | 0.2480 | **0.2539** |
| ROC-AUC | — | **0.8094** |
| F1 (Default) | — | **0.5187** |
| Accuracy | — | **80.63%** |
| FNR | — | 57.1% |
| FPR | — | 7.2% |

---

### Model 2 — Ridge Regression

L2-regularized linear regression (α=1.0). Handles multicollinearity in the
correlated PCA + ratio feature block by shrinking unstable coefficients toward zero.

| Metric | Train | Test |
|---|---|---|
| MAE | 0.2788 | **0.2782** |
| MSE | 0.1382 | **0.1371** |
| RMSE | 0.3718 | **0.3702** |
| R² | 0.2480 | **0.2537** |
| ROC-AUC | — | **0.8093** |
| F1 (Default) | — | **0.5188** |
| Accuracy | — | **80.62%** |
| FNR | — | 57.1% |
| FPR | — | 7.3% |

---

### Model 3 — Gradient Boosting Regressor ← WINNER

Sequential tree ensemble minimizing regression loss on the binary 0/1 target.
Outputs continuous scores directly. Captures non-linear feature interactions
natively — no explicit interaction engineering needed.

| Metric | Train | Test |
|---|---|---|
| MAE | 0.1998 | **0.1991** |
| MSE | 0.0928 | **0.0924** |
| RMSE | 0.3046 | **0.3040** |
| R² | 0.4948 | **0.4967** |
| ROC-AUC | — | **0.8807** |
| F1 (Default) | — | **0.7223** |
| Accuracy | — | **88.63%** |
| FNR | — | 39.3% |
| FPR | — | 2.4% |

**No overfitting:** Train R² = 0.4948, Test R² = 0.4967 — test marginally exceeds train.

---

### Full Comparison Table

![ROC Curves](plots/roc_curves.png)

| Model | MAE | RMSE | R² | ROC-AUC | F1 (Default) | Accuracy | FNR | FPR |
|---|---|---|---|---|---|---|---|---|
| **Baseline LR (Part 3)** | 0.3227 | 0.3944 | 0.1094 | 0.693 | 0.244 | 77.8% | 57.1% | 7.2% |
| Linear Reg (Engineered) | 0.2781 | 0.3707 | 0.2539 | 0.809 | 0.519 | 80.6% | 57.1% | 7.2% |
| Ridge Regression | 0.2782 | 0.3702 | 0.2537 | 0.809 | 0.519 | 80.6% | 57.1% | 7.3% |
| **Gradient Boosting Regressor** | **0.1991** | **0.3040** | **0.4967** | **0.881** | **0.722** | **88.6%** | **39.3%** | **2.4%** |

![Confusion Matrices Part 5](plots/confusion_matrices_part5.png)

**Confusion matrix highlights:**

| Model | False Negatives | False Positives | FNR | FPR |
|---|---|---|---|---|
| Linear Reg (Engineered) | 4,083 | 1,605 | 57.1% | 7.2% |
| Ridge Regression | 4,080 | 1,611 | 57.1% | 7.3% |
| **Gradient Boosting Regressor** | **2,806** | **533** | **39.3%** | **2.4%** |

GBR catches **1,277 more actual defaults** while simultaneously generating
**1,072 fewer false alarms** — reducing both error types simultaneously,
which only happens with genuinely better discrimination.

**What these numbers mean:**

Feature engineering alone — same OLS model, same hyperparameters — doubled R²
from 0.109 to 0.254 and improved AUC by 16.7%. This is the most important
finding in the regression section: the quality of features contributed more
than any model change.

Ridge adding near-zero improvement over Linear Regression confirms that PCA
upstream had already resolved the multicollinearity concern — regularization
was solving a problem that no longer existed.

Gradient Boosting Regressor achieves R²=0.497 because loan default is
fundamentally non-linear. The compound risk zone (LTV 75–90% AND DTI
mid-band) reaching 68.9% default cannot be expressed as a sum of independent
feature contributions — it requires a model that captures multiplicative
interactions. GBR discovers these automatically through sequential tree splits.
The confusion matrix confirms genuine discrimination improvement: 1,277 fewer
missed defaults and 1,072 fewer false alarms simultaneously — reducing both
error types at once only happens with better underlying signal, not threshold
adjustment.

### Feature Importance — Part 5

![Feature Importance Part 5](plots/feature_importance_part_5.png)

**Key findings:**
- `is_compound_risk` is the dominant feature in Ridge Regression (+0.44 coefficient)
  — consistent with being #1 in GBR feature importances
- Linear Regression shows `loan_to_property` with an inflated coefficient (~1.2×10⁷)
  due to scale — this is a visualization artifact from the passthrough path, not a
  modeling problem. Ridge regularizes this correctly.
- GBR top features: `is_compound_risk` (0.29) > `loan_to_value_ratio` (0.12) >
  `loan_to_property` (0.11). Both `cluster_dist` and `is_25yr_term` appear —
  confirming clustering and the 25yr term flag added real signal.

### Winner Declaration

**Winner: Gradient Boosting Regressor**

R² nearly doubles from the best linear model (0.254 → 0.497). The model explains
49.7% of default variance — versus 25.4% for Linear and Ridge. Every metric improves
simultaneously. Exported as `best_regression_model.pkl`.

**Why Ridge ≈ Linear Regression:** Regularization helps when multicollinearity is
severe or features outnumber observations. Neither is critical here — 54 features,
117,463 rows, PCA already compressed the correlated numeric block. Ridge added
stability but not predictive improvement.

**Why GBR dominates:** Sequential boosting concentrates each tree on the hardest
residual cases. Non-linear compound risk interactions (LTV × DTI) discovered
automatically. Trees are scale-invariant — no standardization artifacts.

---

## 🏆 Part 6: Winning Regression Model Export



**File:** `best_regression_model.pkl` | **R²:** 0.4967 | **AUC:** 0.881 | **Accuracy:** 88.6%

---

## 🏷️ Part 7: Regression → Classification
![Regression to Classification](plots/regression_classification.png)
| Class | Label | Threshold | N (Train) | Train % | True Default Rate |
|---|---|---|---|---|---|
| 0 | Low Risk | score < 0.20 | 65,268 | 55.6% | **9.4%** |
| 1 | Medium Risk | 0.20 ≤ score < 0.40 | 27,880 | 23.7% | 26.7% |
| 2 | High Risk | score ≥ 0.40 | 24,315 | 20.7% | **61.8%** |

**52.4pp spread** validates threshold quality. Imbalance ratio 2.68:1 — corrected
with `class_weight='balanced'`.

| Error Type | Consequence | Cost |
|---|---|---|
| False Negative | Approved → defaults → principal loss | **5–10× higher** |
| False Positive | Rejected → missed revenue | Opportunity cost |

**Primary metric:** Macro F1 | **Secondary:** Recall on Class 2 (High Risk)

**Why business rule thresholds — not statistical splits:**

Median split collapses three operationally distinct tiers into two, losing the
ability to differentiate standard review from enhanced scrutiny loans. Quantile
binning forces equal class sizes regardless of risk distribution, producing
classes with no financial meaning. The 0.20 / 0.40 thresholds were chosen
because the resulting true default rates (9.4% / 26.7% / 61.8%) span 52.4
percentage points — validating that the regression scores carry real financial
signal. The score distribution confirms this: the 0.40 threshold cleanly
separates the long right tail of stressed borrowers from the main distribution,
which is why High Risk captures 61.7% true defaults while representing only
20.7% of the portfolio. Each tier maps directly to a lending action — Low Risk
to streamlined approval, Medium Risk to standard review, High Risk to enhanced
scrutiny or manual underwriting.

---

## 🧠 Part 8: Classification Models

### Three Models

| Model | Architecture | Key Parameters |
|---|---|---|
| Random Forest | 300 independent trees | max_depth=12, balanced |
| XGBoost (tuned) | Sequential boosted trees | RandomizedSearchCV, 20 iter, 3-fold CV |
| K-Nearest Neighbors | Distance-based | K=15, distance weights, Euclidean |

### XGBoost Tuning Results

| Model | Macro F1 | Accuracy | ROC-AUC |
|---|---|---|---|
| XGBoost (default) | 0.9463 | 0.9507 | 0.9953 |
| **XGBoost (tuned)** | **0.9662** | **0.9696** | **0.9982** |

**Note on high metrics:** Labels derived from regression scores on the same feature
matrix — classifiers learn to replicate tier assignments. The operationally meaningful
validation is the **true default rate within each predicted tier:**

| Predicted Class | N Loans | True Default Rate |
|---|---|---|
| Low Risk | 15,788 | **8.7%** |
| Medium Risk | 7,361 | **26.4%** |
| High Risk | 6,217 | **61.7%** |

**52.8pp spread** — the model is deployable. Loans flagged as High Risk default at
61.7% — 2.5× the portfolio average. Low Risk loans at 8.7% — safe for auto-approval.

**Why the classification metrics appear near-perfect:**

The labels were derived from the regression model's predicted scores on the
same feature matrix the classifiers train on — so the classifiers are learning
to replicate a deterministic bucketing rule, not predicting raw defaults from
scratch. Near-perfect replication of a deterministic threshold is expected and
is not overfitting. The true default rate table above is the operationally
correct validation: it measures whether the risk assignments align with real
financial outcomes. A 52.8pp spread between Low and High Risk tiers, with
High Risk defaulting at 61.7% versus a 24.3% portfolio average, confirms the
model is deployable.

**Why XGBoost beat Random Forest:**
Sequential error correction focuses each tree on the loans previous trees got
wrong — the hard Medium/High boundary cases where the risk is ambiguous.
Random Forest averages 300 independent trees and cannot iteratively concentrate
on difficult cases. For this specific boundary problem, sequential learning wins.

**Why both beat KNN:**
In 54 dimensions, Euclidean distances between all points converge toward the
same value — nearest neighbors become geometrically meaningless. Tree models
build explicit split rules using one feature at a time, remaining valid in
high-dimensional spaces where KNN memorizes without generalizing.


### Evaluation Results

![Three Classification Models Evaluation](plots/three_models_evaluation.png)
![Confusion Matrices Part 8](plots/confusion_matrices_part8.png)

### Threshold Analysis

![Threshold Analysis](plots/threshold_analysis.png)

Optimal threshold for Class 2: **0.42** (not default 0.50). At 0.42, F1 is maximized
for the High Risk class. A lender should deploy at 0.42 — given false negatives cost
5–10× more than false positives, this is the operationally correct operating point.

### Feature Importance — Classification Models

![Feature Importance Part 8](plots/feature_importance_part_8.png)

**`is_compound_risk` ranks #1 in both Random Forest (0.23) and XGBoost (0.27).**
Convergence across two structurally different model families confirms this is real
signal — not a model-specific artifact. `cluster_dist` appears in both top-20
rankings — K-Means segmentation added genuine atypicality signal.

### Winner — XGBoost (Tuned)

**Winner: XGBoost** → `best_model_xgboost.pkl`

Wins on every operational metric. Sequential error correction targets the hard
boundary cases between Medium and High Risk. KNN degrades in 54 dimensions —
curse of dimensionality makes nearest neighbors meaningless.

---

## 📊 Final Evaluation — Key Results

| Milestone | Metric | Value |
|---|---|---|
| Baseline Linear Regression | R² | 0.1555 |
| Baseline Linear Regression | AUC | 0.693 |
| After Feature Engineering (same model) | R² | 0.2539 (+63%) |
| After Feature Engineering (same model) | AUC | 0.809 (+16.7%) |
| Ridge Regression | R² | 0.2537 |
| **Gradient Boosting Regressor** | **R²** | **0.4967** |
| Gradient Boosting Regressor | AUC | 0.881 |
| Gradient Boosting Regressor | Accuracy | 88.6% |
| Gradient Boosting Regressor | FPR | 2.4% |
| Regression → Classification spread | Class 0 vs 2 DR | 9.4% vs 61.8% (+52.4pp) |
| K-Means cluster spread | Low vs High DR | 13.8% vs 31.8% (+18pp) |
| XGBoost — Low Risk tier | True DR | 8.7% |
| XGBoost — Medium Risk tier | True DR | 26.4% |
| XGBoost — High Risk tier | True DR | 61.7% |
| XGBoost — Tier spread | | 52.8pp |

### Bonus Work
- t-SNE alongside PCA for cluster validation
- Business rule thresholding with financial domain justification
- Interactive Plotly visualizations (LTV×DTI heatmap + cluster profiles)
- RandomizedSearchCV hyperparameter tuning on XGBoost
- Two-panel threshold analysis — optimal deployment point (0.42)
- ColumnTransformer pipeline — production-ready ML engineering
- Comprehensive README with full feature dictionary and embedded visuals

---

## 🚀 How to Load and Use the Models

```python
import pickle
import numpy as np

with open("best_model_xgboost.pkl", "rb") as f:
    clf_model = pickle.load(f)

with open("best_regression_model.pkl", "rb") as f:
    reg_model = pickle.load(f)

# Both expect the 54-feature engineered matrix from Part 4
y_score = np.clip(reg_model.predict(X_new), 0, 1)
y_class = clf_model.predict(X_new)

class_map = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
risk_labels = [class_map[c] for c in y_class]
```

---

## 📦 Requirements

```
pandas>=1.3  numpy>=1.21  scikit-learn>=1.0  xgboost>=1.5
matplotlib>=3.4  seaborn>=0.11  plotly>=5.0  scipy>=1.7
```

---

## 📋 Assignment Structure

| Part | Description | Key Output |
|---|---|---|
| Part 2 | EDA — 11 subsections, 20+ plots | Cleaned 146,829-row dataset |
| Part 3 | Baseline Linear Regression (34 features) | R²=0.1555, AUC=0.693 |
| Part 4 | Feature engineering + PCA + clustering | 54-feature matrix |
| Part 5 | Linear Reg + Ridge + GBR | GBR winner R²=0.497, AUC=0.881 |
| Part 6 | Export regression winner | `best_regression_model.pkl` |
| Part 7 | Regression → Classification | 3 tiers, 52.4pp spread |
| Part 8 | RF + XGBoost + KNN | XGBoost winner |

---

## 📝 Key Design Decisions

| Decision | Justification |
|---|---|
| Exclude `upfront_charges` | 0% default in no-fee segment — data artifact |
| Exclude `credit_score` | Pearson r = 0.003 |
| Exclude `interest_rate` | Post-approval pricing — leakage |
| Exclude `credit_bureau` | EQUI = 100% default — post-default assignment |
| Remove `cluster_default_rate` | Target encoding = leakage |
| Use Ridge not Lasso | Feature elimination not desired — all features interpretable |
| Use GBR not GBC | Regression task requires continuous score output with R² |
| Stratified split | Preserves 24.34% class rate |
| Fit transformers on train only | Zero test set information |
| Deploy at threshold 0.42 | Optimal Class 2 F1 — not default 0.50 |
| Recall > Precision | False negatives cost 5–10× more |
| Macro F1 as primary metric | Equal penalty for ignoring any risk class |

---

*Assignment #2 — Data Science Program | May 2026*
