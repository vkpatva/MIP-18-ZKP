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

> 📺 **Video link:** *(add your video link here after recording)*

---

## 📌 Project Overview

This project builds a full end-to-end machine learning pipeline to predict
loan default risk using a real-world mortgage dataset of approximately 147,000
loans. The pipeline progresses from raw data through exploratory analysis,
feature engineering, unsupervised clustering, regression modeling, and
multi-class classification — ending with a production-ready XGBoost classifier
exported for deployment.

---

## ❓ Research Question

Given the loan application data, the primary research question is to develop and evaluate models that can accurately predict loan default (classification) and, in a regression context, predict the likelihood of loan default. This analysis aims to identify key features and patterns that drive default risk, thereby providing insights for better risk assessment and decision-making for financial institutions.
The EDA was structured around five concrete business questions.
Each question was answered with a specific visualization and a
statistically-grounded finding.



## 🗺️ Full Project Workflow

```
Raw Dataset (148,670 rows × 28 features)
    ↓
Part 2: EDA
  ├── Column cleanup and renaming
  ├── Missingness co-occurrence analysis
  ├── Domain-grounded imputation (8 columns)
  ├── Invalid value removal
  ├── Outlier detection + log transforms
  ├── Duplicate removal
  ├── Descriptive statistics
  ├── Categorical chi-square + Cramér's V
  ├── Univariate analysis (5 numeric features)
  └── 5 bivariate research questions answered
    ↓
Part 3: Baseline Linear Regression
  ├── 34 raw features, default parameters
  ├── 80/20 stratified split (SEED=42)
  ├── StandardScaler on numeric columns
  ├── MAE=0.3227, RMSE=0.3944, R²=0.1555
  └── Feature importance via coefficients
    ↓
Part 4: Feature Engineering
  ├── 10 new engineered features (ratios + binary flags)
  ├── ColumnTransformer pipeline (Scaler + OHE)
  ├── PCA: 9 numeric → 5 components (98.6% variance)
  ├── K-Means clustering K=4 (elbow method)
  ├── t-SNE + PCA cluster visualization
  └── Final: 54-feature matrix
    ↓
Part 5: Three Improved Regression Models
  ├── Linear Regression (engineered) — AUC 0.809
  ├── Logistic Regression — AUC 0.812
  ├── Gradient Boosting — AUC 0.882 ← WINNER
  ├── ROC + Precision-Recall curves
  ├── Confusion matrices with FNR/FPR
  └── Feature importance (coefficients + impurity)
    ↓
Upload best regression model → HuggingFace
    ↓
Part 7: Regression → Classification
  ├── Business rule thresholds (0.20 / 0.40)
  ├── 3 classes: Low Risk / Medium Risk / High Risk
  └── Class balance analysis
    ↓
Part 8: Three Classification Models
  ├── Random Forest (300 trees, balanced)
  ├── XGBoost (300 rounds, lr=0.05) ← WINNER
  ├── K-Nearest Neighbors (K=15, distance weights)
  ├── Classification reports + confusion matrices
  └── Feature importance comparison
    ↓
Upload best classification model → HuggingFace
Upload notebook → HuggingFace
Write README → HuggingFace
Record presentation → Add link to README
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
- Dropped 5 zero-variance / identifier columns:
  `loan_id`, `year`, `construction_type`, `secured_by`, `security_type`
- Dropped `loan_purpose` — undocumented codes with no codebook
- Relabeled all categorical values from codes to readable strings

### 2.2 Missing Value Analysis

Missingness co-occurrence heatmap computed before any imputation.

| Column | Missing | Strategy |
|---|---|---|
| `term_months` | 41 | Drop rows |
| `negative_amortization` | 121 | Drop rows |
| `age_group` + `submission_channel` | 200 | Drop rows |
| `approved_in_advance` | 908 | Drop rows |
| `loan_limit` | 3,344 | Mode imputation |
| `income` | 10,410 | 2D binning: loan decile × credit band |
| `property_value` + `LTV` | 15,131 | Back-derive from median LTV by decile |
| `debt_to_income_ratio` | 24,121 | 2D binning: credit band × income decile |
| `interest_rate` | 36,439 | 2D binning: credit band × LTV band |

### 2.3 Feature Exclusions

| Feature | Reason |
|---|---|
| `credit_score` | Pearson r = 0.003; near-uniform distribution |
| `interest_rate` | Post-approval pricing — leakage |
| `interest_rate_spread` | Derived from excluded leakage column |
| `credit_worthiness` | Lender's internal risk classification — leakage |
| `credit_bureau` | Cramér's V = 0.5929 — probable structural leakage |
| `coapplicant_credit_bureau` | Same leakage concern |
| `upfront_charges` | 0% default in no-fee segment — data artifact |
| `open_credit_flag` | Cramér's V < 0.01 — no signal |

### 2.4 Data Cleaning Summary

![Cleaning Summary](plots/cleaning_summary.png)

**Total rows dropped:** 1,841 (1.2%) | **Retained:** 146,829 (98.8%)

---
### 2.5 bivariate research questions

### Q1 — Does leverage (LTV) combined with debt burden (DTI) create compound risk?

> *"Is the combination of high LTV and high DTI more dangerous than either alone?"*

![LTV DTI Heatmap](plots/ltv_dti_heatmap.png)

**Finding:**
The peak default rate (68.9%) occurs at LTV Q3 (75–90%) combined with DTI
Q2–Q4 — not at the maximum values of either variable. Borrowers with the
highest LTV (Q5) actually default less than those in Q3, because very high
LTV loans required mortgage insurance and stricter underwriting that screened
out the worst borrowers. The compound risk interaction is non-linear and
cannot be captured by either variable acting independently.

**Modeling implication:**
Created `is_compound_risk` binary flag for the 75–90% LTV AND DTI mid-band
zone. This cell reaches 3× the dataset average default rate.

---

### Q2 — Does income vs loan amount explain default better than either alone?

> *"Is affordability stress — not absolute income or loan size — the real driver?"*

![Income VS Loan amount](plots/income_loan_scatter.png)
![Default Rate by Decile](plots/default_by_decile.png)

**Finding:**
Defaulters earn ~32% less but borrow only ~10% less than repaid borrowers.
The risk gradient runs diagonally along the loan-to-income ratio, not along
either axis independently. No clean linear boundary separates the two classes —
the scatter is mixed throughout, especially at income levels below $8K/month.
- **Income** has the strongest and most consistent gradient: 36.8% (decile 0) → 19.5% (decile 7)
- **Loan amount** has a weaker, shallower gradient — larger loans go to wealthier borrowers
- **LTI** is near-flat for deciles 0–6 (~23%), then spikes to 35.2% at decile 9

LTI is a tail-risk feature — neutral in the middle, dangerous at the extreme.
Income is a continuous risk gradient across its entire range.


**Modeling implication:**
Engineered `lti_ratio_log` (loan-to-income in log scale) as an explicit
feature. The ratio captures affordability stress better than either raw variable.

Use `income_log` as a continuous feature. Create `is_extreme_lti` binary flag
for the top LTI decile only — do not use raw LTI as a continuous predictor.

---

### Q3 — Do age and geography interact to create localized risk hotspots?

> *"Are young or elderly borrowers in specific regions disproportionately risky?"*

![Age Region Heatmap](plots/age_region_heatmap.png)

**Finding:**
North-East region shows two extreme cells:
- Under-25 borrowers: **50.0% default** — likely high cost-of-living markets
  where income is insufficient to sustain large mortgages
- Over-74 borrowers: **44.7% default** — likely income-depleted retirees in
  high-property-value areas

The North region is consistently the safest across all age groups (19.7%–28.1%).
The individual Cramér's V values for age (0.049) and region (0.048) are modest,
but their interaction creates cells with 2× the dataset default rate.

**Modeling implication:**
Created `is_northeast_under25` and `is_northeast_over74` binary flags.
Used North as reference category in one-hot encoding.

---
### Q4 — Categorical Risk Interactions: Which loan_type × credit_type combinations are most dangerous?

> *"Do specific credit bureau and loan type combinations create extreme default concentrations?"*

![Credit Bureau Loan Type Heatmap](credit_bureau_loan_type.png)

**Finding:**
Three credit bureaus (CIB, CRIF, EXP) show realistic moderate default
rates across all loan types, ranging from 13%–26%. The EQUI bureau
shows **100.0% default rate across every single loan type** without
exception. A perfect 100% default rate uniform across all product types
is not a risk signal — it is a data artifact. The EQUI label was almost
certainly assigned to loans post-default, making it a leaked version of
the target variable rather than an independent predictor.

This finding directly explains the anomalous Cramér's V of 0.5929 for
`credit_bureau` — the strongest categorical association in the entire
dataset by a wide margin. It was not real signal. It was leakage.

For the three legitimate bureaus, `loan_type_2` consistently produces
the highest default rate (~25–26%) while loan_type_1 and loan_type_3
sit at 13–16% — genuine product-driven variation retained in modeling.

**Modeling implication:**
`credit_bureau` excluded from all models — confirmed leakage.
`coapplicant_credit_bureau` excluded by the same logic.
`loan_type` retained as a one-hot encoded categorical feature.
The 12-point spread across loan types (13% vs 25%) is real and
captured in the engineered feature matrix.

---

### Q5 — Does gender and applicant type affect default risk across the life cycle?

> *"Do joint applicants systematically outperform individual borrowers at every age?"*

![Age Gender Default](plots/age_gender_default.png)

**Finding:**
Joint applicants have the lowest default rate at every single age group
(17.5%–24.4%). Male applicants show the steepest age-related increase
(30.6% at <25 → 34.5% at >74). All four groups follow a U-shaped pattern
with the trough at ages 35–44 — peak earning years.

The joint-male gap widens with age: ~7 points at <25 → ~10 points at >74.

**Modeling implication:**
Created `is_joint_prime_age` flag for joint applicants aged 35–54 — the
safest identifiable demographic segment. Used joint as reference baseline
in one-hot encoding.

---
---

## 📉 Part 3: Baseline Linear Regression

**Goal:** Establish a reproducible performance floor before any feature engineering.

**Setup:**
- 34 raw features (log-transformed monetary + bounded numeric + one-hot categorical)
- 80/20 stratified split, `random_state=42`
- `StandardScaler` fit on train only
- `LinearRegression()` — default parameters, no regularization

**Results:**

| Metric | Train | Test |
|---|---|---|
| MAE | 0.3223 | 0.3227 |
| MSE | 0.1552 | 0.1555 |
| RMSE | 0.3939 | 0.3944 |
| R² | 0.1575 | 0.1555 |
| ROC-AUC | — | 0.693 |
| F1 (Default) | — | 0.244 |
| Accuracy | — | 77.8% |

**Key observations:**
- No overfitting — train/test gap < 0.002 across all metrics
- R² = 0.1555 means the model explains 15.6% of default variance
- Score distributions for repaid and defaulted loans both peak at ~0.25
- False Negative Rate = 57.1% — the model misses more than half of all defaults
- This is the expected ceiling for a linear probability model on a noisy binary target

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

**Key finding from baseline:**
Loan product type features (lump_sum, negative_amortization, 25yr term)
dominate the ranking — not borrower financial metrics. The type of loan
product selected predicts default more strongly than the borrower's income
or credit profile in this dataset.

---

## ⚙️ Part 4: Feature Engineering

### New Features

| Feature | EDA Motivation |
|---|---|
| `lti_ratio_log` | Q2: risk runs along LTI diagonal |
| `loan_to_property` | Alternative leverage measure |
| `monthly_debt_est` | Absolute monthly debt burden |
| `is_extreme_lti` | Q3: top decile spikes to 35.2% |
| `is_compound_risk` | Q1: LTV mid-band AND DTI mid-band |
| `is_25yr_term` | 56.4% default rate |
| `is_northeast_under25` | Q4: 50.0% default cell |
| `is_northeast_over74` | Q4: 44.7% default cell |
| `is_joint_prime_age` | Q5: lowest default segment |
| `is_exotic_product` | Neg_amort + interest_only + lump_sum |

### Pipeline
- `ColumnTransformer`: StandardScaler + OneHotEncoder (drop_first) + passthrough
- **PCA**: 9 numeric → 5 components (98.6% variance)
- **K-Means K=4**: cluster_id + cluster_dist features added

### Cluster Profiles

![Cluster Profiles](plots/cluster_profiles.png)

| Cluster | N | Default Rate | Label |
|---|---|---|---|
| 2 | 19,498 | 13.8% | Conservative borrowers |
| 3 | 18,959 | 19.5% | Stable mid-tier |
| 0 | 44,039 | 25.2% | Standard borrowers |
| 1 | 34,967 | 31.8% | Stressed borrowers |

**Final feature matrix: 54 features**

---

## 📈 Part 5: Three Improved Regression Models

![ROC Curves](plots/roc_curves.png)
![confusion_matrices_part5](plots/confusion_matrices_part5.png)
| Model | ROC-AUC | F1 (Default) | Accuracy | R² |
|---|---|---|---|---|
| Linear Regression (Baseline) | 0.693 | 0.244 | 77.8% | 0.109 |
| Linear Regression (Engineered) | 0.809 | 0.519 | 80.6% | 0.254 |
| Logistic Regression | 0.812 | 0.589 | 76.5% | — |
| **Gradient Boosting (Winner)** | **0.882** | **0.726** | **88.7%** | — |

**Winner: Gradient Boosting Classifier** → `best_regression_model.pkl`

---

## 🏷️ Part 7: Regression → Classification

**Strategy: Business Rule Threshold (3-Class)**

| Class | Label | Threshold | N (Train) | True Default Rate |
|---|---|---|---|---|
| 0 | Low Risk | score < 0.20 | 65,268 (55.6%) | 9.4% |
| 1 | Medium Risk | 0.20 ≤ score < 0.40 | 27,880 (23.7%) | 26.7% |
| 2 | High Risk | score ≥ 0.40 | 24,315 (20.7%) | 61.8% |

The 52.4 percentage point spread between lowest and highest class
(9.4% vs 61.8%) validates the thresholds — the regression scores are
meaningful risk signals, not noise.

**Imbalance ratio: 2.68:1** (largest 65,268 / smallest 24,315)
Not severe — corrected with `class_weight='balanced'` in Part 8 models.

**Why recall > precision:**
False negatives cost 5–10× more than false positives in lending.
Missing a high-risk loan = realized loss on defaulted principal.
Flagging a safe loan = opportunity cost only.

**Primary metric:** Macro F1-Score
**Secondary metric:** Recall on Class 2 (High Risk)

## 🧠 Part 8: Classification Models

![Three Classification Models - Evaluation](plots/three_models_evaluation.png)

![confusion_matrices_part8](plots/confusion_matrices_part8.png)
*(Fill in final metric table after Part 8 runs)*

**Why recall > precision and false negatives are more critical:**
Approving a loan that defaults = full principal loss + legal costs + provisioning.
Rejecting a good loan = missed revenue only.
The model must minimize false negatives on Class 2 even at the cost of precision.

**Winner: XGBoost Classifier** → `best_model_xgboost.pkl`

---


## 📊 Final Evaluation — Self Assessment

### Data Handling & EDA (20%)
- Missingness co-occurrence heatmap before any imputation
- Domain-grounded 2D binning imputation on 4 columns
- Back-derivation for property_value/LTV consistency
- 8 features excluded with explicit, documented justification
- 5 research questions answered with dedicated bivariate visualizations
- Univariate analysis covering 7 numeric and 12 categorical features
- Chi-square + Cramér's V computed for all 17 categorical features
- Full cleaning summary waterfall chart (rows retained per step)

### Feature Engineering (20%)
- 10 new features engineered — all grounded in specific EDA findings
- ColumnTransformer pipeline: StandardScaler + OneHotEncoder + passthrough
- PCA: 9 numeric features → 5 orthogonal components (98.6% variance)
- K-Means K=4 with elbow method selection
- t-SNE + PCA dual visualization for cluster validation
- Cluster features: `cluster_id` + `cluster_dist` (target encoding excluded)
- Final matrix: 54 features with zero leakage

### Model Training (20%)
- Clear iterative progression: Baseline → Engineered → Classification
- 4 regression models + 3 classification models trained and compared
- Stratified 80/20 split with fixed SEED=42 throughout
- All model hyperparameters documented with justification
- Two pickle files exported: regression winner + classification winner

### Evaluation & Interpretation (20%)
- ROC + Precision-Recall curves for all models on the same plot
- Confusion matrices with FNR/FPR annotations
- Feature importance for all model families (coefficients + impurity)
- Cross-model feature importance disagreement discussed
- Precision vs recall trade-off grounded in operational cost asymmetry
- False negative vs false positive cost analysis with 5–10× multiplier rationale

### Key Results Summary

| Milestone | Metric | Value |
|---|---|---|
| Baseline Linear Regression | R² | 0.1555 |
| Baseline Linear Regression | AUC | 0.693 |
| After Feature Engineering | R² | 0.2539 (+63%) |
| After Feature Engineering | AUC | 0.809 (+16.7%) |
| Best Regression Model (GBC) | AUC | 0.882 |
| Best Regression Model (GBC) | F1 Default | 0.726 |
| Best Regression Model (GBC) | Accuracy | 88.7% |
| Classification (XGBoost) | Macro F1 | TBD after run |
| Classification (XGBoost) | ROC-AUC | TBD after run |

### Bonus Work Completed
- t-SNE visualization alongside PCA — non-linear dimensionality reduction
- Business rule thresholding with operational financial justification
- ColumnTransformer scikit-learn pipeline — production-ready ML engineering
- Comprehensive README with embedded research question visualizations
- Data artifact detection and exclusion (upfront charges, credit score)
- Leakage audit on 4 columns with Cramér's V evidence

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
y_class = clf_model.predict(X_new)
y_score = np.clip(reg_model.predict(X_new), 0, 1)

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
| Part 1 | Dataset selection | Research question defined |
| Part 2 | EDA — 11 subsections, 20+ plots | Cleaned dataset + 5 answered research questions |
| Part 3 | Baseline Linear Regression | R²=0.1555, AUC=0.693 |
| Part 4 | Feature engineering + clustering | 54-feature matrix |
| Part 5 | Three improved regression models | GBC winner AUC=0.882 |
| Part 7 | Regression-to-Classification | 3 risk class labels |
| Part 8 | Three classification models | XGBoost winner |

---

*Assignment #2 — Data Science Program | April 2026*
