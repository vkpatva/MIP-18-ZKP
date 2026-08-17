"""Shared loan-risk helpers used by train.py and infer.py.

This project predicts whether a mortgage is likely to default, then
maps that probability to Low / Medium / High risk.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

# Columns that leak the answer or add no useful signal. Never use these
# as model inputs. See README.md "Feature Exclusions".
LEAKAGE_COLUMNS = [
    "credit_score",
    "interest_rate",
    "interest_rate_spread",
    "credit_worthiness",
    "credit_bureau",
    "coapplicant_credit_bureau",
    "upfront_charges",
    "upfront_charges_log",
    "has_upfront_charges",
    "open_credit_flag",
]

TARGET_COLUMN = "Status"

NUMERIC_FEATURES = [
    "loan_amount_log",
    "property_value_log",
    "income_log",
    "loan_to_value_ratio",
    "debt_to_income_ratio",
    "lti_ratio_log",
    "loan_to_property",
    "monthly_debt_est",
]

BINARY_FEATURES = [
    "is_extreme_lti",
    "is_compound_risk",
    "is_25yr_term",
    "is_northeast_under25",
    "is_northeast_over74",
    "is_joint_prime_age",
    "is_exotic_product",
]

CATEGORICAL_FEATURES = [
    "loan_limit",
    "gender",
    "approved_in_advance",
    "loan_type",
    "business_or_commercial",
    "negative_amortization",
    "interest_only_flag",
    "lump_sum_payment_flag",
    "occupancy_type",
    "total_units",
    "age_group",
    "submission_channel",
    "region",
    "term_category",
]

# Same business cutoffs used in the original assignment (Part 7).
LOW_RISK_MAX = 0.20
MEDIUM_RISK_MAX = 0.40

CLASS_MAP = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}


def term_category_from_months(term_months: float) -> str:
    mapping = {360: "30yr", 180: "15yr", 240: "20yr", 300: "25yr"}
    return mapping.get(int(round(float(term_months))), "other")


def add_engineered_features(df: pd.DataFrame, extreme_lti_threshold: float | None = None) -> pd.DataFrame:
    """Add the extra columns the model needs.

    If extreme_lti_threshold is None (training), use the 90th percentile of
    loan-to-income. During inference, pass the saved training threshold.
    """
    out = df.copy()

    out["loan_amount_log"] = np.log1p(out["loan_amount"].astype(float))
    out["property_value_log"] = np.log1p(out["property_value"].astype(float))
    out["income_log"] = np.log1p(out["income"].astype(float))

    income = out["income"].astype(float).clip(lower=1.0)
    property_value = out["property_value"].astype(float).clip(lower=1.0)
    ltv = out["loan_to_value_ratio"].astype(float)
    dti = out["debt_to_income_ratio"].astype(float)

    out["lti_ratio"] = out["loan_amount"].astype(float) / income
    out["lti_ratio_log"] = np.log1p(out["lti_ratio"])
    out["loan_to_property"] = out["loan_amount"].astype(float) / property_value
    out["monthly_debt_est"] = dti * income / 100.0

    if "term_category" not in out.columns:
        out["term_category"] = out["term_months"].apply(term_category_from_months)

    if extreme_lti_threshold is None:
        extreme_lti_threshold = float(out["lti_ratio"].quantile(0.90))

    region = out["region"].astype(str)
    age = out["age_group"].astype(str)
    gender = out["gender"].astype(str)

    out["is_extreme_lti"] = (out["lti_ratio"] >= extreme_lti_threshold).astype(int)
    out["is_compound_risk"] = ((ltv >= 75) & (ltv <= 90) & (dti >= 34) & (dti <= 43)).astype(int)
    out["is_25yr_term"] = (out["term_category"] == "25yr").astype(int)
    out["is_northeast_under25"] = ((region.str.lower() == "north-east") & (age == "<25")).astype(int)
    out["is_northeast_over74"] = ((region.str.lower() == "north-east") & (age == ">74")).astype(int)
    out["is_joint_prime_age"] = ((gender == "Joint") & age.isin(["35-44", "45-54"])).astype(int)
    out["is_exotic_product"] = (
        (out["negative_amortization"].astype(str).str.lower() == "yes")
        | (out["interest_only_flag"].astype(str).str.lower() == "yes")
        | (out["lump_sum_payment_flag"].astype(str).str.lower() == "yes")
    ).astype(int)

    out.attrs["extreme_lti_threshold"] = float(extreme_lti_threshold)
    return out


def model_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df[NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES]


def build_model() -> Pipeline:
    """One object that scales numbers, encodes categories, then trains XGBoost."""
    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            (
                "model",
                XGBClassifier(
                    n_estimators=200,
                    max_depth=5,
                    learning_rate=0.08,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    n_jobs=4,
                    random_state=42,
                ),
            ),
        ]
    )


def risk_label(default_probability: float) -> str:
    if default_probability < LOW_RISK_MAX:
        return CLASS_MAP[0]
    if default_probability < MEDIUM_RISK_MAX:
        return CLASS_MAP[1]
    return CLASS_MAP[2]
