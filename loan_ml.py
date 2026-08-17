"""Shared loan-risk helpers used by train.py and infer.py.

This project predicts whether a mortgage is likely to default, then
maps that probability to Low / Medium / High risk.

The official model is a small MLP (a tiny neural net) so EZKL can later
prove the exact same forward pass. Feature engineering stays here in
Python; the ONNX file is only the net, not pandas.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Columns that leak the answer or add no useful signal. Never use these
# as model inputs.
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

# One hidden layer, kept small so a ZK circuit can actually prove it.
HIDDEN_WIDTH = 32

PREPROCESS_PATH = Path("preprocess.joblib")
ONNX_PATH = Path("mlp.onnx")
COMMITMENT_PATH = Path("model_commitment.txt")


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


def loan_feature_vector(loan: dict, artifact: dict) -> np.ndarray:
    """Turn one loan JSON into the 59 numbers the MLP/ONNX expects.

    This pandas step is *outside* the ZK circuit. The proof later attests
    only that those numbers were run through the committed net.
    """
    row = pd.DataFrame([loan])
    featured = add_engineered_features(
        row, extreme_lti_threshold=artifact["extreme_lti_threshold"]
    )
    X = model_frame(featured)
    return np.asarray(artifact["preprocess"].transform(X), dtype=np.float32).reshape(-1)


def build_preprocessor() -> ColumnTransformer:
    """Scale numbers and one-hot encode categories.

    This step lives in Python, not in the ZK circuit. The proved net only
    sees the already-scaled vector of numbers.
    """
    return ColumnTransformer(
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


def build_mlp() -> MLPClassifier:
    """Small 1-hidden-layer net: input -> 32 ReLU neurons -> one probability.

    sklearn trains it; we later copy the same weights into ONNX so EZKL
    proves this net, not a different one.
    """
    return MLPClassifier(
        hidden_layer_sizes=(HIDDEN_WIDTH,),
        activation="relu",
        solver="adam",
        max_iter=80,
        random_state=42,
        # Early stop so training does not run all 80 epochs if the loss
        # already stopped improving. Still uses the same random_state.
        early_stopping=True,
        validation_fraction=0.10,
        n_iter_no_change=8,
    )


def build_model() -> Pipeline:
    """Preprocessor + MLP in one sklearn object (used for training)."""
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("model", build_mlp()),
        ]
    )


def risk_label(default_probability: float) -> str:
    if default_probability < LOW_RISK_MAX:
        return CLASS_MAP[0]
    if default_probability < MEDIUM_RISK_MAX:
        return CLASS_MAP[1]
    return CLASS_MAP[2]


def export_mlp_onnx(mlp: MLPClassifier, path: Path) -> int:
    """Write a tiny ONNX file: Gemm -> ReLU -> Gemm -> Sigmoid.

    ONNX is a list of math ops (not Python). EZKL can read it. Batch size
    is fixed at 1 so the circuit shape never changes.
    """
    from onnx import TensorProto, helper, numpy_helper, save
    from onnx import checker as onnx_checker

    if len(mlp.coefs_) != 2:
        raise ValueError("Expected exactly one hidden layer (two weight matrices).")

    # sklearn stores W as (in, out). ONNX Gemm uses Y = X @ W + B with the
    # same layout, so we copy the arrays as-is.
    w1 = np.asarray(mlp.coefs_[0], dtype=np.float32)
    b1 = np.asarray(mlp.intercepts_[0], dtype=np.float32)
    w2 = np.asarray(mlp.coefs_[1], dtype=np.float32)
    b2 = np.asarray(mlp.intercepts_[1], dtype=np.float32)

    if w2.shape[1] != 1:
        raise ValueError(f"Expected a single sigmoid output, got {w2.shape}")

    n_in = int(w1.shape[0])
    graph = helper.make_graph(
        nodes=[
            helper.make_node("Gemm", ["input", "W1", "B1"], ["hidden"], name="hidden_linear"),
            helper.make_node("Relu", ["hidden"], ["hidden_relu"], name="hidden_relu"),
            helper.make_node("Gemm", ["hidden_relu", "W2", "B2"], ["logit"], name="output_linear"),
            helper.make_node("Sigmoid", ["logit"], ["output"], name="output_sigmoid"),
        ],
        name="loan_mlp",
        inputs=[helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, n_in])],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1])],
        initializer=[
            numpy_helper.from_array(w1, name="W1"),
            numpy_helper.from_array(b1, name="B1"),
            numpy_helper.from_array(w2, name="W2"),
            numpy_helper.from_array(b2, name="B2"),
        ],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
        ir_version=8,
        producer_name="loan-default-risk-predictor",
    )
    onnx_checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    save(model, str(path))
    return n_in


def sha256_file(path: Path) -> str:
    """Fingerprint of a file. Change one weight and this hex string changes."""
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
