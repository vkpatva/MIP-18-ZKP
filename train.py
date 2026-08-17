"""Train the official MLP from data.csv, then save ONNX + a hash of it.

Also fits one leftover XGBoost model on the **same** test split so we can
honestly print both AUCs. XGBoost is not saved and is not the deployed net.

Run:
    python train.py
"""

from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

from loan_ml import (
    COMMITMENT_PATH,
    ONNX_PATH,
    PREPROCESS_PATH,
    TARGET_COLUMN,
    add_engineered_features,
    build_mlp,
    build_preprocessor,
    export_mlp_onnx,
    model_frame,
    sha256_file,
)

DATA_PATH = Path("data.csv")
SEED = 42


def _fit_xgboost_baseline(X_train, y_train):
    """Same hyperparameters as the old official model, used only for AUC."""
    from xgboost import XGBClassifier

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=4,
        random_state=SEED,
    )
    model.fit(X_train, y_train)
    return model


def _onnx_probability(onnx_path: Path, x_row: np.ndarray) -> float:
    """Run the exported ONNX on one row. Used to check it matches sklearn."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp = np.asarray(x_row, dtype=np.float32).reshape(1, -1)
    out = session.run(None, {"input": inp})[0]
    return float(out.reshape(-1)[0])


def main() -> None:
    if not DATA_PATH.exists() or DATA_PATH.stat().st_size < 1000:
        raise SystemExit(
            "data.csv is missing or still a Git LFS pointer.\n"
            "Download it first, then run this script again."
        )

    print("Loading data.csv ...")
    import pandas as pd

    raw = pd.read_csv(DATA_PATH)
    featured = add_engineered_features(raw)
    extreme_lti_threshold = featured.attrs["extreme_lti_threshold"]

    X = model_frame(featured)
    y = featured[TARGET_COLUMN].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )

    print("Fitting scaler + one-hot encoder on the training loans ...")
    preprocess = build_preprocessor()
    Xt_train = preprocess.fit_transform(X_train)
    Xt_test = preprocess.transform(X_test)
    n_features = int(Xt_train.shape[1])
    print(f"Each loan is now a vector of {n_features} numbers (this is what the MLP sees).")

    print(f"Training MLP on {len(X_train):,} loans, testing on {len(X_test):,} loans ...")
    mlp = build_mlp()
    mlp.fit(Xt_train, y_train)

    mlp_proba = mlp.predict_proba(Xt_test)[:, 1]
    mlp_pred = (mlp_proba >= 0.50).astype(int)
    mlp_auc = float(roc_auc_score(y_test, mlp_proba))
    mlp_acc = float(accuracy_score(y_test, mlp_pred))

    print("\n=== MLP test results (official model) ===")
    print(f"Accuracy : {mlp_acc:.3f}")
    print(f"ROC-AUC  : {mlp_auc:.3f}")
    print("\nClassification report (0 = repaid, 1 = defaulted):")
    print(classification_report(y_test, mlp_pred, digits=3))

    print("Training leftover XGBoost on the same split (comparison only, not saved) ...")
    xgb = _fit_xgboost_baseline(Xt_train, y_train)
    xgb_proba = xgb.predict_proba(Xt_test)[:, 1]
    xgb_auc = float(roc_auc_score(y_test, xgb_proba))
    print("\n=== Same-split AUC comparison ===")
    print(f"MLP      ROC-AUC : {mlp_auc:.3f}   <-- this is the model we deploy and prove")
    print(f"XGBoost  ROC-AUC : {xgb_auc:.3f}   <-- stronger on this table, not ZK-friendly")

    n_in = export_mlp_onnx(mlp, ONNX_PATH)
    if n_in != n_features:
        raise RuntimeError(f"ONNX input width {n_in} != preprocessor width {n_features}")

    sample = np.asarray(Xt_test[0], dtype=np.float32)
    sklearn_p = float(mlp.predict_proba(sample.reshape(1, -1))[0, 1])
    onnx_p = _onnx_probability(ONNX_PATH, sample)
    if abs(sklearn_p - onnx_p) > 1e-5:
        raise RuntimeError(
            f"ONNX does not match sklearn on a test row ({sklearn_p:.6f} vs {onnx_p:.6f})."
        )
    print(f"\nONNX matches sklearn on a test row (p={onnx_p:.4f}). Saved {ONNX_PATH}")

    commitment = sha256_file(ONNX_PATH)
    COMMITMENT_PATH.write_text(commitment + "\n", encoding="utf-8")
    print(f"Model commitment C (SHA-256 of {ONNX_PATH}): {commitment}")

    artifact = {
        "extreme_lti_threshold": extreme_lti_threshold,
        "feature_columns": list(X.columns),
        "n_features": n_features,
        "mlp_test_auc": mlp_auc,
        "xgb_test_auc": xgb_auc,
        "preprocess": preprocess,
    }
    joblib.dump(artifact, PREPROCESS_PATH)
    print(f"Saved preprocessor (needed for infer/prove, not for verify) to {PREPROCESS_PATH}")
    print("Next step: python infer.py")


if __name__ == "__main__":
    main()
