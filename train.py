"""Train a loan-default model from data.csv and save it.

Run:
    python train.py
"""

from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

from loan_ml import TARGET_COLUMN, add_engineered_features, build_model, model_frame

DATA_PATH = Path("data.csv")
MODEL_PATH = Path("loan_risk_model.joblib")
SEED = 42


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

    print(f"Training on {len(X_train):,} loans, testing on {len(X_test):,} loans ...")
    pipeline = build_model()
    pipeline.fit(X_train, y_train)

    proba = pipeline.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.50).astype(int)

    print("\n=== Test results ===")
    print(f"Accuracy : {accuracy_score(y_test, pred):.3f}")
    print(f"ROC-AUC  : {roc_auc_score(y_test, proba):.3f}")
    print("\nClassification report (0 = repaid, 1 = defaulted):")
    print(classification_report(y_test, pred, digits=3))

    artifact = {
        "pipeline": pipeline,
        "extreme_lti_threshold": extreme_lti_threshold,
        "feature_columns": list(X.columns),
    }
    joblib.dump(artifact, MODEL_PATH)
    print(f"Saved trained model to {MODEL_PATH}")
    print("Next step: python infer.py")


if __name__ == "__main__":
    main()
