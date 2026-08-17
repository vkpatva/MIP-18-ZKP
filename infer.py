"""Score one new loan (or a known row from data.csv).

Examples:
    python infer.py
    python infer.py --loan examples/new_loan.json
    python infer.py --demo-row 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from loan_ml import add_engineered_features, model_frame, risk_label

MODEL_PATH = Path("loan_risk_model.joblib")
DEFAULT_LOAN_PATH = Path("examples/new_loan.json")


def load_artifact():
    if not MODEL_PATH.exists():
        raise SystemExit("No trained model yet. Run: python train.py")
    return joblib.load(MODEL_PATH)


def score_loan(loan: dict, artifact: dict) -> dict:
    row = pd.DataFrame([loan])
    featured = add_engineered_features(
        row, extreme_lti_threshold=artifact["extreme_lti_threshold"]
    )
    X = model_frame(featured)
    pipeline = artifact["pipeline"]
    default_probability = float(pipeline.predict_proba(X)[0, 1])
    return {
        "default_probability": round(default_probability, 4),
        "will_likely_default": bool(default_probability >= 0.50),
        "risk_tier": risk_label(default_probability),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict default risk for one loan")
    parser.add_argument("--loan", type=Path, default=DEFAULT_LOAN_PATH, help="JSON file with one loan")
    parser.add_argument("--demo-row", type=int, default=None, help="Score a row from data.csv and compare to the true label")
    args = parser.parse_args()

    artifact = load_artifact()

    if args.demo_row is not None:
        df = pd.read_csv("data.csv")
        row = df.iloc[args.demo_row]
        loan = row.drop(labels=["Status"], errors="ignore").to_dict()
        result = score_loan(loan, artifact)
        actual = int(row["Status"])
        print(f"Scoring data.csv row {args.demo_row}")
        print(f"True label     : {actual} ({'defaulted' if actual == 1 else 'repaid'})")
        print(f"Predicted P(default) : {result['default_probability']:.2%}")
        print(f"Risk tier      : {result['risk_tier']}")
        return

    if not args.loan.exists():
        raise SystemExit(f"Loan file not found: {args.loan}")

    loan = json.loads(args.loan.read_text())
    result = score_loan(loan, artifact)

    print(f"Loan file: {args.loan}")
    print(f"Predicted P(default) : {result['default_probability']:.2%}")
    print(f"Likely to default    : {result['will_likely_default']}")
    print(f"Risk tier            : {result['risk_tier']}")
    print("\nHow to read this:")
    print("  Low Risk    = under 20% chance of default")
    print("  Medium Risk = 20% to 40%")
    print("  High Risk   = 40% or higher")


if __name__ == "__main__":
    main()
