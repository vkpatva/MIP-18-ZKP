"""Score one new loan with the committed MLP (plaintext, no ZK yet).

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
import numpy as np
import onnxruntime as ort
import pandas as pd

from loan_ml import (
    COMMITMENT_PATH,
    ONNX_PATH,
    PREPROCESS_PATH,
    loan_feature_vector,
    risk_label,
    sha256_file,
)

DEFAULT_LOAN_PATH = Path("examples/new_loan.json")


def load_artifact() -> dict:
    if not PREPROCESS_PATH.exists() or not ONNX_PATH.exists():
        raise SystemExit("No trained MLP yet. Run: python train.py")
    if COMMITMENT_PATH.exists():
        expected = COMMITMENT_PATH.read_text(encoding="utf-8").strip()
        actual = sha256_file(ONNX_PATH)
        if actual != expected:
            raise SystemExit(
                "mlp.onnx does not match model_commitment.txt.\n"
                "Re-run python train.py, or restore the committed files."
            )
    return joblib.load(PREPROCESS_PATH)


def score_vector(x_row: np.ndarray) -> float:
    """Run the ONNX net on one already-preprocessed row (batch size 1)."""
    session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    inp = np.asarray(x_row, dtype=np.float32).reshape(1, -1)
    out = session.run(None, {"input": inp})[0]
    return float(out.reshape(-1)[0])


def score_loan(loan: dict, artifact: dict) -> dict:
    x_row = loan_feature_vector(loan, artifact)
    default_probability = score_vector(x_row)
    return {
        "default_probability": round(default_probability, 4),
        "will_likely_default": bool(default_probability >= 0.50),
        "risk_tier": risk_label(default_probability),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict default risk for one loan")
    parser.add_argument("--loan", type=Path, default=DEFAULT_LOAN_PATH, help="JSON file with one loan")
    parser.add_argument(
        "--demo-row",
        type=int,
        default=None,
        help="Score a row from data.csv and compare to the true label",
    )
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
    print(f"Model commitment C   : {COMMITMENT_PATH.read_text(encoding='utf-8').strip()}")
    print("\nHow to read this:")
    print("  Low Risk    = under 20% chance of default")
    print("  Medium Risk = 20% to 40%")
    print("  High Risk   = 40% or higher")


if __name__ == "__main__":
    main()
