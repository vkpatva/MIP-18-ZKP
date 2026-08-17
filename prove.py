"""Prove that a private loan was scored by the committed MLP.

Does not print the loan JSON or any weight tensors.

Run:
    python prove.py --loan examples/new_loan.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from infer import load_artifact, score_vector
from loan_ml import COMMITMENT_PATH, loan_feature_vector, risk_label
from setup_ezkl import _extract_param_hash, ensure_setup
from zk_paths import (
    COMPILED_PATH,
    INPUT_PATH,
    PARAM_COMMITMENT_PATH,
    PK_PATH,
    PROOF_PATH,
    PUBLIC_PATH,
    SRS_PATH,
    VK_PATH,
    WITNESS_PATH,
)


def _vector_to_input_json(x_row: np.ndarray, path: Path) -> None:
    path.write_text(
        json.dumps({"input_data": [np.asarray(x_row, dtype=float).reshape(-1).tolist()]}),
        encoding="utf-8",
    )


def _probability_from_witness(witness: dict) -> float:
    pretty = witness.get("pretty_elements") or {}
    rescaled = pretty.get("rescaled_outputs")
    if rescaled:
        return float(rescaled[0][0] if isinstance(rescaled[0], list) else rescaled[0])
    raise RuntimeError("EZKL witness is missing rescaled_outputs")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prove MLP scoring of one private loan")
    parser.add_argument("--loan", type=Path, default=Path("examples/new_loan.json"))
    parser.add_argument("--proof", type=Path, default=PROOF_PATH, help="Where to write the proof")
    parser.add_argument("--public", type=Path, default=PUBLIC_PATH, help="Where to write public outputs")
    args = parser.parse_args()

    if not args.loan.exists():
        raise SystemExit(f"Loan file not found: {args.loan}")

    # Feature engineering stays in Python (not in the circuit).
    artifact = load_artifact()
    loan = json.loads(args.loan.read_text(encoding="utf-8"))
    x_row = loan_feature_vector(loan, artifact)
    plaintext_p = score_vector(x_row)

    ensure_setup(force=False)

    import ezkl

    _vector_to_input_json(x_row, INPUT_PATH)
    ezkl.gen_witness(
        str(INPUT_PATH),
        str(COMPILED_PATH),
        str(WITNESS_PATH),
        vk_path=str(VK_PATH),
        srs_path=str(SRS_PATH),
    )
    witness = json.loads(WITNESS_PATH.read_text(encoding="utf-8"))

    zk_p = _probability_from_witness(witness)
    param_hash = _extract_param_hash(witness)
    expected_hash = PARAM_COMMITMENT_PATH.read_text(encoding="utf-8").strip()
    if param_hash.lower() != expected_hash.lower():
        raise SystemExit(
            "Weight hash in this witness does not match ezkl/param_commitment.txt.\n"
            "Re-run: python setup_ezkl.py --force"
        )

    print("Building SNARK (does not print the loan or weights) ...")
    args.proof.parent.mkdir(parents=True, exist_ok=True)
    ok = ezkl.prove(
        str(WITNESS_PATH),
        str(COMPILED_PATH),
        str(PK_PATH),
        str(args.proof),
        srs_path=str(SRS_PATH),
    )
    if not ok:
        raise RuntimeError("ezkl.prove failed")

    if not COMMITMENT_PATH.exists():
        raise SystemExit("model_commitment.txt is missing. Run: python train.py")
    commitment = COMMITMENT_PATH.read_text(encoding="utf-8").strip()

    public = {
        "commitment": commitment,
        "param_commitment": param_hash,
        "risk_tier": risk_label(zk_p),
        "default_probability": round(zk_p, 4),
        "plaintext_probability": round(plaintext_p, 4),
    }
    args.public.write_text(json.dumps(public, indent=2) + "\n", encoding="utf-8")

    print(f"Proof written to {args.proof}")
    print(f"Public outputs written to {args.public}")
    print(f"Model commitment C : {commitment}")
    print(f"Risk tier          : {public['risk_tier']}")
    print(f"ZK P(default)      : {public['default_probability']:.2%}")


if __name__ == "__main__":
    main()
