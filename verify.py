"""Check a proof without the loan file or the weight file.

Needs only:
  - the proof
  - public outputs (risk tier + commitment)
  - verifying key + settings + SRS from setup (not mlp.onnx)

Run:
    python verify.py --proof ezkl/proof.json --commitment model_commitment.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from zk_paths import PARAM_COMMITMENT_PATH, PUBLIC_PATH, SETTINGS_PATH, SRS_PATH, VK_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify an EZKL loan-scoring proof")
    parser.add_argument("--proof", type=Path, required=True, help="Proof file from prove.py")
    parser.add_argument(
        "--commitment",
        type=Path,
        default=Path("model_commitment.txt"),
        help="SHA-256 of mlp.onnx (the file-level commitment C)",
    )
    parser.add_argument("--public", type=Path, default=PUBLIC_PATH, help="Public JSON from prove.py")
    parser.add_argument("--vk", type=Path, default=VK_PATH)
    parser.add_argument("--settings", type=Path, default=SETTINGS_PATH)
    parser.add_argument("--srs", type=Path, default=SRS_PATH)
    args = parser.parse_args()

    missing = [
        str(p)
        for p in (args.proof, args.commitment, args.public, args.vk, args.settings, args.srs)
        if not p.exists()
    ]
    if missing:
        raise SystemExit(
            "Missing files:\n  "
            + "\n  ".join(missing)
            + "\nVerifier never needs mlp.onnx, data.csv, or the loan JSON.\n"
            "If vk/settings/srs are missing, run: python setup_ezkl.py"
        )

    expected_c = args.commitment.read_text(encoding="utf-8").strip()
    public = json.loads(args.public.read_text(encoding="utf-8"))
    if public.get("commitment") != expected_c:
        raise SystemExit(
            "Commitment mismatch.\n"
            f"  --commitment file : {expected_c}\n"
            f"  public.json       : {public.get('commitment')}"
        )

    expected_param = PARAM_COMMITMENT_PATH.read_text(encoding="utf-8").strip()
    if public.get("param_commitment", "").lower() != expected_param.lower():
        raise SystemExit(
            "In-circuit weight hash does not match ezkl/param_commitment.txt.\n"
            "This proof is not for the committed MLP."
        )

    import ezkl

    ok = ezkl.verify(
        str(args.proof),
        str(args.settings),
        str(args.vk),
        srs_path=str(args.srs),
    )
    if not ok:
        raise SystemExit("EZKL verify returned false. Proof is invalid.")

    print("VERIFY SUCCESS")
    print(f"Model commitment C : {expected_c}")
    print(f"Risk tier          : {public.get('risk_tier')}")
    if "default_probability" in public:
        print(f"P(default)         : {public['default_probability']:.2%}")
    print("Verifier did not use mlp.onnx, data.csv, or the loan file.")


if __name__ == "__main__":
    main()
