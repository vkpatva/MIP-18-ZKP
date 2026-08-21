"""Pure verifier math: no loan JSON, no ONNX, no training data.

A beginner picture:

  Prover sent a sealed envelope (the proof) plus a public sticky note
  (risk tier + hashes). This module checks the seal against keys the
  model owner already published. It never opens the loan file.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent


@dataclass
class VerifyResult:
    ok: bool
    reason: str
    risk_tier: str | None = None
    default_probability: float | None = None
    commitment: str | None = None
    param_commitment: str | None = None
    used_onnx: bool = False
    used_loan: bool = False
    used_data_csv: bool = False


def default_paths() -> dict[str, Path]:
    """Public files only. Prefer verifier/bundle/, else the repo ezkl/ folder."""
    bundle = APP_DIR / "bundle"
    ezkl = bundle if (bundle / "vk.key").exists() else REPO_ROOT / "ezkl"
    commitment = bundle / "model_commitment.txt"
    if not commitment.exists():
        commitment = REPO_ROOT / "model_commitment.txt"
    return {
        "proof": ezkl / "proof.json",
        "public": ezkl / "public.json",
        "vk": ezkl / "vk.key",
        "settings": ezkl / "settings.json",
        "srs": ezkl / "kzg.srs",
        "param_commitment": ezkl / "param_commitment.txt",
        "commitment": commitment,
    }


def verify_files(
    proof: Path,
    public_path: Path,
    *,
    vk: Path | None = None,
    settings: Path | None = None,
    srs: Path | None = None,
    commitment: Path | None = None,
    param_commitment: Path | None = None,
) -> VerifyResult:
    paths = default_paths()
    vk = vk or paths["vk"]
    settings = settings or paths["settings"]
    srs = srs or paths["srs"]
    commitment = commitment or paths["commitment"]
    param_commitment = param_commitment or paths["param_commitment"]

    missing = [
        str(p)
        for p in (proof, public_path, vk, settings, srs, commitment, param_commitment)
        if not p.exists()
    ]
    if missing:
        return VerifyResult(
            ok=False,
            reason="Missing public files (still not the loan or weights):\n  " + "\n  ".join(missing),
        )

    expected_c = commitment.read_text(encoding="utf-8").strip()
    public = json.loads(public_path.read_text(encoding="utf-8"))
    if public.get("commitment") != expected_c:
        return VerifyResult(
            ok=False,
            reason="SHA-256 commitment C on the sticky note does not match the published model fingerprint.",
            commitment=str(public.get("commitment")),
        )

    expected_param = param_commitment.read_text(encoding="utf-8").strip()
    got_param = str(public.get("param_commitment", ""))
    if got_param.lower() != expected_param.lower():
        return VerifyResult(
            ok=False,
            reason="Weight hash in the public note is not the committed MLP.",
            param_commitment=got_param,
        )

    import ezkl

    ok = ezkl.verify(str(proof), str(settings), str(vk), srs_path=str(srs))
    if not ok:
        return VerifyResult(ok=False, reason="EZKL rejected the proof. The seal is invalid.")

    prob = public.get("default_probability")
    return VerifyResult(
        ok=True,
        reason="Proof is valid. The committed MLP produced this risk tier. Loan and weights were not used.",
        risk_tier=public.get("risk_tier"),
        default_probability=float(prob) if prob is not None else None,
        commitment=expected_c,
        param_commitment=expected_param,
    )


def main() -> None:
    paths = default_paths()
    parser = argparse.ArgumentParser(description="Verify a loan-risk proof without the loan or weights")
    parser.add_argument("--proof", type=Path, default=paths["proof"])
    parser.add_argument("--public", type=Path, default=paths["public"])
    args = parser.parse_args()
    result = verify_files(args.proof, args.public)
    print(json.dumps(asdict(result), indent=2))
    raise SystemExit(0 if result.ok else 1)


if __name__ == "__main__":
    main()
