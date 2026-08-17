"""Compile the MLP ONNX into an EZKL circuit and create proving/verifying keys.

This is the slow, one-time step. After it finishes, prove.py can make a
proof for a loan and verify.py can check that proof without the weights.

Run:
    python setup_ezkl.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from loan_ml import ONNX_PATH
from zk_paths import (
    CAL_DATA_PATH,
    COMPILED_PATH,
    EZKL_DIR,
    PARAM_COMMITMENT_PATH,
    PK_PATH,
    SETTINGS_PATH,
    SRS_PATH,
    VK_PATH,
    WITNESS_PATH,
)


def _write_cal_data() -> None:
    """A few real-ish vectors so EZKL can size its lookup tables.

    Values are already scaled (what the MLP sees), not raw loan JSON.
    """
    import joblib
    import pandas as pd

    from loan_ml import PREPROCESS_PATH, add_engineered_features, loan_feature_vector, model_frame

    artifact = joblib.load(PREPROCESS_PATH)
    samples: list[list[float]] = []

    example = json.loads(Path("examples/new_loan.json").read_text(encoding="utf-8"))
    samples.append(loan_feature_vector(example, artifact).astype(float).tolist())

    # Extra rows from the training table so lookup ranges cover more than
    # one loan. These stay on the prover machine (gitignored).
    if Path("data.csv").exists() and Path("data.csv").stat().st_size > 1000:
        df = pd.read_csv("data.csv")
        for idx in (0, 10, 100, 1000):
            if idx >= len(df):
                continue
            row = df.iloc[idx].drop(labels=["Status"], errors="ignore").to_dict()
            samples.append(loan_feature_vector(row, artifact).astype(float).tolist())

    # EZKL calibration reads one graph input; we use the example loan as
    # the tensor, which is enough for this tiny net.
    payload = {"input_data": [samples[0]]}
    CAL_DATA_PATH.write_text(json.dumps(payload), encoding="utf-8")


def _extract_param_hash(witness: dict) -> str:
    """Poseidon hash of the weights. This is the in-circuit commitment C."""
    pretty = witness.get("pretty_elements") or {}
    processed = pretty.get("processed_params")
    if processed and isinstance(processed, list) and processed[0]:
        cell = processed[0][0] if isinstance(processed[0], list) else processed[0]
        return str(cell)
    raw = witness.get("processed_params") or {}
    if isinstance(raw, dict) and raw.get("poseidon_hash"):
        return str(raw["poseidon_hash"][0])
    raise RuntimeError(
        "Could not find the weight hash in the EZKL witness. "
        f"Witness keys: {list(witness.keys())}"
    )


# A real k=17 SRS is about 16MB. Empty / half-downloaded files must not pass.
_MIN_SRS_BYTES = 1_000_000


def _srs_is_usable() -> bool:
    return SRS_PATH.exists() and SRS_PATH.stat().st_size >= _MIN_SRS_BYTES


def _ensure_srs(ezkl) -> None:
    """Write a local SRS. Do not use ezkl.get_srs from Python.

    EZKL 23's public download can return before the file exists (or panic
    inside Tokio with 'Python interpreter is not initialized'). The proof
    is still a real SNARK; prover and verifier just share this local file.
    """
    if _srs_is_usable():
        return
    if SRS_PATH.exists():
        SRS_PATH.unlink()
    settings = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    logrows = int(settings["run_args"]["logrows"])
    print(f"Generating a local SRS (logrows={logrows}) ...")
    print("This is shared math, not a secret. It should take well under a minute.")
    ezkl.gen_srs(str(SRS_PATH), logrows)
    if not _srs_is_usable():
        raise RuntimeError(
            f"SRS is missing or too small after gen_srs ({SRS_PATH})."
        )


def ensure_setup(*, force: bool = False) -> None:
    """Create ezkl/ artifacts if they are missing. Safe to call from prove.py."""
    import ezkl

    if not ONNX_PATH.exists():
        raise SystemExit("mlp.onnx is missing. Run: python train.py")

    EZKL_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    need_settings = force or not SETTINGS_PATH.exists()
    if need_settings:
        print("Generating circuit settings (private input, hashed weights, public output) ...")
        run_args = ezkl.PyRunArgs()
        # Loan vector stays hidden. Weights stay hidden. What the verifier
        # sees: Poseidon hash of the weights + the model's output number.
        run_args.input_visibility = "private"
        run_args.param_visibility = "hashed/public"
        run_args.output_visibility = "public"
        ok = ezkl.gen_settings(str(ONNX_PATH), str(SETTINGS_PATH), py_run_args=run_args)
        if not ok:
            raise RuntimeError("ezkl.gen_settings failed")

        _write_cal_data()
        print("Calibrating (this sizes lookup tables from a sample input) ...")
        ok = ezkl.calibrate_settings(
            str(CAL_DATA_PATH),
            str(ONNX_PATH),
            str(SETTINGS_PATH),
            "resources",
            lookup_safety_margin=2,
        )
        if not ok:
            raise RuntimeError("ezkl.calibrate_settings failed")

    if force or not COMPILED_PATH.exists():
        print("Compiling ONNX into an EZKL circuit ...")
        ok = ezkl.compile_circuit(str(ONNX_PATH), str(COMPILED_PATH), str(SETTINGS_PATH))
        if not ok:
            raise RuntimeError("ezkl.compile_circuit failed")

    if force or not _srs_is_usable():
        _ensure_srs(ezkl)

    if force or not VK_PATH.exists() or not PK_PATH.exists():
        print("Running setup (creating proving key + verifying key) ...")
        print("CPU only; often 1–5 minutes for this tiny net.")
        ok = ezkl.setup(
            str(COMPILED_PATH),
            str(VK_PATH),
            str(PK_PATH),
            srs_path=str(SRS_PATH),
        )
        if not ok:
            raise RuntimeError("ezkl.setup failed")

    if force or not PARAM_COMMITMENT_PATH.exists():
        print("Computing the public hash of the (private) weights ...")
        _write_cal_data()
        ezkl.gen_witness(
            str(CAL_DATA_PATH),
            str(COMPILED_PATH),
            str(WITNESS_PATH),
            vk_path=str(VK_PATH),
            srs_path=str(SRS_PATH),
        )
        witness = json.loads(WITNESS_PATH.read_text(encoding="utf-8"))
        param_hash = _extract_param_hash(witness)
        PARAM_COMMITMENT_PATH.write_text(param_hash.strip() + "\n", encoding="utf-8")

    elapsed = time.time() - t0
    print(f"EZKL setup artifacts are ready in {EZKL_DIR}/ ({elapsed:.1f}s).")
    print(f"  public : {SETTINGS_PATH.name}, {VK_PATH.name}, {PARAM_COMMITMENT_PATH.name}")
    print(f"  prover : {COMPILED_PATH.name}, {PK_PATH.name}  (not needed to verify)")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Compile the MLP and create EZKL keys")
    parser.add_argument("--force", action="store_true", help="Rebuild artifacts even if they exist")
    args = parser.parse_args()
    ensure_setup(force=args.force)


if __name__ == "__main__":
    main()
