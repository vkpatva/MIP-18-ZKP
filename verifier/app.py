"""Tiny local web app for a verifier who should never see the loan.

Run from anywhere:

    python verifier/app.py

Then open http://127.0.0.1:8765
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from flask import Flask, jsonify, render_template, request

from check import VerifyResult, default_paths, verify_files

app = Flask(
    __name__,
    static_folder=str(APP_DIR / "static"),
    template_folder=str(APP_DIR / "templates"),
)


def _save_upload(upload, dest: Path) -> None:
    dest.write_bytes(upload.read())


def _result_dict(result: VerifyResult) -> dict:
    return {
        "ok": result.ok,
        "reason": result.reason,
        "risk_tier": result.risk_tier,
        "default_probability": result.default_probability,
        "commitment": result.commitment,
        "param_commitment": result.param_commitment,
        "used_onnx": False,
        "used_loan": False,
        "used_data_csv": False,
    }


@app.get("/")
def index():
    paths = default_paths()
    defaults_ready = paths["proof"].exists() and paths["public"].exists() and paths["vk"].exists()
    return render_template(
        "index.html",
        defaults_ready=defaults_ready,
        proof_name=paths["proof"].name if defaults_ready else "",
        commitment_preview=(
            paths["commitment"].read_text(encoding="utf-8").strip()[:16] + "…"
            if paths["commitment"].exists()
            else "—"
        ),
    )


@app.post("/verify")
def verify():
    """Accept optional uploads. Empty form = use the sample public proof in ezkl/."""
    paths = default_paths()
    proof_file = request.files.get("proof")
    public_file = request.files.get("public")
    using_uploads = bool(proof_file and proof_file.filename and public_file and public_file.filename)

    if using_uploads:
        with tempfile.TemporaryDirectory(prefix="zk-verify-") as tmp:
            tmpdir = Path(tmp)
            proof_path = tmpdir / "proof.json"
            public_path = tmpdir / "public.json"
            _save_upload(proof_file, proof_path)
            _save_upload(public_file, public_path)
            result = verify_files(proof_path, public_path)
    else:
        if not paths["proof"].exists() or not paths["public"].exists():
            result = VerifyResult(
                ok=False,
                reason="No sample proof on disk. Upload proof.json and public.json from the prover.",
            )
        else:
            result = verify_files(paths["proof"], paths["public"])

    return jsonify(_result_dict(result))


def main() -> None:
    print("Verifier desk: http://127.0.0.1:8765")
    print("This process does not load mlp.onnx, data.csv, or any loan JSON.")
    app.run(host="127.0.0.1", port=8765, debug=False)


if __name__ == "__main__":
    main()
