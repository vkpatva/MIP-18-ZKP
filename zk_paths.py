"""Shared EZKL file locations.

Public (safe to ship to a verifier): settings.json, vk.key, param_commitment.txt
Private to the prover: mlp.onnx, pk.key, compiled circuit, loan input, witness
"""

from pathlib import Path

EZKL_DIR = Path("ezkl")
SETTINGS_PATH = EZKL_DIR / "settings.json"
COMPILED_PATH = EZKL_DIR / "model.compiled"
VK_PATH = EZKL_DIR / "vk.key"
PK_PATH = EZKL_DIR / "pk.key"
SRS_PATH = EZKL_DIR / "kzg.srs"
CAL_DATA_PATH = EZKL_DIR / "cal_data.json"
INPUT_PATH = EZKL_DIR / "input.json"
WITNESS_PATH = EZKL_DIR / "witness.json"
PROOF_PATH = EZKL_DIR / "proof.json"
PUBLIC_PATH = EZKL_DIR / "public.json"
PARAM_COMMITMENT_PATH = EZKL_DIR / "param_commitment.txt"
