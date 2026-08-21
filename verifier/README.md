# Verifier desk

A tiny local web app for someone who should **not** see the loan or the model weights.

It only reads public files: the SNARK proof, the sticky-note JSON (risk tier + hashes), and the verifying keys the model owner already published.

## Run

From the repo (reuse the same `.venv` that has EZKL):

```text
source ../.venv/bin/activate   # if you are already in verifier/
pip install -r requirements.txt
python app.py
```

Or from the repo root:

```text
source .venv/bin/activate
pip install -r verifier/requirements.txt
python verifier/app.py
```

Open **http://127.0.0.1:8765**.

- Leave both file boxes empty and click **Stamp the packet** to check the sample already in `ezkl/` (after `python prove.py`). Expect **VERIFIED** / **Low Risk** / ~1.4%.
- Or choose **Proof** = `../ezkl/proof.json` and **Public note** = `../ezkl/public.json`, then stamp.

You still never need `mlp.onnx` or `examples/new_loan.json`. If the stamp says REJECTED, rebuild proofs first (`rm -rf ezkl` is optional; then `python setup_ezkl.py` and `python prove.py --loan examples/new_loan.json`).

Command line, no browser:

```text
python check.py --proof ../ezkl/proof.json --public ../ezkl/public.json
```

## Files this folder may read

| File | Why |
| --- | --- |
| `../ezkl/proof.json` | the seal |
| `../ezkl/public.json` | risk tier + hashes the prover chose to publish |
| `../ezkl/vk.key`, `settings.json`, `kzg.srs` | published verifying kit |
| `../model_commitment.txt` | fingerprint `C` of the ONNX file |

It will not look for `data.csv`, `mlp.onnx`, or the loan JSON. To run this folder on another machine, copy those public files into `bundle/` (same names) next to `app.py`.
