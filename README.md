# Loan default risk predictor (MLP + EZKL)

A small neural net scores one mortgage as **Low / Medium / High** default risk. A zero-knowledge proof (EZKL) then shows that a **private** loan was run through a **committed** MLP — without sharing the loan JSON or the weight file.

XGBoost is stronger on this table (test AUC **0.886**). We still deploy the MLP (test AUC **0.857**) so EZKL can prove the **same** net we score with, not a fake copy of the trees.

## Install

```text
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r verifier/requirements.txt
```

You need `data.csv` in this folder to **train**. Infer, prove, verify, and the UI do not need it once `mlp.onnx` exists.

## Commands

```text
python train.py                              # MLP + ONNX + SHA-256 commitment C
python infer.py --loan examples/new_loan.json
python infer.py --loan examples/new_loan.json --prove
python setup_ezkl.py                         # circuit + keys (or let prove.py do this)
python verify.py --proof ezkl/proof.json --commitment model_commitment.txt
```

`infer.py` prints the plaintext score. Add `--prove` to also build the SNARK (`ezkl/proof.json`) for the UI. `python prove.py --loan ...` does the same combined step. On this CPU, `setup_ezkl.py` was about **20–30 seconds**, proving about **15 seconds**, `verify.py` under **1 second**. Setup writes a proving key of about **1.4 GB** at `ezkl/pk.key` (gitignored, prover only).

Skip `python train.py` unless you deleted `mlp.onnx` / `preprocess.joblib`. Retrain only to get a new net, not to rebuild proofs.

## Verifier UI

This is a local page for someone who should **not** see the loan or the weights.

```text
python verifier/app.py
```

Keep that terminal open. In a browser open **http://127.0.0.1:8765**.

1. **Sample packet** — leave both file pickers empty. Click **Stamp the packet**. It uses `ezkl/proof.json` + `ezkl/public.json` from the last `--prove`. The sample loan is **Low Risk** (~1.4%).
2. **Your own packet** — after `python infer.py --loan ... --prove`, choose **Proof** = `ezkl/proof.json` and **Public note** = `ezkl/public.json`, then stamp.

Do not upload `mlp.onnx` or the loan JSON. The desk only needs the proof, the public note, and the verifying files already on disk (`vk.key`, `settings.json`, `kzg.srs`, commitment hashes).

Stop the server with Ctrl+C. More detail: [`verifier/README.md`](verifier/README.md).

Terminal-only check (same math, no browser):

```text
python verify.py --proof ezkl/proof.json --commitment model_commitment.txt
python verifier/check.py
```

## Rebuild `ezkl/` from scratch

You can delete the whole folder. It is generated. Keep `mlp.onnx` unless you also want to retrain.

```text
rm -rf ezkl
python setup_ezkl.py
python prove.py --loan examples/new_loan.json
python verify.py --proof ezkl/proof.json --commitment model_commitment.txt
```

Then refresh the UI and stamp again. Setup generates a **local** `ezkl/kzg.srs` (do not rely on EZKL’s public download from Python — it can return success without writing a usable file). Prove and verify must share that same SRS.

To free ~1.4 GB without wiping keys, delete only `ezkl/pk.key`. The next `prove.py` will recreate it.

## What the verifier sees

| Public | Private (prover only) |
| --- | --- |
| `C` = SHA-256 of `mlp.onnx` | loan JSON |
| Poseidon hash of the weights | `mlp.onnx` / weights |
| risk tier + probability | `data.csv` |
| SNARK proof | proving key |

## Risk tiers

- **Low** — default probability under 0.20
- **Medium** — under 0.40
- **High** — 0.40 or above

## Known limits

- Feature engineering (pandas, scaling, one-hot) happens in Python **outside** the circuit. The proof attests the **MLP forward pass**, not “this JSON was a valid loan application.”
- Trees (XGBoost) are not proved.
- `ezkl/kzg.srs` is local demo math, not a secret, but a proof only verifies against the SRS it was built with.

Beginner map of public vs private: [`docs/zk-flow.md`](docs/zk-flow.md).
