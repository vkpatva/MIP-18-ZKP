# Loan default risk predictor (MLP + EZKL)

A small neural net scores one mortgage as **Low / Medium / High** default risk. A zero-knowledge proof (EZKL) then shows that a **private** loan was run through a **committed** MLP — without sharing the loan JSON or the weight file.

XGBoost is stronger on this table (test AUC **0.886**). We still deploy the MLP (test AUC **0.857**) so EZKL can prove the **same** net we score with, not a fake copy of the trees.

## Install

```text
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` already includes EZKL. You need `data.csv` in this folder to train (it is large). Inference and verify do not read it if the trained files are already present.

## Commands

```text
python train.py                              # MLP + ONNX + SHA-256 commitment C
python infer.py                              # plaintext score (same ONNX)
python setup_ezkl.py                         # once: circuit + keys (optional; prove.py can do this)
python prove.py --loan examples/new_loan.json
python verify.py --proof ezkl/proof.json --commitment model_commitment.txt
```

On this CPU, a fresh `python setup_ezkl.py` was about **30 seconds**, then `python prove.py` about **15 seconds**, and `python verify.py` under **1 second**. Setup also writes a proving key of about **1.4 GB** under `ezkl/pk.key` (gitignored, prover only).

## What the verifier sees

| Public | Private (prover only) |
| --- | --- |
| `C` = SHA-256 of `mlp.onnx` | loan JSON |
| Poseidon hash of the weights | `mlp.onnx` / weights |
| risk tier + probability | `data.csv` |
| SNARK proof | proving key |

`python verify.py` was run successfully **after hiding** `mlp.onnx`, `data.csv`, and `examples/new_loan.json`. It only needs the proof, `ezkl/public.json`, `ezkl/vk.key`, `ezkl/settings.json`, `ezkl/kzg.srs`, and the commitment files.

## Risk tiers

Same cutoffs as the original model:

- **Low** — default probability under 0.20
- **Medium** — under 0.40
- **High** — 0.40 or above

The sample file `examples/new_loan.json` scores as **Low Risk** (~1.4%).

## Known limits

- Feature engineering (pandas, scaling, one-hot) happens in Python **outside** the circuit. The proof attests the **MLP forward pass**, not “this JSON was a valid loan application.”
- Trees (XGBoost) are not proved. A tiny MLP is the official model because EZKL consumes ONNX nets.
- `ezkl/kzg.srs` in this repo is a **local demo SRS** (shared by prove and verify). It is public math, not a secret.

More detail in beginner language: [`docs/zk-flow.md`](docs/zk-flow.md).
