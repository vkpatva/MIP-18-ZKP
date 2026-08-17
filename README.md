# Loan default risk predictor (MLP + EZKL)

A small neural net scores a mortgage application as **Low / Medium / High** default risk. A zero-knowledge proof (EZKL) can then show that the private loan was scored by a **committed** model, without sharing the loan JSON or the weight file.

XGBoost was stronger on this tabular data (test AUC about **0.88**). We use a small MLP anyway so EZKL can prove the **same** net we deploy — not a fake copy of the trees. MLP test AUC will be filled in after `python train.py`.

## Status

Cleanup is done. Training, ONNX export, and proving land in later steps. Until then:

- `data.csv` — cleaned loans (target column `Status`)
- `loan_ml.py` — feature engineering and risk labels
- `examples/new_loan.json` — sample private loan
- `train.py` / `infer.py` — still the previous XGBoost path until the MLP swap

## Planned commands

```text
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python train.py          # trains MLP, writes onnx + commitment
python infer.py          # plaintext score
python prove.py --loan examples/new_loan.json
python verify.py --proof ... --commitment ...
```

## Risk tiers

Same cutoffs as the original model:

- **Low** — default probability under 0.20
- **Medium** — under 0.40
- **High** — 0.40 or above

## Private vs public (preview)

The prover holds the loan and the weights. The verifier is meant to see only:

- `C` = hash of the committed ONNX / weights
- risk tier (and optionally the probability)
- the proof

They should not need `data.csv` or the weight file. Feature engineering still happens in Python outside the circuit (limitation, documented later).
