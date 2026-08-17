# How the zero-knowledge part works

This is a short map of who sees what. You do not need to know cryptography internals to use the scripts.

## Two machines, two jobs

**Prover** (holds secrets):

1. Reads the private loan JSON.
2. Turns it into numbers in Python (`loan_ml.py`: logs, ratios, scaling, one-hot).
3. Runs those numbers through `mlp.onnx` inside EZKL.
4. Builds a proof: “I ran **this committed net** on **some hidden input** and got **this probability**.”

**Verifier** (should learn almost nothing extra):

1. Gets the proof, the public JSON, the verifying key, and commitment `C`.
2. Checks the proof is valid.
3. Reads Low / Medium / High (and optionally the probability).
4. Never opens `data.csv`, the loan file, or the ONNX weights.

## Public vs private

```
private:  loan JSON, MLP weights (mlp.onnx)
public:   C = SHA-256(mlp.onnx)
          Poseidon hash of the weights (inside the SNARK)
          risk tier + probability
          proof
```

`C` is a fingerprint of the ONNX file. Change one weight, `C` changes. EZKL also hashes the weights *inside* the circuit (`hashed/public`), so a valid proof is tied to that same net, not a swapped one.

## Why an MLP and not XGBoost

XGBoost is a forest of decision trees. EZKL’s usual path is: **ONNX neural net → circuit → proof**. A one-hidden-layer MLP (59 inputs → 32 ReLU neurons → 1 sigmoid) is small enough to prove on a laptop. XGBoost still wins on this dataset (AUC 0.886 vs 0.857). We are not pretending the MLP is more accurate; we chose it so the proved graph is the **deployed** scorer.

## What is *not* in the circuit (v1)

Pandas feature code stays in Python. The ONNX file is only:

`Gemm → ReLU → Gemm → Sigmoid`

So a cheating prover could feed a made-up 59-vector that never came from a real application form. The proof still shows that **whatever** vector they used was scored by the committed MLP. Closing that gap would mean putting feature checks inside the circuit (future work).

## Files in `ezkl/`

| File | Who needs it |
| --- | --- |
| `settings.json`, `vk.key`, `kzg.srs`, `param_commitment.txt` | verifier (public) |
| `proof.json`, `public.json` | verifier (public outputs of one run) |
| `pk.key`, `model.compiled`, `input.json`, `witness.json` | prover only (gitignored except what we publish) |
