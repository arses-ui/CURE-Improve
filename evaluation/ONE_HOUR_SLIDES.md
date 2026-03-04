# One-Hour Slide Plan

## Slide 1 — Problem and Fix

Title: **Sequential CURE Interference and Our Fix**

Key points:
- Base CURE applies sequential edits that can accumulate cross-term interference.
- This manifests primarily as collateral degradation on untargeted generations.
- Our `cure_seq` orthogonalization enforces cleaner projector composition.

Visual:
- Use: `outputs/slide_pack/overview.png`

---

## Slide 2 — Immediate Qualitative Results

Title: **Before/After Sequential Erasure (Current Run)**

Use 2-3 panels:
- `outputs/slide_pack/panels/01_car.png`
- `outputs/slide_pack/panels/02_cat.png`
- `outputs/slide_pack/panels/03_dog.png`

Speaker note:
- Same prompt family, before/after comparison after sequential erasure run.
- Visual shift indicates nontrivial model behavior change post-erasure.

---

## Slide 3 — Reproducibility Upgrade

Title: **Unified Evaluation Protocol Added**

Key points:
- One runner now supports `cure`, `cure_seq`, and `cure_dit`.
- Shared config and schema for consistent cross-branch comparisons.
- Embedding aggregation is now configurable and explicit.

References:
- `evaluation/run_shared_eval.py`
- `evaluation/protocol.py`
- `evaluation/README.md`

---

## Optional 20–40 Minute Fast Re-Run (if GPU env is ready)

```bash
python evaluation/run_shared_eval.py \
  --branch cure \
  --concept-set objects10 \
  --max-concepts 3 \
  --samples-per-concept 1 \
  --steps 20 \
  --embedding-mode mean_masked

python evaluation/run_shared_eval.py \
  --branch cure_seq \
  --concept-set objects10 \
  --max-concepts 3 \
  --samples-per-concept 1 \
  --steps 20 \
  --embedding-mode mean_masked \
  --erasure-mode sequential
```

Outputs:
- `outputs/shared_eval/<run_id>/results.json`
- `outputs/shared_eval/<run_id>/images/...`
