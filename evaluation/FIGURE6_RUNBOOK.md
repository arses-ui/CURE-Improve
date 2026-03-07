# Figure-6 Script Runbook

This runbook explains how to run:
- `evaluation/paper_figure6_metrics.py`

The script reproduces a Figure-6-style sequential interference analysis with:
- `LPIPSe` (erased styles): higher = stronger target divergence
- `LPIPSu` (unerased styles): lower = better preservation
- `CLIP_u`  (unerased prompts): higher = better prompt-image alignment

## 1) Environment Setup

From repo root:

```bash
pip install -r requirements.txt lpips matplotlib
```

## 2) Quick Sanity Check (No Model Inference)

```bash
python evaluation/paper_figure6_metrics.py \
  --dry-run \
  --methods "cure,cure_seq" \
  --erased-concepts-file evaluation/artist_lists/erased_artists_50.txt \
  --unerased-artists-file evaluation/artist_lists/unerased_artists_10.txt \
  --checkpoints "1,5,10,25,50" \
  --seeds "11,22,33" \
  --max-prompts-per-group 180 \
  --output-dir outputs/figure6_eval
```

This should create `dry_run.json` and confirm your checkpoint setup before GPU spend.

## 3) Full 50-Artist Run (Paper-Style Trend)

```bash
python evaluation/paper_figure6_metrics.py \
  --methods "cure,cure_seq" \
  --erased-concepts-file evaluation/artist_lists/erased_artists_50.txt \
  --unerased-artists-file evaluation/artist_lists/unerased_artists_10.txt \
  --checkpoints "1,5,10,25,50" \
  --seeds "11,22,33" \
  --alpha 2.0 \
  --steps 20 \
  --height 384 \
  --width 384 \
  --embedding-mode mean_masked \
  --spectral-mode tikhonov \
  --max-prompts-per-group 180 \
  --cache-dir ./models \
  --output-dir outputs/figure6_eval
```

Optional:
- add `--save-images` to save checkpoint images
- switch `--spectral-mode gavish_donoho` for GD ablation

## 4) Output Structure

For each run:

`outputs/figure6_eval/<timestamp>/`
- `results.json`: all numeric outputs
- `summary.md`: compact metric table by checkpoint
- `plots/figure6_like_cure.png`
- `plots/figure6_like_cure_seq.png`
- `plots/lpips_u_comparison.png`
- `baseline/` (if generated/saved)

## 5) Interpreting Results

- `LPIPSe` rising with erased count: stronger visual change for erased styles
- `LPIPSu` staying low: less collateral drift on unerased styles
- `CLIP_u` staying high: good text-image alignment retained on unerased prompts

For Figure-6-like interference discussion, focus on trends after checkpoints `25 -> 50`.

## 6) Common Issues

- `ModuleNotFoundError: lpips`:
  - install with `pip install lpips`
- CUDA OOM:
  - reduce `--max-prompts-per-group`
  - reduce `--height/--width`
  - reduce checkpoint count temporarily
- Slow first run:
  - model/download and LPIPS backbone cache warm-up is expected
