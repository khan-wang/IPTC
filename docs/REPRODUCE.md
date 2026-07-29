# Reproduction

## Data Protocol

The main protocol contains 36,500 images divided into six mask-ratio buckets:

| Bucket | Count |
|---|---:|
| 0.01-0.10 | 6,084 |
| 0.10-0.20 | 6,083 |
| 0.20-0.30 | 6,083 |
| 0.30-0.40 | 6,083 |
| 0.40-0.50 | 6,084 |
| 0.50-0.60 | 6,083 |

The fixed sample manifest is expected at:

```text
exp_data/01_main_256_places2_rebinned36500_full/sample_manifest.csv
```

The manifest is intentionally not committed because it contains
machine-specific dataset paths. Generate an equivalent manifest with the
provided protocol tools or rewrite the path columns while preserving the
sample identifiers, masks, bucket counts, and random seed.

## Required Variables

```bash
export PYTHON_BIN="$(command -v python)"
export PUT_CHECKPOINT=/absolute/path/to/put/checkpoint.pth
export SBVC_IMAGE_ROOT=/absolute/path/to/places365/val_large
export SBVC_MASK_ROOT=/absolute/path/to/testing_mask_dataset
```

## PUT Full Protocol

Run the principal PUT operating point:

```bash
PYTHONPATH="$PWD/tools:$PWD/third_party/PUT" \
"$PYTHON_BIN" tools/evaluate_put.py \
  --put-root "$PWD/third_party/PUT" \
  --checkpoint "$PUT_CHECKPOINT" \
  --image-root "$SBVC_IMAGE_ROOT" \
  --mask-root "$SBVC_MASK_ROOT" \
  --output-dir runs/put_sbvc_r224 \
  --reuse-existing-subset-dir exp_data/01_main_256_places2_rebinned36500_full \
  --dataset-name "Places2/NaturalScene 36500" \
  --validation-source-label "fixed 36500-image manifest" \
  --image-selection-mode recursive_image_root_scan \
  --mask-selection-mode deterministic_cycle \
  --count-01-10 6084 \
  --count-10-20 6083 \
  --count-20-30 6083 \
  --count-30-40 6083 \
  --count-40-50 6084 \
  --count-50-60 6083 \
  --methods sbvc_r224_zero_pad_fastpath \
  --safe-tome-r 224 \
  --seed 20260502 \
  --gpu 0 \
  --resume-existing
```

Run the PUT rate sweep:

```bash
bash scripts/launch_put_sbvc_rebinned36500_r_sweep.sh
```

## Latent Codes Full Protocol

Dense baseline:

```bash
bash scripts/launch_latent_codes_baseline_rebinned36500_full.sh
```

SBVC r64, QKV compression, layers 20-39:

```bash
bash scripts/launch_latent_codes_sbvc_rebinned36500_full.sh
```

The default paper route is `LATENT_SBVC_ROUTE_MODE=safe_similarity`.
For the matched 1,500-image mechanism subset, set
`LATENT_SBVC_ROUTE_MODE=safe_distance` for eligible-pool geometry matching or
`LATENT_SBVC_ROUTE_MODE=global_similarity` for the unconstrained similarity
control. Keep all other SBVC settings fixed.

Rate sweep:

```bash
bash scripts/launch_latent_codes_sbvc_rebinned36500_r_sweep.sh
```

## Face-Domain Protocol

The supplementary face protocol uses the PUT FFHQ checkpoint and the Latent
Codes CelebA-HQ checkpoint family. Place the latter under either `ckpts/` or
`ckpts/CelebA-HQ/` with these names:

```text
celeba_vqgan.ckpt
celeba_encoder.ckpt
celeba_decoder.ckpt
celeba_transformer.ckpt
```

Pass `configs/celeba_inpainting.yaml` through `--config` when running
`tools/evaluate_latent_codes.py`. The submitted protocol evaluates the fixed
FFHQ validation set of 1,000 images and a fixed CelebA-HQ validation set of
2,000 images, using the same six mask-ratio buckets and seeds as the main
evaluation.

## Expected Main Results

The checked reference values are stored in:

```text
results/paper_main_results.csv
```

Exact image metrics depend on the official checkpoints, fixed image/mask
manifest, and deterministic runtime settings. Validate the sample count and
manifest identity before comparing aggregate values.
