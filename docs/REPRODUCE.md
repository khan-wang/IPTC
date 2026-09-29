# Reproduction notes

This release separates the public core implementation from the large,
licensed benchmark assets used to obtain the reported aggregates.

## Included in the release

- IPTC routing, reconstruction, liveness, and execution modules.
- The PUT integration point used by the public overlay.
- Aggregate five-identity results for Places 1500 and COCO 600 under
  `results/iptc/`.
- Source hashes and a checkpoint-free release verifier.

## Required external assets

A full benchmark reproduction additionally requires:

1. the official PUT NaturalScene 256 checkpoint;
2. the Places365-derived image collection and the COCO evaluation images;
3. the fixed image-mask manifests used for the reported Places 1500 and COCO
   600 evaluations;
4. the corresponding preprocessing and metric packages.

These assets are not redistributed. The manifests contain local paths and the
datasets and checkpoint remain subject to their original access terms. Frozen
manifests and additional per-image records can be requested from the
corresponding author for a reproducibility check.

## Static release check

Run the following command from the repository root:

```bash
python tools/verify_iptc_release.py
```

The check validates Python syntax, local module dependencies, source hashes,
and configuration. It does not load a checkpoint, access a dataset, or run a
GPU inference job.

## Model-backed smoke test

After installing PUT and obtaining the checkpoint, use the integration pattern
in the root README:

```python
import torch
from iptc import configure_environment, inference_context

configure_environment()
# Construct PUT and load the official checkpoint here.
model.eval()

with torch.inference_mode(), inference_context(model, "example.png", seed=20260908):
    output = model.generate_content(
        batch=batch,
        filter_ratio=200,
        filter_type="count",
        replicate=1,
        with_process_bar=False,
        mask_low_to_high=False,
        sample_largest=True,
        calculate_acc_and_prob=False,
        num_token_per_iter=20,
        accumulate_time=None,
        raster_order=False,
    )
```

Use the same preprocessing as PUT and create a fresh context for every image.
The public wrapper is not a standalone generative model and the static check
does not establish checkpoint-backed numerical equivalence.

## Reported measurement protocol

The aggregate CSVs use FP32, batch size one, a single RTX 5090, CUDA
synchronization around model generation, and a warm-up phase. Model loading,
disk I/O, and metric computation are excluded from latency. Forward and reverse
evaluation orders share image identities; the paired aggregate is reported in
the CSV files. The exact operating-point definitions and all columns are
preserved in `results/iptc/places1500.csv` and `results/iptc/coco600.csv`.

## Results

Use the aggregate CSVs as the public numerical record. The image-level outputs,
mask files, and frozen manifests are deliberately kept outside the repository
and can be shared under the applicable dataset terms.


