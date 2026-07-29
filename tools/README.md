# Tools

Public entry points:

- `evaluate_put.py`: PUT full-protocol or subset evaluation.
- `evaluate_latent_codes.py`: Latent Codes dense/SBVC inference.
- `evaluate_outputs.py`: PSNR, SSIM, LPIPS, and FID evaluation.

The `phase5*` and `phase7*` modules preserve the exact implementation used by
the paper runs. The public entry points delegate to those modules so the
evaluation logic remains unchanged while the user-facing commands stay stable.

All dataset, checkpoint, and output paths are command-line arguments or
environment variables. No machine-specific path is required.
