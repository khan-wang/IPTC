# Installation

## Recorded environment

The published aggregate measurements were produced in the following
environment:

- Linux
- Python 3.10.19
- PyTorch 2.9.0 with CUDA 12.8
- torchvision 0.24.0 with CUDA 12.8
- NVIDIA RTX 5090

The public release is an inference overlay for PUT. It does not include the
PUT checkpoint or either evaluation dataset.

## Environment setup

Create an isolated Python environment and install a CUDA-matched PyTorch build
first. The exact CUDA wheel depends on the target driver, so verify the
PyTorch installation before installing the shared runtime dependencies.

```bash
conda create -n iptc python=3.10 -y
conda activate iptc

# Install the CUDA-matched torch and torchvision builds for the target host.
python -m pip install -r requirements-runtime.txt
```

The repository records the reported dependency versions in
`requirements-runtime.txt`. The upstream requirement files under
`third_party/` are retained for provenance and should not be installed on top
of the reported paper environment; some pin substantially older PyTorch,
torchvision, and NumPy versions.

## PUT checkpoint

Obtain the official PUT TPAMI 2024 NaturalScene checkpoint through the
upstream project and keep it outside this repository. The evaluated route uses
the 256-resolution UQ-Transformer checkpoint. The loader should receive an
absolute checkpoint path, for example:

```bash
export PUT_CHECKPOINT=/absolute/path/to/put_naturalscene_256_checkpoint.pth
```

Checkpoint redistribution is outside the scope of this repository.

## Import path and verification

When running the bundled PUT integration, expose the repository and PUT source
directories to Python:

```bash
export PYTHONPATH="$PWD:$PWD/third_party/PUT"
python tools/verify_iptc_release.py
```

The verifier is intentionally checkpoint-free and CPU-only. For a model-backed
run, call `configure_environment()` before constructing the PUT model and use
`inference_context(...)` for each image. See the root README and
`docs/REPRODUCE.md` for the execution and measurement protocol.


