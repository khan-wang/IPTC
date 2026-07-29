# Installation

## Requirements

The reported experiments used:

- Linux
- Python 3.10.19
- PyTorch 2.9.0+cu128
- torchvision 0.24.0+cu128
- CUDA 12.8
- NVIDIA RTX 5090

The upstream repositories were originally developed against older software.
Use the versions above for the paper environment and apply only the
compatibility changes already present in this repository.

## Environment

Create an isolated environment:

```bash
conda create -n sbvc python=3.10 -y
conda activate sbvc
```

Install the CUDA-enabled PyTorch and torchvision builds appropriate for the
target machine, then install the shared runtime dependencies:

```bash
python -m pip install torch==2.9.0+cu128 torchvision==0.24.0+cu128 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-runtime.txt
```

The requirement files inside `third_party/` are retained from the upstream
projects for provenance. Do not install them into the paper environment: the
PUT file pins `torchvision==0.8.2`, NumPy 1.19, and other packages that are
incompatible with the reported RTX 5090 environment.

## Offline Installation

For a compute node without external network access, prepare all artifacts on a
networked Linux machine that matches the target architecture and Python 3.10.
Do not build the wheelhouse on Windows because pip would select Windows wheels.

```bash
mkdir -p wheelhouse
python -m pip download --dest wheelhouse \
  torch==2.9.0+cu128 torchvision==0.24.0+cu128 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip download --dest wheelhouse -r requirements-runtime.txt
tar -czf sbvc-wheelhouse-linux-py310.tar.gz wheelhouse
```

Transfer the repository, wheelhouse archive, checkpoints, and data to the
compute node through the approved local channel. Install without contacting a
package index:

```bash
tar -xzf sbvc-wheelhouse-linux-py310.tar.gz
python -m pip install --no-index --find-links wheelhouse \
  torch==2.9.0+cu128 torchvision==0.24.0+cu128
python -m pip install --no-index --find-links wheelhouse \
  -r requirements-runtime.txt
```

If the target environment already contains the reported CUDA-enabled PyTorch
stack, keep it and install only missing packages from the wheelhouse.

## PUT Checkpoints

Download the official PUT TPAMI 2024 NaturalScene checkpoints. The paper uses
the 256-resolution UQ-Transformer checkpoint. Pass its absolute path with
`--checkpoint` or set:

```bash
export PUT_CHECKPOINT=/absolute/path/to/checkpoint.pth
```

## Latent Codes Checkpoints

Download the official Places365 checkpoints and place them under:

```text
third_party/baselines/latent-code-inpainting/ckpts/Places365/
```

Expected filenames:

```text
places256_decoder.ckpt
places256_partialencoder.ckpt
places256_transformer.ckpt
places256_unet.ckpt
places256_vqgan1024_BASE.ckpt
```

The scripts also accept the same files directly under `ckpts/`.

## Import Paths

PUT:

```bash
export PYTHONPATH="$PWD/tools:$PWD/third_party/PUT"
```

Latent Codes:

```bash
export PYTHONPATH="$PWD/tools:$PWD/third_party/baselines/latent-code-inpainting"
```
