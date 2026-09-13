# IPTC

Core implementation of **Compress the Context, Preserve the Interface:
Efficient Transformer Inpainting through Interface-Preserving Token Coarsening**.

IPTC preserves singleton tokens at the mask and its eight-connected known-side
interface, coarsens the remaining background, and reconstructs the original
grid through the retained group map. Static input routing, tail-block output
liveness, and aligned budgets turn the compact representation into an efficient
inference path. Pretrained weights and the 20-token reveal quota are unchanged.

## Results

PUT is the dense backbone reference. These measurements use the same FP32,
batch-one RTX 5090 protocol and the pretrained NaturalScene 256 checkpoint.

| Collection | Method | RGB PSNR | LPIPS | Time (ms) | Peak allocated (MiB) |
|---|---|---:|---:|---:|---:|
| Places 1500 | PUT backbone | 25.066 | 0.14835 | 184.004 | 656.949 |
| Places 1500 | IPTC | 25.123 | 0.14877 | 148.405 | 560.550 |
| COCO 600 | PUT backbone | 23.779 | 0.15185 | 184.072 | 656.768 |
| COCO 600 | IPTC | 23.847 | 0.15236 | 148.093 | 560.677 |

Five-method tables including ToMe-SD, SiTo, and ToMA adapters are in
[results/iptc](results/iptc). IPTC approaches dense reconstruction quality and
provides higher quality than the tested reduction adapters at comparable
latency. Peak memory describes the complete protocol, not coarsening alone.
The Places list is a fixed Places365-derived collection, not an official
Places2 test-split claim. Forward/reverse orders share image identities.

## Core code

- [PUT integration](third_party/PUT/image_synthesis/modeling/models/masked_image_inpainting_transformer.py)
- [Public configuration and per-image context](iptc/__init__.py)
- [Interface-aware matching](iptc/routing.py)
- [Static route reuse and output liveness](iptc/liveness.py)
- [SDPA execution](iptc/execution.py)
- [Source hashes](iptc/source_hashes.json)

The implementation is a PUT overlay with extracted inference interventions.
It uses the bundled host, codec, and official checkpoint; it is not a standalone
image model. Historical identifiers such as `dirichlet` and `PUT_SAFE_TOME_R`
remain in the implementation to preserve the evaluated configuration. The
pair score is a distance-modulated selection surrogate.

## Installation and integration

Clone this repository and install the PUT runtime dependencies described in
[docs/INSTALL.md](docs/INSTALL.md), including the official PUT checkpoint.
The recorded environment used Python 3.10, PyTorch 2.9/CUDA 12.8 and torchvision
0.24. Existing installation and checkpoint instructions remain applicable;
the **IPTC configuration below supersedes the old SBVC paper settings**.

```python
import torch
from iptc import configure_environment, inference_context

configure_environment()  # Before the bundled PUT model is constructed.
# model = ...           # Load the official PUT checkpoint using its loader.
# batch = ...           # PUT input dictionary: image, mask, relative_path.
model.eval()
seed = 20260908
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
with torch.inference_mode(), inference_context(model, "example.png", seed):
    result = model.generate_content(
        batch=batch, filter_ratio=200, filter_type="count", replicate=1,
        with_process_bar=False, mask_low_to_high=False,
        sample_largest=True, calculate_acc_and_prob=False,
        num_token_per_iter=20, accumulate_time=None, raster_order=False,
    )
```

Use FP32, CUDA, batch one, and a fresh context for each image. Keep the same
image/mask preprocessing as PUT. `configure_environment` clears inherited
`PUT_*` flags; call it in a dedicated inference process. The initial host r224
value is replaced on the first routing call by the evaluated cap256/128-aligned
per-image capacity rule. The runtime wrapper must be enabled; setting the
environment alone does not activate the complete IPTC protocol.

For end-to-end measurement, warm up first, synchronize CUDA around generation,
and separate model execution from loading, disk I/O and metric calculation.
Do not run latency tests alongside another GPU workload. This release adds
the core inference path and aggregate results; a turnkey frozen-dataset
evaluation package is not included in this update.

## Verification and scope

```bash
python tools/verify_iptc_release.py
```

This checks syntax, local module dependencies, source hashes and configuration
with the standard library. The extracted source has historical GPU evaluation
evidence; the public wrapper still requires a checkpoint-backed smoke test in
the target environment. No model inference runs as part of this verifier.
The reported acceleration applies to FP32 PUT. Low-precision and multibranch
transfer diagnostics did not reproduce that acceleration.

## Historical files and attribution

Older SBVC launchers, results, and Latent Codes integrations remain for
backward compatibility. They are not IPTC results or IPTC backbone evidence.
Use `iptc/` and `results/iptc/` for the current method.

Built on [PUT](https://github.com/liuqk3/PUT) and the bipartite matching
primitive of [ToMe](https://github.com/facebookresearch/ToMe). Third-party
code retains its existing notices and license terms; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [LICENSE.md](LICENSE.md).
No pretrained weights, private datasets, credentials, or internal research
notes are included in this update.
