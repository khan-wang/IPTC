# Code Provenance

## Unified Repository

This repository combines two previously separate SBVC host implementations.

| Component | Source | Reference |
|---|---|---|
| Latent Codes host | `khan-wang/Latent-Codes-SBVC` | original repository commit `37b42538356c07bebb1cf79cc4e7913b0e4307a3` |
| PUT host and paper evaluation tools | `khan-wang/PUT-SBVC` | branch `codex/phase7-paper-eval-tooling`, commit `903bb2a5429a3a891398bb45a52d7229eaae7560` |
| Latent Codes upstream | `nintendops/latent-code-inpainting` | audited upstream HEAD `e6b49c9375bbb53db2e7f3a1afaa853ff85caad8` |
| PUT upstream | `liuqk3/PUT` | baseline snapshot documented as `30df86e76dd9c2c1c69986ed1ff4ff6b50fedc3c` |

The former Latent Codes repository was not a GitHub fork. It was a manual
source import with one local commit, which is why its root README and GIF still
described the upstream Latent Codes paper. The upstream README and media are
retained only inside `third_party/baselines/latent-code-inpainting/`.

## Paper Implementation Files

PUT integration:

```text
third_party/PUT/image_synthesis/modeling/models/masked_image_inpainting_transformer.py
```

Latent Codes integration:

```text
third_party/baselines/latent-code-inpainting/core/modules/transformer/mingpt.py
```

Paper protocol tools:

```text
tools/phase7_main256_rebinned_protocol_eval.py
tools/baseline_adapters/run_latent_codes_places2_fv.py
tools/baseline_adapters/eval_places2_fv_outputs.py
```

The unified repository also incorporates the paper-relevant uncommitted state
audited on the 5090 workstation on 2026-07-29: the Latent Codes route-mode
controls, CelebA-HQ checkpoint resolution, and the PUT structure ablations.
Machine-specific launch paths, logs, generated outputs, and internal report
builders were excluded.

## Asset Provenance

`assets/method_overview.png` is a raster export of the final manuscript
`fig1_sbvc_framework_v11.pdf`.

`assets/qualitative_comparison.png` is the manuscript SBVC qualitative figure.
The GIF under the third-party Latent Codes directory is an upstream Latent
Codes demo and is not presented as an SBVC result.
