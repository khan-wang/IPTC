# Latent Codes-SBVC

This repository is a lightweight SBVC fork of `latent-code-inpainting`.

## Scope

- Host: `core/modules/transformer/mingpt.py`
- Insertion point: `CausalSelfAttention.forward`
- Default behavior: SBVC is off unless `LATENT_SBVC_ENABLE=1`
- Supported modes: `qkv` and `kv`
- Primary paper setting: `LATENT_SBVC_MODE=qkv`, `LATENT_SBVC_R=64`, `LATENT_SBVC_LAYER_IDS=20-39`, `LATENT_SBVC_ROUTE_CACHE=1`
- No retraining, no VQ/decoder/synthesis changes, no GA-SPG-Lite

## Runtime Switches

```bash
LATENT_SBVC_ENABLE=1
LATENT_SBVC_R=64
LATENT_SBVC_MODE=qkv
LATENT_SBVC_LAYER_IDS=20-39
LATENT_SBVC_ROUTE_CACHE=1
LATENT_SBVC_COLLECT_STATS=1
LATENT_SBVC_PROFILE=0
```

## Included Tools

- `tools/baseline_adapters/run_latent_codes_places2_fv.py`: Places2 manifest runner with SBVC stats.
- `tools/latent_codes_readonly_probe.py`: read-only topology and timing probe.
- `tools/latent_codes_sbvc_sweep.py`: small smoke/R sweep.
- `tools/latent_codes_sbvc_pareto.py`: two-stage Pareto sweep.
- `tools/summarize_latent_codes_sbvc_full_eval.py`: full-eval report builder.

## Checkpoints And Data

Checkpoints and evaluation outputs are intentionally not committed. Put Latent Codes checkpoints under:

```text
ckpts/
ckpts/Places365/
```

The expected Places256 checkpoint names are listed in `tools/baseline_adapters/run_latent_codes_places2_fv.py`.

## Full Evaluation Caveat

`tools/baseline_adapters/eval_places2_fv_outputs.py` reuses the existing PUT-SBVC evaluator helpers for exact protocol compatibility. For standalone use, set:

```bash
export PUT_SBVC_TOOLS_ROOT=/path/to/PUT-SBVC/tools
export PUT_ROOT=/path/to/official/PUT
```

Latency is not the paper claim for this host. The intended claim is quality parity with lower Transformer-core GMACs.
