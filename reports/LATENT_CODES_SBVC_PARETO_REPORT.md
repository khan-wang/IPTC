# LATENT_CODES_SBVC_PARETO_REPORT

## Scope
- Host: Latent Codes `core.modules.transformer.mingpt.CausalSelfAttention` only.
- No retraining, no GA-SPG-Lite, no VQ/Decoder/Synthesis changes.
- Latency is intentionally not optimized or claimed; this report focuses on quality and formula-based Transformer-core GMACs.
- Stage-1 subset: `50` balanced images; Stage-2 subset: `250` balanced images.
- Sampling: `sample=False`, `sampling_ratio=0.2`, route cache `True`.

## Baseline On Subset-250
| PSNR | SSIM | LPIPS | Transformer-core GMACs/img | Final Known MAE max |
|---:|---:|---:|---:|---:|
| 24.2836 | 0.843209 | 0.141260 | 17.1263 | 0.00000000 |

## Stage-1 Selected Candidates
| Config | Mode | r | Layers | GMACs Red. | PSNR Delta | LPIPS Delta | Gate |
|---|---|---:|---|---:|---:|---:|---|
| `qkv_r64_last20` | qkv | 64 | last20 | 21.80% | -0.0801 | +0.001260 | True |
| `qkv_r48_last20` | qkv | 48 | last20 | 16.93% | +0.0415 | +0.000431 | True |
| `qkv_r32_all` | qkv | 32 | all | 23.35% | +0.0759 | +0.001379 | True |
| `kv_r64_all` | kv | 64 | all | 24.90% | +0.0948 | +0.002238 | True |
| `qkv_r64_last10` | qkv | 64 | last10 | 10.90% | +0.0166 | +0.000243 | True |

## Stage-2 Pareto Confirmation
| Config | Mode | r | Layers | PSNR | SSIM | LPIPS | PSNR Delta | LPIPS Delta | GMACs Red. | Actual CR | Known MAE max | Gate |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `kv_r64_all` | kv | 64 | all | 24.1201 | 0.837822 | 0.144102 | -0.1634 | +0.002843 | 24.90% | 0.249020 | 0.00000000 | Fail |
| `qkv_r32_all` | qkv | 32 | all | 24.2438 | 0.842106 | 0.142173 | -0.0397 | +0.000913 | 23.35% | 0.124514 | 0.00000000 | Pass |
| `qkv_r64_last20` | qkv | 64 | last20 | 24.3445 | 0.845346 | 0.141018 | +0.0609 | -0.000242 | 21.80% | 0.124510 | 0.00000000 | Pass |
| `qkv_r48_last20` | qkv | 48 | last20 | 24.3548 | 0.844924 | 0.141240 | +0.0712 | -0.000019 | 16.93% | 0.093385 | 0.00000000 | Pass |
| `qkv_r64_last10` | qkv | 64 | last10 | 24.3103 | 0.844183 | 0.141199 | +0.0268 | -0.000061 | 10.90% | 0.062255 | 0.00000000 | Pass |

Note: `kv_r64_all` is not recommended despite the largest GMACs reduction because Subset-250 PSNR delta is `-0.1634dB`, exceeding the `0.1dB` quality gate.

## Recommended Balanced Config
- Recommended: `qkv_r64_last20`.
- Reason: GMACs reduction `21.80%`, LPIPS delta `-0.000242`, PSNR delta `+0.0609dB`, final known MAE max `0.00000000`.
- This is the recommended Table 1 Latent Codes plug-and-play point if the quality deltas remain acceptable under the paper protocol.

## Failure Case: Why Latency Is Not Reported
- Earlier smoke tests showed route-cache can remove most pair-selection overhead, but merge/restore tensor movement dominates runtime.
- Even when formula GMACs drop, wall-time does not reliably improve because dynamic gather/scatter and restore overhead hide the attention-matmul savings.
- Therefore latency is deliberately excluded from the claim; the valid claim is Transformer-core GMACs reduction at roughly matched reconstruction quality.

## Evidence Files
- Stage-1 summary: `outputs/latent_codes_sbvc_pareto_20260502/stage1/summary_by_config.csv`
- Stage-2 summary: `outputs/latent_codes_sbvc_pareto_20260502/stage2/summary_by_config.csv`
- Stage-1 per-image: `outputs/latent_codes_sbvc_pareto_20260502/stage1/per_image_metrics.csv`
- Stage-2 per-image: `outputs/latent_codes_sbvc_pareto_20260502/stage2/per_image_metrics.csv`
