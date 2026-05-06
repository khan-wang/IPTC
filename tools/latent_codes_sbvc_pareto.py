#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = REPO_ROOT / "tools"
DEFAULT_MANIFEST = REPO_ROOT / "outputs" / "places2_fv_reproduction" / "rebinned_full_11997" / "LatentCodes" / "selected_manifest.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "latent_codes_sbvc_pareto_20260502"

for import_path in (TOOLS_ROOT,):
    import_str = str(import_path)
    if import_str not in sys.path:
        sys.path.insert(0, import_str)

import latent_codes_readonly_probe as probe  # noqa: E402


LAYER_MODE_TO_IDS = {
    "last5": "35-39",
    "last10": "30-39",
    "last20": "20-39",
    "all": "all",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-stage Latent Codes-SBVC Pareto sweep.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--config", default=str(probe.latent_adapter.DEFAULT_CONFIG))
    parser.add_argument("--stage1-size", type=int, default=50)
    parser.add_argument("--stage2-size", type=int, default=250)
    parser.add_argument("--modes", default="qkv,kv")
    parser.add_argument("--r-values", default="8,16,32,48,64")
    parser.add_argument("--layer-modes", default="last5,last10,last20,all")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260502)
    parser.add_argument("--sampling-ratio", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--temperature-degradation", type=float, default=0.9)
    parser.add_argument("--clamp-ratio", type=float, default=0.25)
    parser.add_argument("--sample", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--route-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--psnr-drop-threshold", type=float, default=0.1)
    parser.add_argument("--lpips-rise-threshold", type=float, default=0.005)
    parser.add_argument("--balanced-min-reduction", type=float, default=15.0)
    parser.add_argument("--balanced-max-reduction", type=float, default=25.0)
    parser.add_argument("--balanced-max-lpips-delta", type=float, default=0.003)
    parser.add_argument("--skip-stage2", action="store_true", default=False)
    return parser.parse_args()


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


@contextmanager
def patched_env(values: dict[str, str]):
    old_values = {key: os.environ.get(key) for key in values}
    try:
        for key, value in values.items():
            os.environ[key] = value
        yield
    finally:
        for key, old in old_values.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


def select_balanced_rows(rows: list[dict[str, str]], size: int, seed: int) -> list[dict[str, str]]:
    by_bucket: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_bucket.setdefault(row["bucket"], []).append(row)
    buckets = sorted(by_bucket)
    base = size // len(buckets)
    remainder = size % len(buckets)
    rng = np.random.default_rng(seed)
    selected: list[dict[str, str]] = []
    for bucket_index, bucket in enumerate(buckets):
        bucket_rows = by_bucket[bucket]
        take = base + (1 if bucket_index < remainder else 0)
        take = min(take, len(bucket_rows))
        indices = rng.choice(len(bucket_rows), size=take, replace=False)
        selected.extend(bucket_rows[int(index)] for index in sorted(indices))
    selected.sort(key=lambda row: row["sample_id"])
    return selected[:size]


def write_manifest(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_configs(args: argparse.Namespace) -> list[dict[str, Any]]:
    configs = []
    for mode in parse_csv_list(args.modes):
        for r in parse_int_list(args.r_values):
            for layer_mode in parse_csv_list(args.layer_modes):
                configs.append(
                    {
                        "config_id": f"{mode}_r{r}_{layer_mode}",
                        "mode": mode,
                        "r": int(r),
                        "layer_mode": layer_mode,
                        "layer_ids": LAYER_MODE_TO_IDS[layer_mode],
                    }
                )
    return configs


def to_01(tensor: torch.Tensor) -> torch.Tensor:
    return ((tensor.detach().float() + 1.0) * 0.5).clamp(0.0, 1.0)


def to_255(tensor: torch.Tensor) -> torch.Tensor:
    return (to_01(tensor) * 255.0).round().clamp(0.0, 255.0)


def final_composite(x: torch.Tensor, rec: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return x * mask + rec * (1.0 - mask)


def known_mae(final: torch.Tensor, x: torch.Tensor, mask: torch.Tensor) -> float:
    known = (mask > 0.5).expand_as(final)
    if not bool(known.any()):
        return float("nan")
    return float((to_01(final) - to_01(x)).abs()[known].mean().item())


def compute_quality(final: torch.Tensor, x: torch.Tensor, lpips_model, device: torch.device) -> dict[str, float]:
    gt_255 = to_255(x)
    pred_255 = to_255(final)
    gt_np = gt_255.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    pred_np = pred_255.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    with torch.no_grad():
        lpips_value = float(lpips_model(x.to(device), final.to(device)).mean().detach().cpu())
    return {
        "psnr": float(peak_signal_noise_ratio(gt_np, pred_np, data_range=255)),
        "ssim": float(structural_similarity(gt_np, pred_np, channel_axis=-1, data_range=255, win_size=51)),
        "lpips": lpips_value,
    }


def stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": float("nan"), "std": float("nan")}
    return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=0))}


def gmacs_from_stats(stats_rows: list[dict[str, Any]], head_dim: int) -> dict[str, float | int]:
    before_score = int(sum(int(row["score_elements_before"]) for row in stats_rows))
    after_score = int(sum(int(row["score_elements_after"]) for row in stats_rows))
    original_tokens = int(sum(int(row["original_tokens"]) for row in stats_rows))
    removed_tokens = int(sum(int(row["removed_tokens"]) for row in stats_rows))
    before_gmacs = before_score * head_dim / 1.0e9
    after_gmacs = after_score * head_dim / 1.0e9
    return {
        "attention_calls": len(stats_rows),
        "enabled_calls": int(sum(1 for row in stats_rows if row.get("enabled"))),
        "score_elements_before": before_score,
        "score_elements_after": after_score,
        "core_gmacs_before": float(before_gmacs),
        "core_gmacs_after": float(after_gmacs),
        "gmacs_reduction_pct": float((1.0 - after_gmacs / before_gmacs) * 100.0) if before_gmacs > 0 else 0.0,
        "actual_removed_tokens": removed_tokens,
        "actual_compression_ratio": float(removed_tokens / original_tokens) if original_tokens > 0 else 0.0,
        "route_cache_hits": int(sum(1 for row in stats_rows if row.get("route_cache_hit"))),
    }


def run_one(
    *,
    args: argparse.Namespace,
    row: dict[str, str],
    config: dict[str, Any] | None,
    model_parts,
    device: torch.device,
    mingpt,
    lpips_model,
    head_dim: int,
) -> tuple[dict[str, Any], torch.Tensor]:
    if config is None:
        env = {
            "LATENT_SBVC_ENABLE": "0",
            "LATENT_SBVC_R": "0",
            "LATENT_SBVC_PROFILE": "1",
            "LATENT_SBVC_LAYER_IDS": "all",
            "LATENT_SBVC_ROUTE_CACHE": "0",
            "LATENT_SBVC_MODE": "qkv",
        }
        config_id = "baseline"
    else:
        env = {
            "LATENT_SBVC_ENABLE": "1",
            "LATENT_SBVC_R": str(config["r"]),
            "LATENT_SBVC_PROFILE": "1",
            "LATENT_SBVC_LAYER_IDS": str(config["layer_ids"]),
            "LATENT_SBVC_ROUTE_CACHE": "1" if args.route_cache else "0",
            "LATENT_SBVC_MODE": str(config["mode"]),
        }
        config_id = str(config["config_id"])

    with patched_env(env):
        mingpt.reset_latent_sbvc_stats()
        token_info, outputs = probe.run_pipeline(args, row, model_parts, device, timer=None)
        stats_rows = mingpt.get_latent_sbvc_stats()

    x = outputs["x"]
    mask = outputs["mask"]
    final = final_composite(x, outputs["rec"], mask)
    quality = compute_quality(final, x, lpips_model, device)
    gmacs = gmacs_from_stats(stats_rows, head_dim=head_dim)
    record = {
        "sample_id": row["sample_id"],
        "bucket": row["bucket"],
        "report_group": row["report_group"],
        "mask_ratio": float(row["mask_ratio"]),
        "config_id": config_id,
        "mode": "baseline" if config is None else config["mode"],
        "r": 0 if config is None else int(config["r"]),
        "layer_mode": "all" if config is None else config["layer_mode"],
        "layer_ids": "all" if config is None else config["layer_ids"],
        "output_shape": list(final.shape),
        "z_tokens": int(token_info["z_token_length"]),
        "transformer_tokens": int(token_info["transformer_token_length"]),
        "known_latent_tokens": int(token_info["known_latent_tokens"]),
        "unknown_latent_tokens": int(token_info["unknown_latent_tokens"]),
        "sample_iterations": int(token_info["sample_iterations"]),
        "final_known_mae": known_mae(final, x, mask),
        **quality,
        **gmacs,
    }
    return record, final


def aggregate_records(records: list[dict[str, Any]], baseline_summary: dict[str, Any] | None = None) -> dict[str, Any]:
    out = {
        "config_id": records[0]["config_id"],
        "mode": records[0]["mode"],
        "r": int(records[0]["r"]),
        "layer_mode": records[0]["layer_mode"],
        "layer_ids": records[0]["layer_ids"],
        "num_images": len(records),
        "psnr_mean": stats([float(row["psnr"]) for row in records])["mean"],
        "ssim_mean": stats([float(row["ssim"]) for row in records])["mean"],
        "lpips_mean": stats([float(row["lpips"]) for row in records])["mean"],
        "final_known_mae_max": float(max(float(row["final_known_mae"]) for row in records)),
        "core_gmacs_before_mean": stats([float(row["core_gmacs_before"]) for row in records])["mean"],
        "core_gmacs_after_mean": stats([float(row["core_gmacs_after"]) for row in records])["mean"],
        "gmacs_reduction_pct": float(
            (1.0 - sum(float(row["core_gmacs_after"]) for row in records) / sum(float(row["core_gmacs_before"]) for row in records)) * 100.0
        )
        if sum(float(row["core_gmacs_before"]) for row in records) > 0
        else 0.0,
        "actual_removed_tokens_mean": stats([float(row["actual_removed_tokens"]) for row in records])["mean"],
        "actual_compression_ratio": float(
            sum(float(row["actual_removed_tokens"]) for row in records)
            / sum(float(row["attention_calls"] * row["transformer_tokens"]) for row in records)
        )
        if sum(float(row["attention_calls"] * row["transformer_tokens"]) for row in records) > 0
        else 0.0,
        "attention_calls_mean": stats([float(row["attention_calls"]) for row in records])["mean"],
        "enabled_calls_mean": stats([float(row["enabled_calls"]) for row in records])["mean"],
    }
    if baseline_summary is not None:
        out["psnr_delta"] = float(out["psnr_mean"] - baseline_summary["psnr_mean"])
        out["ssim_delta"] = float(out["ssim_mean"] - baseline_summary["ssim_mean"])
        out["lpips_delta"] = float(out["lpips_mean"] - baseline_summary["lpips_mean"])
        out["passes_quality_gate"] = bool(out["psnr_delta"] >= -0.1 and out["lpips_delta"] <= 0.005 and out["final_known_mae_max"] == 0.0)
    else:
        out["psnr_delta"] = 0.0
        out["ssim_delta"] = 0.0
        out["lpips_delta"] = 0.0
        out["passes_quality_gate"] = True
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def pareto_front(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = [row for row in rows if row.get("passes_quality_gate")]
    front = []
    for row in candidates:
        dominated = False
        for other in candidates:
            if other is row:
                continue
            no_worse = (
                float(other["gmacs_reduction_pct"]) >= float(row["gmacs_reduction_pct"])
                and float(other["psnr_delta"]) >= float(row["psnr_delta"])
                and float(other["lpips_delta"]) <= float(row["lpips_delta"])
            )
            strictly_better = (
                float(other["gmacs_reduction_pct"]) > float(row["gmacs_reduction_pct"])
                or float(other["psnr_delta"]) > float(row["psnr_delta"])
                or float(other["lpips_delta"]) < float(row["lpips_delta"])
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            front.append(row)
    return front


def select_stage2_configs(stage1_rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    front = pareto_front(stage1_rows)
    if not front:
        front = [row for row in stage1_rows if row.get("passes_quality_gate")]
    if not front:
        front = sorted(stage1_rows, key=lambda row: (float(row["lpips_delta"]), -float(row["gmacs_reduction_pct"])))[: args.top_k]

    def rank(row: dict[str, Any]) -> tuple:
        reduction = float(row["gmacs_reduction_pct"])
        lpips_delta = float(row["lpips_delta"])
        psnr_delta = float(row["psnr_delta"])
        balanced = args.balanced_min_reduction <= reduction <= args.balanced_max_reduction and lpips_delta <= args.balanced_max_lpips_delta
        qkv_priority = row["mode"] == "qkv"
        return (
            0 if balanced else 1,
            0 if qkv_priority else 1,
            abs(reduction - 20.0),
            max(0.0, lpips_delta),
            -psnr_delta,
            -reduction,
        )

    selected = sorted(front, key=rank)[: args.top_k]
    if not any(row["mode"] == "qkv" for row in selected):
        qkv_rows = [row for row in stage1_rows if row.get("passes_quality_gate") and row["mode"] == "qkv"]
        if qkv_rows:
            selected = selected[:-1] + [sorted(qkv_rows, key=rank)[0]]
    return selected[: args.top_k]


def config_from_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "config_id": row["config_id"],
        "mode": row["mode"],
        "r": int(row["r"]),
        "layer_mode": row["layer_mode"],
        "layer_ids": row["layer_ids"],
    }


def run_variant_set(
    *,
    stage_name: str,
    rows: list[dict[str, str]],
    configs: list[dict[str, Any]],
    args: argparse.Namespace,
    output_dir: Path,
    model_parts,
    device: torch.device,
    mingpt,
    lpips_model,
    head_dim: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    stage_dir = output_dir / stage_name
    stage_dir.mkdir(parents=True, exist_ok=True)
    per_image_records: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    baseline_records = []
    baseline_dir = stage_dir / "baseline_examples"
    baseline_dir.mkdir(exist_ok=True)
    for index, row in enumerate(rows):
        record, final = run_one(
            args=args,
            row=row,
            config=None,
            model_parts=model_parts,
            device=device,
            mingpt=mingpt,
            lpips_model=lpips_model,
            head_dim=head_dim,
        )
        baseline_records.append(record)
        per_image_records.append(record)
        if index < 3:
            probe.latent_adapter.to_uint8_image(final).save(baseline_dir / row["sample_id"])
    baseline_summary = aggregate_records(baseline_records, baseline_summary=None)
    summary_rows.append(baseline_summary)
    write_csv(stage_dir / "per_image_metrics_partial.csv", per_image_records)
    write_csv(stage_dir / "summary_by_config_partial.csv", summary_rows)

    for config_index, config in enumerate(configs, start=1):
        config_records = []
        example_dir = stage_dir / "examples" / config["config_id"]
        example_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{stage_name}] {config_index}/{len(configs)} {config['config_id']}", flush=True)
        for image_index, row in enumerate(rows):
            record, final = run_one(
                args=args,
                row=row,
                config=config,
                model_parts=model_parts,
                device=device,
                mingpt=mingpt,
                lpips_model=lpips_model,
                head_dim=head_dim,
            )
            config_records.append(record)
            per_image_records.append(record)
            if image_index == 0:
                probe.latent_adapter.to_uint8_image(final).save(example_dir / row["sample_id"])
        summary = aggregate_records(config_records, baseline_summary=baseline_summary)
        summary_rows.append(summary)
        write_csv(stage_dir / "per_image_metrics_partial.csv", per_image_records)
        write_csv(stage_dir / "summary_by_config_partial.csv", summary_rows)

    write_csv(stage_dir / "per_image_metrics.csv", per_image_records)
    write_csv(stage_dir / "summary_by_config.csv", summary_rows)
    return baseline_summary, summary_rows, per_image_records


def format_float(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(numeric):
        return "nan"
    return f"{numeric:.{digits}f}"


def choose_balanced(stage2_rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    candidates = [row for row in stage2_rows if row["config_id"] != "baseline" and row.get("passes_quality_gate")]
    balanced = [
        row
        for row in candidates
        if args.balanced_min_reduction <= float(row["gmacs_reduction_pct"]) <= args.balanced_max_reduction
        and float(row["lpips_delta"]) <= args.balanced_max_lpips_delta
    ]
    pool = balanced if balanced else candidates
    if not pool:
        pool = [row for row in stage2_rows if row["config_id"] != "baseline"]
    return sorted(
        pool,
        key=lambda row: (
            abs(float(row["gmacs_reduction_pct"]) - 20.0),
            float(row["lpips_delta"]),
            -float(row["psnr_delta"]),
            0 if row["mode"] == "qkv" else 1,
        ),
    )[0]


def write_final_report(
    *,
    path: Path,
    stage1_summary: list[dict[str, Any]],
    selected_stage1: list[dict[str, Any]],
    stage2_summary: list[dict[str, Any]],
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    baseline = next(row for row in stage2_summary if row["config_id"] == "baseline")
    candidate_rows = [row for row in stage2_summary if row["config_id"] != "baseline"]
    balanced = choose_balanced(stage2_summary, args)

    lines = [
        "# LATENT_CODES_SBVC_PARETO_REPORT",
        "",
        "## Scope",
        "- Host: Latent Codes `core.modules.transformer.mingpt.CausalSelfAttention` only.",
        "- No retraining, no GA-SPG-Lite, no VQ/Decoder/Synthesis changes.",
        "- Latency is intentionally not optimized or claimed; this report focuses on quality and formula-based Transformer-core GMACs.",
        f"- Stage-1 subset: `{args.stage1_size}` balanced images; Stage-2 subset: `{args.stage2_size}` balanced images.",
        f"- Sampling: `sample={args.sample}`, `sampling_ratio={args.sampling_ratio}`, route cache `{args.route_cache}`.",
        "",
        "## Baseline On Subset-250",
        "| PSNR | SSIM | LPIPS | Transformer-core GMACs/img | Final Known MAE max |",
        "|---:|---:|---:|---:|---:|",
        f"| {baseline['psnr_mean']:.4f} | {baseline['ssim_mean']:.6f} | {baseline['lpips_mean']:.6f} | {baseline['core_gmacs_before_mean']:.4f} | {baseline['final_known_mae_max']:.8f} |",
        "",
        "## Stage-1 Selected Candidates",
        "| Config | Mode | r | Layers | GMACs Red. | PSNR Delta | LPIPS Delta | Gate |",
        "|---|---|---:|---|---:|---:|---:|---|",
    ]
    for row in selected_stage1:
        lines.append(
            f"| `{row['config_id']}` | {row['mode']} | {row['r']} | {row['layer_mode']} | "
            f"{row['gmacs_reduction_pct']:.2f}% | {row['psnr_delta']:+.4f} | {row['lpips_delta']:+.6f} | {row['passes_quality_gate']} |"
        )

    lines.extend(
        [
            "",
            "## Stage-2 Pareto Confirmation",
            "| Config | Mode | r | Layers | PSNR | SSIM | LPIPS | PSNR Delta | LPIPS Delta | GMACs Red. | Actual CR | Known MAE max | Gate |",
            "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in sorted(candidate_rows, key=lambda item: (-float(item["gmacs_reduction_pct"]), float(item["lpips_delta"]))):
        lines.append(
            f"| `{row['config_id']}` | {row['mode']} | {row['r']} | {row['layer_mode']} | "
            f"{row['psnr_mean']:.4f} | {row['ssim_mean']:.6f} | {row['lpips_mean']:.6f} | "
            f"{row['psnr_delta']:+.4f} | {row['lpips_delta']:+.6f} | "
            f"{row['gmacs_reduction_pct']:.2f}% | {row['actual_compression_ratio']:.6f} | {row['final_known_mae_max']:.8f} | "
            f"{'Pass' if row.get('passes_quality_gate') else 'Fail'} |"
        )

    lines.extend(
        [
            "",
            "Note: configurations marked `Fail` are not recommended even if GMACs reduction is high, because they exceed the Stage-2 quality gate.",
            "",
            "## Recommended Balanced Config",
            f"- Recommended: `{balanced['config_id']}`.",
            f"- Reason: GMACs reduction `{balanced['gmacs_reduction_pct']:.2f}%`, LPIPS delta `{balanced['lpips_delta']:+.6f}`, PSNR delta `{balanced['psnr_delta']:+.4f}dB`, final known MAE max `{balanced['final_known_mae_max']:.8f}`.",
            "- This is the recommended Table 1 Latent Codes plug-and-play point if the quality deltas remain acceptable under the paper protocol.",
            "",
            "## Failure Case: Why Latency Is Not Reported",
            "- Earlier smoke tests showed route-cache can remove most pair-selection overhead, but merge/restore tensor movement dominates runtime.",
            "- Even when formula GMACs drop, wall-time does not reliably improve because dynamic gather/scatter and restore overhead hide the attention-matmul savings.",
            "- Therefore latency is deliberately excluded from the claim; the valid claim is Transformer-core GMACs reduction at roughly matched reconstruction quality.",
            "",
            "## Evidence Files",
            f"- Stage-1 summary: `{output_dir / 'stage1' / 'summary_by_config.csv'}`",
            f"- Stage-2 summary: `{output_dir / 'stage2' / 'summary_by_config.csv'}`",
            f"- Stage-1 per-image: `{output_dir / 'stage1' / 'per_image_metrics.csv'}`",
            f"- Stage-2 per-image: `{output_dir / 'stage2' / 'per_image_metrics.csv'}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = probe.latent_adapter.load_manifest(Path(args.manifest).resolve())
    stage1_rows = select_balanced_rows(rows, args.stage1_size, args.seed)
    stage2_rows = select_balanced_rows(rows, args.stage2_size, args.seed + 250)
    write_manifest(stage1_rows, output_dir / "stage1_manifest.csv")
    write_manifest(stage2_rows, output_dir / "stage2_manifest.csv")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model_parts = probe.load_latent_codes_model(args, device)
    _model, _vq, _encoder, transformer, _unet, _config = model_parts
    head_dim = int(transformer.transformer.config.n_embd // transformer.transformer.config.n_head)
    import core.modules.transformer.mingpt as mingpt
    import lpips

    lpips_model = lpips.LPIPS(net="vgg").to(device).eval()
    configs = build_configs(args)

    metadata = {
        "manifest": str(Path(args.manifest).resolve()),
        "output_dir": str(output_dir),
        "stage1_size": len(stage1_rows),
        "stage2_size": len(stage2_rows),
        "configs": configs,
        "seed": args.seed,
        "sample": args.sample,
        "route_cache": args.route_cache,
        "gmacs_formula": "sum(score_elements_before_or_after * head_dim) / 1e9, where score_elements = B * heads * Nq * Nk",
    }
    (output_dir / "CONFIG.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    stage1_baseline, stage1_summary, _stage1_per_image = run_variant_set(
        stage_name="stage1",
        rows=stage1_rows,
        configs=configs,
        args=args,
        output_dir=output_dir,
        model_parts=model_parts,
        device=device,
        mingpt=mingpt,
        lpips_model=lpips_model,
        head_dim=head_dim,
    )
    selected = select_stage2_configs([row for row in stage1_summary if row["config_id"] != "baseline"], args)
    selected_configs = [config_from_summary(row) for row in selected]
    (output_dir / "stage1_selected_configs.json").write_text(json.dumps(selected, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.skip_stage2:
        print(json.dumps({"status": "stage1_only", "selected": selected}, indent=2, ensure_ascii=False))
        return

    _stage2_baseline, stage2_summary, _stage2_per_image = run_variant_set(
        stage_name="stage2",
        rows=stage2_rows,
        configs=selected_configs,
        args=args,
        output_dir=output_dir,
        model_parts=model_parts,
        device=device,
        mingpt=mingpt,
        lpips_model=lpips_model,
        head_dim=head_dim,
    )
    report_path = output_dir / "LATENT_CODES_SBVC_PARETO_REPORT.md"
    write_final_report(
        path=report_path,
        stage1_summary=stage1_summary,
        selected_stage1=selected,
        stage2_summary=stage2_summary,
        args=args,
        output_dir=output_dir,
    )
    shutil.copyfile(report_path, REPO_ROOT / "LATENT_CODES_SBVC_PARETO_REPORT.md")
    print(json.dumps({"status": "ok", "report": str(report_path), "selected": selected_configs}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
