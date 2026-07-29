#!/usr/bin/env python3
"""Phase 7 Places2/NaturalScene 256x256 sanity data production.

Runs the fixed SBVC paper methods on the already-audited Places2 assets and
writes Phase 7 detailed + paper-summary CSVs under exp_data/01_main_256.
No algorithm, scorer, padding, restore, or selected-pair logic is changed here.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

import phase5a_standardized_comparison as phase5a
import phase5c_selective_finetune_pilot as phase5c
import phase5e_ga_spg_probe as phase5e
import phase5f_ga_spg_overhead_attribution as phase5f


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_GROUPS = ("20-40", "40-60", "10-60")
BUCKET_ORDER = ("10-20", "20-30", "30-40", "40-50", "50-60")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--put-root", default=str(REPO_ROOT / "third_party" / "PUT"))
    checkpoint_default = os.environ.get("PUT_CHECKPOINT", "")
    parser.add_argument(
        "--checkpoint",
        default=checkpoint_default,
        required=not checkpoint_default,
    )
    parser.add_argument(
        "--sample-manifest",
        default=str(REPO_ROOT / "runs" / "protocol" / "sample_manifest.csv"),
    )
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "runs" / "put_sanity"))
    parser.add_argument("--input-res", default="256,256")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260428)
    parser.add_argument("--samples-per-bucket", type=int, default=20)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--measure-runs", type=int, default=1)
    parser.add_argument("--num-token-per-iter", type=int, default=20)
    parser.add_argument("--num-token-for-sampling", type=int, default=200)
    parser.add_argument("--safe-tome-r", type=int, default=224)
    parser.add_argument("--global-tome-r", type=int, default=224)
    parser.add_argument("--boundary-ring-radius", type=int, default=1)
    parser.add_argument("--boundary-band-radius", type=int, default=5)
    parser.add_argument("--lite-lambda", type=float, default=0.20)
    parser.add_argument("--lite-w-texture", type=float, default=0.45)
    parser.add_argument("--lite-w-boundary", type=float, default=0.35)
    parser.add_argument("--lite-w-smoothness", type=float, default=0.20)
    parser.add_argument("--lite-w-image-border", type=float, default=0.20)
    return parser.parse_args()


def select_sanity_records(path: Path, samples_per_bucket: int) -> list[dict[str, Any]]:
    rows = phase5c.load_sample_records(path)
    selected: list[dict[str, Any]] = []
    for bucket in BUCKET_ORDER:
        bucket_rows = [row for row in rows if row["bucket"] == bucket]
        if len(bucket_rows) < samples_per_bucket:
            raise RuntimeError(f"Need {samples_per_bucket} samples for {bucket}, found {len(bucket_rows)}")
        selected.extend(bucket_rows[:samples_per_bucket])
    return selected


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def method_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    return [
        {"name": "put_baseline", "label": "PUT Baseline", "kind": "baseline", "r": 0},
        {"name": "global_tome_r224", "label": "PUT + Global ToMe-r224", "kind": "global_tome", "r": args.global_tome_r},
        {"name": "pure_distance_r224", "label": "PUT + PureDistance-r224", "kind": "pure_distance", "r": args.safe_tome_r},
        {
            "name": "sbvc_r224_zero_pad_fastpath",
            "label": "PUT + SBVC-r224-zero-pad-fastpath",
            "kind": "sbvc",
            "r": args.safe_tome_r,
        },
    ]


def env_for_method(spec: dict[str, Any], args: argparse.Namespace) -> dict[str, str]:
    env = {
        "PUT_PHASE4A_TIMING": "1",
        "PUT_PHASE5F_TIMING": "1",
        "PUT_PHASE5K_LEAN_PROFILE": "1",
        "PUT_SAFE_TOME_DEBUG": "0",
        "PUT_SIMILARITY_SCORER": "0",
        "PUT_SIMILARITY_LEARNABLE": "0",
        "PUT_SIMILARITY_GUIDANCE_SOURCE": "codec_quantized_detached",
        "PUT_GA_SPG_AUDIT": "0",
        "PUT_GASPG_DEBUG": "0",
        "PUT_GASPG_SAVE_PAIR_AUDIT": "0",
        "PUT_GASPG_SAVE_OVERLAY": "0",
        "PUT_GASPG_SAVE_ERROR_MAP": "0",
        "PUT_GA_SPG_LITE_LAMBDA": str(args.lite_lambda),
        "PUT_GA_SPG_LITE_W_TEXTURE": str(args.lite_w_texture),
        "PUT_GA_SPG_LITE_W_BOUNDARY": str(args.lite_w_boundary),
        "PUT_GA_SPG_LITE_W_SMOOTHNESS": str(args.lite_w_smoothness),
        "PUT_GA_SPG_LITE_W_IMAGE_BORDER": str(args.lite_w_image_border),
        "PUT_GA_SPG_LITE_EXPLICIT_BORDER_RISK": "0",
    }
    if spec["kind"] == "baseline":
        env.update(
            {
                "PUT_GLOBAL_TOME_R": "0",
                "PUT_BOUNDARY_SPLIT": "0",
                "PUT_SAFE_TOME_R": "0",
                "PUT_SAFE_TOME_SCORE_MODE": "similarity",
                "PUT_GA_SPG_LITE_OPTIMIZED": "0",
                "PUT_GA_SPG_LITE_RUNTIME_CACHE": "0",
                "PUT_GA_SPG_LITE_PAD_MODE": "zero",
            }
        )
    elif spec["kind"] == "global_tome":
        env.update(
            {
                "PUT_GLOBAL_TOME_R": str(spec["r"]),
                "PUT_BOUNDARY_SPLIT": "0",
                "PUT_SAFE_TOME_R": "0",
                "PUT_SAFE_TOME_SCORE_MODE": "similarity",
                "PUT_GA_SPG_LITE_OPTIMIZED": "0",
                "PUT_GA_SPG_LITE_RUNTIME_CACHE": "0",
                "PUT_GA_SPG_LITE_PAD_MODE": "zero",
            }
        )
    elif spec["kind"] == "pure_distance":
        env.update(
            {
                "PUT_GLOBAL_TOME_R": "0",
                "PUT_BOUNDARY_SPLIT": "1",
                "PUT_BOUNDARY_RING_RADIUS": str(args.boundary_ring_radius),
                "PUT_SAFE_TOME_R": str(spec["r"]),
                "PUT_SAFE_TOME_SCORE_MODE": "distance",
                "PUT_GA_SPG_LITE_OPTIMIZED": "0",
                "PUT_GA_SPG_LITE_RUNTIME_CACHE": "0",
                "PUT_GA_SPG_LITE_PAD_MODE": "zero",
            }
        )
    elif spec["kind"] == "sbvc":
        env.update(
            {
                "PUT_GLOBAL_TOME_R": "0",
                "PUT_BOUNDARY_SPLIT": "1",
                "PUT_BOUNDARY_RING_RADIUS": str(args.boundary_ring_radius),
                "PUT_SAFE_TOME_R": str(spec["r"]),
                "PUT_SAFE_TOME_SCORE_MODE": "ga_spg_lite",
                "PUT_GA_SPG_LITE_OPTIMIZED": "1",
                "PUT_GA_SPG_LITE_RUNTIME_CACHE": "1",
                "PUT_GA_SPG_LITE_PAD_MODE": "zero",
            }
        )
    else:
        raise ValueError(spec["kind"])
    return env


def route_profile(record: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    if spec["kind"] == "global_tome":
        profile = record.get("global_tome_profile") or {}
        original = int(profile.get("original_tokens", 1024))
        removed = int(profile.get("removed_tokens", 0))
        compressed = int(profile.get("merged_tokens", original - removed))
        return {
            "input_tokens": original,
            "safe_tokens": 0,
            "protect_tokens": 0,
            "eligible_tokens": original,
            "actual_removed_tokens": removed,
            "compressed_tokens": compressed,
            "restored_tokens": int(profile.get("restored_tokens", original)),
            "actual_compression_ratio": float(removed / max(original, 1)),
            "under_compression": bool(removed < int(spec["r"])),
            "bad_restore": 0,
            "protect_overlap": 0,
        }
    safe = record.get("safe_tome_profile") or {}
    if spec["kind"] == "baseline":
        return {
            "input_tokens": 1024,
            "safe_tokens": 0,
            "protect_tokens": 0,
            "eligible_tokens": 0,
            "actual_removed_tokens": 0,
            "compressed_tokens": 1024,
            "restored_tokens": 1024,
            "actual_compression_ratio": 0.0,
            "under_compression": False,
            "bad_restore": 0,
            "protect_overlap": 0,
        }
    return {
        "input_tokens": int(safe.get("original_tokens", 1024)),
        "safe_tokens": int(safe.get("safe_tokens", 0)),
        "protect_tokens": int(safe.get("protect_tokens", 0)),
        "eligible_tokens": int(safe.get("eligible_tokens", 0)),
        "actual_removed_tokens": int(safe.get("removed_tokens", 0)),
        "compressed_tokens": int(safe.get("compressed_tokens", safe.get("merged_tokens", 0))),
        "restored_tokens": int(safe.get("restored_tokens", 0)),
        "actual_compression_ratio": float(safe.get("actual_compression_ratio", 0.0)),
        "under_compression": bool(safe.get("under_compression", False)),
        "bad_restore": int(not bool(safe.get("restore_alignment_ok", True))),
        "protect_overlap": int(safe.get("protect_overlap_tokens", 0)),
    }


def core_gmacs(record: dict[str, Any], spec: dict[str, Any], dim: int, hidden_dim: int, num_layers: int) -> float:
    profile = record.get("phase4a_timing_profile") or {}
    steps = int(profile.get("num_sampling_steps", 0))
    route = route_profile(record, spec)
    tokens = int(route["compressed_tokens"] or route["input_tokens"])
    return float(phase5a.transformer_core_macs(tokens, dim=dim, hidden_dim=hidden_dim, num_layers=num_layers) * steps / 1e9)


def group_for_sample(sample: dict[str, Any]) -> list[str]:
    groups = [sample["report_group"], "10-60"]
    return groups


def summarize_rows(rows: list[dict[str, Any]], group_key: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row[group_key], row["method"])].append(row)
    out = []
    numeric_fields = [
        "psnr",
        "ssim",
        "lpips",
        "masked_psnr",
        "masked_ssim",
        "masked_lpips",
        "boundary_psnr",
        "boundary_ssim",
        "boundary_lpips",
        "core_gmacs",
        "params_m",
        "latency_sec",
        "cuda_total_ms",
        "peak_memory_mb",
        "actual_removed_tokens",
        "actual_compression_ratio",
        "under_compression",
        "bad_restore",
        "protect_overlap",
    ]
    for (group, method), items in sorted(grouped.items()):
        row = {group_key: group, "method": method, "num_images": len(items)}
        for field in numeric_fields:
            values = [float(item[field]) for item in items if field in item and str(item[field]) != ""]
            row[f"{field}_mean"] = float(np.mean(values)) if values else 0.0
            row[f"{field}_std"] = float(np.std(values, ddof=0)) if values else 0.0
        row["fid_status"] = "pilot_not_computed_lt_1k"
        out.append(row)
    return out


def latex_from_rows(rows: list[dict[str, Any]]) -> str:
    lines = [
        r"\\begin{tabular}{lrrrrrr}",
        r"Method & PSNR & SSIM & LPIPS & GMACs & Latency(s) & Actual CR \\",
        r"\\hline",
    ]
    for row in rows:
        if row.get("mask_group") != "10-60":
            continue
        lines.append(
            "{method} & {psnr:.4f} & {ssim:.4f} & {lpips:.6f} & {gmacs:.3f} & {lat:.4f} & {cr:.6f} \\\\".format(
                method=row["method"],
                psnr=float(row["psnr_mean"]),
                ssim=float(row["ssim_mean"]),
                lpips=float(row["lpips_mean"]),
                gmacs=float(row["core_gmacs_mean"]),
                lat=float(row["latency_sec_mean"]),
                cr=float(row["actual_compression_ratio_mean"]),
            )
        )
    lines.append(r"\\end{tabular}")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    put_root = Path(args.put_root).resolve()
    checkpoint = str(Path(args.checkpoint).resolve())
    input_res = phase5a.parse_hw(args.input_res)
    sample_records = select_sanity_records(Path(args.sample_manifest), args.samples_per_bucket)
    sample_map = phase5a.build_sample_group_map(sample_records)

    phase5a.set_seed(args.seed)
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    sys.path.insert(0, str(put_root))
    os.chdir(put_root)
    from scripts.inference import ImagePathDataset
    import lpips

    subset_image_dir = Path(sample_records[0]["resized_image_path"]).parent
    subset_mask_dir = Path(sample_records[0]["resized_mask_path"]).parent
    base_dataset = ImagePathDataset(str(subset_image_dir), str(subset_mask_dir), size=input_res)
    dataset = phase5e.FilteredDataset(base_dataset, [record["sample_id"] for record in sample_records])

    config = {
        "phase": "Phase 7 Places2/NaturalScene 256 sanity",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "seed": int(args.seed),
        "put_root": str(put_root),
        "checkpoint": checkpoint,
        "sample_manifest": str(Path(args.sample_manifest).resolve()),
        "num_images": len(sample_records),
        "samples_per_bucket": int(args.samples_per_bucket),
        "input_res": list(input_res),
        "warmup_runs": int(args.warmup_runs),
        "measure_runs": int(args.measure_runs),
        "num_token_per_iter": int(args.num_token_per_iter),
        "num_token_for_sampling": int(args.num_token_for_sampling),
        "methods": method_specs(args),
        "fid_policy": "not computed for sanity sample count < 1000",
    }
    (output_dir / "CONFIG.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lpips_model = lpips.LPIPS(net="vgg", spatial=True).to(device)
    high_texture_ids: set[str] = set()
    per_image_metrics = []
    per_image_compression = []
    per_image_timing = []
    outputs_manifest = []
    commands = []

    for spec in method_specs(args):
        env = env_for_method(spec, args)
        phase5f.apply_put_env(env)
        phase5a.set_seed(args.seed)
        model = phase5a.load_model(put_root, checkpoint, device)
        params_m = float(sum(parameter.numel() for parameter in model.parameters()) / 1e6)
        commands.append(" ".join([f"{key}={value}" for key, value in sorted(env.items()) if key.startswith("PUT_")]))

        for _ in range(args.warmup_runs):
            _records, _meta = phase5f.run_dataset_pass_phase5f(
                model=model,
                dataset=dataset,
                device=device,
                args=args,
                debug_artifact_dir=None,
                collect_debug=False,
                write_debug_side_effects=False,
                save_pair_audit=False,
                save_overlay=False,
                save_error_map=False,
            )
            torch.cuda.synchronize(device)

        repeat_summaries = []
        timing_records = []
        for _ in range(args.measure_runs):
            phase5a.set_seed(args.seed)
            torch.cuda.synchronize(device)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(args.gpu)
            current_records, pass_meta = phase5f.run_dataset_pass_phase5f(
                model=model,
                dataset=dataset,
                device=device,
                args=args,
                debug_artifact_dir=None,
                collect_debug=False,
                write_debug_side_effects=False,
                save_pair_audit=False,
                save_overlay=False,
                save_error_map=False,
            )
            timing_records = current_records
            repeat_summaries.append(phase5f.summarize_phase5f_repeat(current_records, pass_meta, args.gpu))
        timing_summary = phase5f.summarize_variant_repeats(repeat_summaries)

        variant_dir = output_dir / spec["name"]
        completed_dir = variant_dir / "completed_single"
        masked_dir = variant_dir / "masked_gt"
        completed_dir.mkdir(parents=True, exist_ok=True)
        masked_dir.mkdir(parents=True, exist_ok=True)
        phase5a.set_seed(args.seed)
        artifact_records = phase5e.run_dataset_pass_extended(
            model=model,
            dataset=dataset,
            device=device,
            args=args,
            save_outputs=True,
            completed_dir=completed_dir,
            masked_dir=masked_dir,
            collect_debug=False,
        )
        dim = int(model.dim)
        hidden_dim = int(model.blocks[0].mlp.fc1.out_features)
        num_layers = int(len(model.blocks))
        del model
        torch.cuda.empty_cache()

        quality = phase5e.compute_quality_metrics(
            dataset=dataset,
            completed_dir=completed_dir,
            sample_map=sample_map,
            high_texture_ids=high_texture_ids,
            lpips_model=lpips_model,
            device=device,
            boundary_band_radius=args.boundary_band_radius,
        )
        quality_by_id = {record["relative_path"]: record for record in quality["records"]}
        artifact_by_id = {record["relative_path"]: record for record in artifact_records}
        timing_by_id = {record["relative_path"]: record for record in timing_records}

        for sample in sample_records:
            sample_id = sample["sample_id"]
            q = quality_by_id[sample_id]
            artifact = artifact_by_id[sample_id]
            timing = timing_by_id.get(sample_id, artifact)
            route = route_profile(artifact, spec)
            profile = timing.get("phase5f_timing_profile") or {}
            timings_ms = profile.get("timings_ms") or {}
            cuda_total_ms = float(timings_ms.get("total_inference_time", 0.0))
            gmacs = core_gmacs(artifact, spec, dim=dim, hidden_dim=hidden_dim, num_layers=num_layers)
            base = {
                "image_id": sample_id,
                "dataset": "Places2/NaturalScene",
                "mask_ratio": float(sample["mask_ratio"]),
                "mask_ratio_bin": sample["report_group"],
                "mask_bucket": sample["bucket"],
                "method": spec["label"],
            }
            per_image_metrics.append(
                {
                    **base,
                    "psnr": q["full_psnr"],
                    "ssim": q["full_ssim"],
                    "lpips": q["full_lpips"],
                    "masked_psnr": q["masked_psnr"],
                    "masked_ssim": q["masked_ssim"],
                    "masked_lpips": q["masked_lpips"],
                    "boundary_psnr": q["boundary_psnr"],
                    "boundary_ssim": q["boundary_ssim"],
                    "boundary_lpips": q["boundary_lpips"],
                    "core_gmacs": gmacs,
                    "params_m": params_m,
                    "latency_sec": float(timing["wall_time_sec"]),
                    "cuda_total_ms": cuda_total_ms,
                    "peak_memory_mb": float(timing_summary["peak_memory_allocated_mb"]["mean"]),
                    "actual_removed_tokens": route["actual_removed_tokens"],
                    "actual_compression_ratio": route["actual_compression_ratio"],
                    "under_compression": int(route["under_compression"]),
                    "bad_restore": int(route["bad_restore"]),
                    "protect_overlap": int(route["protect_overlap"]),
                    "fid_status": "pilot_not_computed_lt_1k",
                }
            )
            per_image_compression.append({**base, **route})
            per_image_timing.append(
                {
                    **base,
                    "latency_sec": float(timing["wall_time_sec"]),
                    "cuda_total_ms": cuda_total_ms,
                    "transformer_blocks_ms": float(timings_ms.get("transformer_blocks_time", 0.0)),
                    "matching_ms": float(timings_ms.get("topk_matching_time", 0.0)),
                    "merge_ms": float(timings_ms.get("merge_time", 0.0)),
                    "restore_ms": float(timings_ms.get("restore_time", 0.0)),
                    "token_risk_compute_ms": float(timings_ms.get("token_risk_compute_time", 0.0)),
                    "peak_memory_mb": float(timing_summary["peak_memory_allocated_mb"]["mean"]),
                }
            )
            outputs_manifest.append(
                {
                    **base,
                    "input_path": sample["resized_image_path"],
                    "mask_path": sample["resized_mask_path"],
                    "gt_path": sample["resized_image_path"],
                    "completed_output_path": str((completed_dir / sample_id).resolve()),
                    "masked_gt_path": str((masked_dir / sample_id).resolve()),
                }
            )

    del lpips_model
    torch.cuda.empty_cache()

    write_csv(output_dir / "per_image_metrics.csv", per_image_metrics)
    write_csv(output_dir / "per_image_compression.csv", per_image_compression)
    write_csv(output_dir / "per_image_timing.csv", per_image_timing)
    write_csv(output_dir / "per_image_outputs_manifest.csv", outputs_manifest)

    summary_by_dataset = summarize_rows(per_image_metrics, "dataset")
    expanded_for_groups = []
    for row in per_image_metrics:
        sample = sample_map[row["image_id"]]
        for group in group_for_sample(sample):
            expanded = dict(row)
            expanded["mask_group"] = group
            expanded_for_groups.append(expanded)
    summary_by_mask = summarize_rows(expanded_for_groups, "mask_group")
    summary_main = [row for row in summary_by_mask if row["mask_group"] in REPORT_GROUPS]
    write_csv(output_dir / "summary_by_dataset.csv", summary_by_dataset)
    write_csv(output_dir / "summary_by_mask_ratio.csv", summary_by_mask)
    write_csv(output_dir / "summary_main_table.csv", summary_main)
    (output_dir / "summary_main_table_latex.txt").write_text(latex_from_rows(summary_main), encoding="utf-8")

    (output_dir / "COMMANDS.md").write_text(
        "# Commands\n\n"
        f"- `PYTHONPATH={REPO_ROOT / 'tools'}:{REPO_ROOT / 'third_party' / 'PUT'} "
        f"{sys.executable} tools/phase7_main256_places2_sanity.py "
        f"--put-root {REPO_ROOT / 'third_party' / 'PUT'} "
        f"--output-dir {output_dir} "
        "--samples-per-bucket 20 --warmup-runs 1 --measure-runs 1 --gpu 0`\n\n"
        "## Method Environments\n\n"
        + "\n".join(f"- `{command}`" for command in commands)
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(
        "# Phase 7 Main 256 Sanity\n\n"
        "Places2/NaturalScene 256x256 sanity run for SBVC paper data production. "
        "This is a 100-image pilot, not the final 5k/10k paper table.\n",
        encoding="utf-8",
    )
    metric_lines = [
        "# Metrics",
        "",
        "- Dataset: `Places2/NaturalScene`",
        f"- Images: `{len(sample_records)}`",
        "- FID: `pilot_not_computed_lt_1k`",
        "",
        "| Mask Group | Method | PSNR | SSIM | LPIPS | GMACs | Latency(s) | Actual CR | bad_restore | protect_overlap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_main:
        metric_lines.append(
            "| {group} | {method} | {psnr:.4f} | {ssim:.4f} | {lpips:.6f} | {gmacs:.3f} | {lat:.4f} | {cr:.6f} | {bad:.3f} | {protect:.3f} |".format(
                group=row["mask_group"],
                method=row["method"],
                psnr=float(row["psnr_mean"]),
                ssim=float(row["ssim_mean"]),
                lpips=float(row["lpips_mean"]),
                gmacs=float(row["core_gmacs_mean"]),
                lat=float(row["latency_sec_mean"]),
                cr=float(row["actual_compression_ratio_mean"]),
                bad=float(row["bad_restore_mean"]),
                protect=float(row["protect_overlap_mean"]),
            )
        )
    (output_dir / "metrics.md").write_text("\n".join(metric_lines) + "\n", encoding="utf-8")

    summary = {
        "config": config,
        "artifacts": {
            "per_image_metrics": str((output_dir / "per_image_metrics.csv").resolve()),
            "per_image_compression": str((output_dir / "per_image_compression.csv").resolve()),
            "per_image_timing": str((output_dir / "per_image_timing.csv").resolve()),
            "per_image_outputs_manifest": str((output_dir / "per_image_outputs_manifest.csv").resolve()),
            "summary_main_table": str((output_dir / "summary_main_table.csv").resolve()),
            "metrics_md": str((output_dir / "metrics.md").resolve()),
        },
    }
    (output_dir / "phase7_main256_sanity_summary.json").write_text(
        json.dumps(phase5a.to_serializable(summary), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "ok", "output_dir": str(output_dir), "num_images": len(sample_records)}, indent=2))


if __name__ == "__main__":
    main()
