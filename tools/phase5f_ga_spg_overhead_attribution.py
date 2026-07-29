#!/usr/bin/env python3
"""Phase 5F GA-SPG overhead attribution and lean inference check."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

import phase5a_standardized_comparison as phase5a
import phase5c_selective_finetune_pilot as phase5c
import phase5e_ga_spg_probe as phase5e


PHASE5F_BUCKETS = (
    "boundary_split_time",
    "pure_distance_pool_time",
    "codec_feature_prepare_time",
    "avgpool_3x3_time",
    "avgpool_9x9_time",
    "pair_metric_base_time",
    "risk_penalty_matrix_time",
    "adjusted_metric_time",
    "topk_matching_time",
    "merge_time",
    "restore_time",
    "transformer_blocks_time",
    "total_inference_time",
)
REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--put-root", required=True)
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("PUT_CHECKPOINT", ""),
    )
    parser.add_argument(
        "--sample-manifest",
        default=str(REPO_ROOT / "runs" / "protocol" / "sample_manifest.csv"),
    )
    parser.add_argument(
        "--phase5e-dir",
        default=str(REPO_ROOT / "runs" / "ga_spg_probe"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "runs" / "ga_spg_overhead"),
    )
    parser.add_argument("--input-res", default="256,256")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260424)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--measure-runs", type=int, default=3)
    parser.add_argument("--num-token-per-iter", type=int, default=20)
    parser.add_argument("--num-token-for-sampling", type=int, default=200)
    parser.add_argument("--boundary-ring-radius", type=int, default=1)
    parser.add_argument("--boundary-band-radius", type=int, default=5)
    parser.add_argument("--ga-lambda-risk", type=float, default=0.35)
    parser.add_argument("--ga-w1", type=float, default=0.35)
    parser.add_argument("--ga-w2", type=float, default=0.35)
    parser.add_argument("--ga-w3", type=float, default=0.15)
    parser.add_argument("--ga-w4", type=float, default=0.15)
    parser.add_argument("--ga-hard-threshold", type=float, default=0.65)
    parser.add_argument("--ga-hard-penalty", type=float, default=1000000.0)
    return parser.parse_args()


def arm_specs() -> list[dict[str, Any]]:
    return [
        {
            "name": "pure_distance_r192_lean",
            "label": "PureDistance-r192-lean",
            "safe_tome_r": 192,
            "score_mode": "distance",
            "gaspg_debug": False,
            "save_pair_audit": False,
            "save_overlay": False,
            "save_error_map": False,
        },
        {
            "name": "ga_spg_soft_r192_lean",
            "label": "GA-SPG-soft-r192-lean",
            "safe_tome_r": 192,
            "score_mode": "ga_spg_soft",
            "gaspg_debug": False,
            "save_pair_audit": False,
            "save_overlay": False,
            "save_error_map": False,
        },
        {
            "name": "ga_spg_soft_r192_debug",
            "label": "GA-SPG-soft-r192-debug",
            "safe_tome_r": 192,
            "score_mode": "ga_spg_soft",
            "gaspg_debug": True,
            "save_pair_audit": True,
            "save_overlay": True,
            "save_error_map": True,
        },
    ]


def arm_env(spec: dict[str, Any], args: argparse.Namespace) -> dict[str, str]:
    return {
        "PUT_GLOBAL_TOME_R": "0",
        "PUT_PHASE4A_TIMING": "1",
        "PUT_PHASE5F_TIMING": "1",
        "PUT_BOUNDARY_SPLIT": "1",
        "PUT_BOUNDARY_RING_RADIUS": str(args.boundary_ring_radius),
        "PUT_SAFE_TOME_R": str(spec["safe_tome_r"]),
        "PUT_SAFE_TOME_SCORE_MODE": str(spec["score_mode"]),
        "PUT_SAFE_TOME_DEBUG": "1" if spec["gaspg_debug"] else "0",
        "PUT_SIMILARITY_SCORER": "0",
        "PUT_SIMILARITY_LEARNABLE": "0",
        "PUT_SIMILARITY_GUIDANCE_SOURCE": "codec_quantized_detached",
        "PUT_GA_SPG_AUDIT": "1" if spec["gaspg_debug"] else "0",
        "PUT_GASPG_DEBUG": "1" if spec["gaspg_debug"] else "0",
        "PUT_GASPG_SAVE_PAIR_AUDIT": "1" if spec["save_pair_audit"] else "0",
        "PUT_GASPG_SAVE_OVERLAY": "1" if spec["save_overlay"] else "0",
        "PUT_GASPG_SAVE_ERROR_MAP": "1" if spec["save_error_map"] else "0",
        "PUT_GA_SPG_LAMBDA_RISK": str(args.ga_lambda_risk),
        "PUT_GA_SPG_W1": str(args.ga_w1),
        "PUT_GA_SPG_W2": str(args.ga_w2),
        "PUT_GA_SPG_W3": str(args.ga_w3),
        "PUT_GA_SPG_W4": str(args.ga_w4),
        "PUT_GA_SPG_HARD_THRESHOLD": str(args.ga_hard_threshold),
        "PUT_GA_SPG_HARD_PENALTY": str(args.ga_hard_penalty),
    }


def apply_put_env(env: dict[str, str]) -> None:
    for key in list(os.environ.keys()):
        if key.startswith("PUT_"):
            os.environ.pop(key)
    os.environ.update(env)


def tensor_to_pil_rgb(image: torch.Tensor) -> Image.Image:
    arr = image.detach().cpu().clamp(0, 255).to(torch.uint8).permute(1, 2, 0).numpy()
    return Image.fromarray(arr)


def debug_pair_rows(relative_path: str, debug: dict[str, Any] | None) -> list[dict[str, Any]]:
    if debug is None or debug.get("selected_pair_record") is None:
        return []
    pair_record = debug["selected_pair_record"]
    total = len(pair_record.get("pair_src_idx", []))
    rows = []
    for idx in range(total):
        rows.append(
            {
                "image_id": relative_path,
                "pair_src_idx": int(pair_record["pair_src_idx"][idx]),
                "pair_dst_idx": int(pair_record["pair_dst_idx"][idx]),
                "base_metric": float(pair_record["base_metric"][idx]),
                "risk_penalty": float(pair_record["risk_penalty"][idx]),
                "adjusted_metric": float(pair_record["adjusted_metric"][idx]),
            }
        )
    return rows


def save_debug_overlay(path: Path, base_image: Image.Image, debug: dict[str, Any]) -> None:
    overlay = phase5e.token_mask_overlay(base_image, debug["source_mask"], debug["destination_mask"])
    phase5a.ensure_parent(path)
    overlay.save(path)


def save_debug_error_map(
    path: Path,
    sample: dict[str, Any],
    completed: torch.Tensor,
    boundary_band_radius: int,
) -> None:
    gt = sample["image"].permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)
    pred = completed.permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)
    hole_mask = phase5e.hole_mask_from_sample(sample)
    boundary_band = phase5e.boundary_band_from_hole(hole_mask, boundary_band_radius)
    error = np.linalg.norm(pred - gt, axis=-1)
    error = np.where(boundary_band, error, 0.0)
    error = np.clip(error / 64.0, 0.0, 1.0)
    heat = cv2.applyColorMap(np.round(error * 255.0).astype(np.uint8), cv2.COLORMAP_INFERNO)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    phase5a.ensure_parent(path)
    Image.fromarray(heat).save(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        phase5a.ensure_parent(path)
        path.write_text("", encoding="utf-8")
        return
    phase5a.ensure_parent(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_dataset_pass_phase5f(
    *,
    model,
    dataset,
    device: torch.device,
    args: argparse.Namespace,
    debug_artifact_dir: Path | None,
    collect_debug: bool,
    write_debug_side_effects: bool,
    save_pair_audit: bool,
    save_overlay: bool,
    save_error_map: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = []
    pair_rows = []
    record_summaries = []
    dataset_start = time.perf_counter()
    with torch.no_grad():
        for idx in range(len(dataset)):
            sample = dataset[idx]
            relative_path = sample["relative_path"]
            batch = {
                "relative_path": [relative_path],
                "image": sample["image"].unsqueeze(0),
                "mask": sample["mask"].unsqueeze(0),
            }
            start = time.perf_counter()
            output = model.generate_content(
                batch=batch,
                filter_ratio=args.num_token_for_sampling,
                filter_type="count",
                replicate=1,
                with_process_bar=False,
                mask_low_to_high=False,
                sample_largest=True,
                calculate_acc_and_prob=False,
                num_token_per_iter=args.num_token_per_iter,
                accumulate_time=None,
                raster_order=False,
            )
            torch.cuda.synchronize(device)
            debug = phase5e.compact_debug(output.get("safe_tome_debug")) if collect_debug else None
            if write_debug_side_effects and debug_artifact_dir is not None and debug is not None:
                base_image = tensor_to_pil_rgb(sample["image"])
                if save_overlay:
                    save_debug_overlay(debug_artifact_dir / "overlays" / relative_path, base_image, debug)
                if save_error_map:
                    save_debug_error_map(
                        debug_artifact_dir / "error_maps" / relative_path,
                        sample=sample,
                        completed=output["completed"][0],
                        boundary_band_radius=args.boundary_band_radius,
                    )
                if save_pair_audit:
                    pair_rows.extend(debug_pair_rows(relative_path, debug))
                record_summaries.append(
                    {
                        "relative_path": relative_path,
                        "phase5f_timing_profile": phase5a.to_serializable(output.get("phase5f_timing_profile")),
                        "safe_tome_profile": phase5a.to_serializable(output.get("safe_tome_profile")),
                        "safe_tome_debug_summary": phase5e.compact_debug_summary(debug),
                    }
                )
            elapsed = time.perf_counter() - start
            records.append(
                {
                    "relative_path": relative_path,
                    "wall_time_sec": float(elapsed),
                    "phase4a_timing_profile": output.get("phase4a_timing_profile"),
                    "phase5f_timing_profile": output.get("phase5f_timing_profile"),
                    "safe_tome_profile": output.get("safe_tome_profile"),
                    "ga_spg_profile": output.get("ga_spg_profile"),
                    "safe_tome_debug": debug,
                    "shape_or_sampling_crash": False,
                }
            )
    if write_debug_side_effects and debug_artifact_dir is not None:
        if save_pair_audit:
            write_csv(debug_artifact_dir / "pair_audit_debug.csv", pair_rows)
        (debug_artifact_dir / "record_debug_summary.json").write_text(
            json.dumps(phase5a.to_serializable(record_summaries), indent=2),
            encoding="utf-8",
        )
    dataset_elapsed = time.perf_counter() - dataset_start
    return records, {
        "dataset_wall_time_sec": float(dataset_elapsed),
        "dataset_wall_time_per_image_sec": float(dataset_elapsed / max(len(dataset), 1)),
        "debug_pair_rows": int(len(pair_rows)),
        "debug_records": int(len(record_summaries)),
    }


def summarize_phase5f_repeat(records: list[dict[str, Any]], pass_meta: dict[str, Any], gpu: int) -> dict[str, Any]:
    wall_times = [float(record["wall_time_sec"]) for record in records]
    timing_means = {}
    event_count_means = {}
    for bucket in PHASE5F_BUCKETS:
        values = []
        counts = []
        for record in records:
            profile = record.get("phase5f_timing_profile") or {}
            values.append(float((profile.get("timings_ms") or {}).get(bucket, 0.0)))
            counts.append(int((profile.get("event_counts") or {}).get(bucket, 0)))
        timing_means[bucket] = float(np.mean(values)) if values else 0.0
        event_count_means[bucket] = float(np.mean(counts)) if counts else 0.0

    safe_fields = [
        "removed_tokens",
        "compressed_tokens",
        "eligible_tokens",
        "safe_tokens",
        "protect_tokens",
        "actual_compression_ratio",
    ]
    safe_profile_means = {}
    for field in safe_fields:
        values = [float((record.get("safe_tome_profile") or {}).get(field, 0.0)) for record in records]
        safe_profile_means[field] = float(np.mean(values)) if values else 0.0

    return {
        "sample_wall_time_sec_mean": float(np.mean(wall_times)) if wall_times else 0.0,
        "sample_wall_time_sec_std_within_repeat": float(np.std(wall_times, ddof=0)) if wall_times else 0.0,
        "dataset_wall_time_sec": float(pass_meta["dataset_wall_time_sec"]),
        "dataset_wall_time_per_image_sec": float(pass_meta["dataset_wall_time_per_image_sec"]),
        "peak_memory_allocated_mb": float(torch.cuda.max_memory_allocated(gpu) / 1024 / 1024),
        "peak_memory_reserved_mb": float(torch.cuda.max_memory_reserved(gpu) / 1024 / 1024),
        "timings_ms_mean": timing_means,
        "event_count_mean": event_count_means,
        "safe_profile_mean": safe_profile_means,
        "debug_pair_rows": int(pass_meta["debug_pair_rows"]),
        "debug_records": int(pass_meta["debug_records"]),
    }


def summarize_variant_repeats(repeats: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "sample_wall_time_sec": phase5a.stats([float(repeat["sample_wall_time_sec_mean"]) for repeat in repeats]),
        "dataset_wall_time_sec": phase5a.stats([float(repeat["dataset_wall_time_sec"]) for repeat in repeats]),
        "dataset_wall_time_per_image_sec": phase5a.stats(
            [float(repeat["dataset_wall_time_per_image_sec"]) for repeat in repeats]
        ),
        "peak_memory_allocated_mb": phase5a.stats([float(repeat["peak_memory_allocated_mb"]) for repeat in repeats]),
        "peak_memory_reserved_mb": phase5a.stats([float(repeat["peak_memory_reserved_mb"]) for repeat in repeats]),
        "timings_ms": {
            bucket: phase5a.stats([float(repeat["timings_ms_mean"][bucket]) for repeat in repeats])
            for bucket in PHASE5F_BUCKETS
        },
        "event_count_mean": {
            bucket: phase5a.stats([float(repeat["event_count_mean"][bucket]) for repeat in repeats])
            for bucket in PHASE5F_BUCKETS
        },
        "safe_profile": {
            field: phase5a.stats([float(repeat["safe_profile_mean"][field]) for repeat in repeats])
            for field in repeats[0]["safe_profile_mean"].keys()
        },
        "debug_pair_rows": phase5a.stats([float(repeat["debug_pair_rows"]) for repeat in repeats]),
        "debug_records": phase5a.stats([float(repeat["debug_records"]) for repeat in repeats]),
    }


def load_phase5e_quality(phase5e_dir: Path) -> dict[tuple[str, str], dict[str, float]]:
    table = {}
    with (phase5e_dir / "comparison_table.csv").open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            table[(row["variant"], row["mask_group"])] = {key: row[key] for key in row.keys()}
    return table


def run_equivalence_smoke(
    *,
    put_root: Path,
    checkpoint: str,
    device: torch.device,
    args: argparse.Namespace,
    dataset,
) -> dict[str, Any]:
    sample = dataset[0]
    batch = {
        "relative_path": [sample["relative_path"]],
        "image": sample["image"].unsqueeze(0).to(device),
        "mask": sample["mask"].unsqueeze(0).to(device),
    }
    outputs = {}
    for key, gaspg_debug in (("lean", False), ("debug", True)):
        env = arm_env(
            {
                "safe_tome_r": 192,
                "score_mode": "ga_spg_soft",
                "gaspg_debug": gaspg_debug,
                "save_pair_audit": gaspg_debug,
                "save_overlay": gaspg_debug,
                "save_error_map": gaspg_debug,
            },
            args,
        )
        apply_put_env(env)
        phase5a.set_seed(args.seed)
        model = phase5a.load_model(put_root, checkpoint, device)
        with torch.no_grad():
            output = model.generate_content(
                batch=batch,
                filter_ratio=args.num_token_for_sampling,
                filter_type="count",
                replicate=1,
                with_process_bar=False,
                mask_low_to_high=False,
                sample_largest=True,
                calculate_acc_and_prob=False,
                num_token_per_iter=args.num_token_per_iter,
                accumulate_time=None,
                raster_order=False,
            )
        outputs[key] = {
            "completed": output["completed"][0].detach().cpu(),
            "safe_tome_profile": phase5a.to_serializable(output.get("safe_tome_profile")),
            "phase5f_timing_profile": phase5a.to_serializable(output.get("phase5f_timing_profile")),
        }
        del model
        torch.cuda.empty_cache()
    diff = (outputs["lean"]["completed"].float() - outputs["debug"]["completed"].float()).abs()
    return {
        "sample_id": sample["relative_path"],
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
        "removed_tokens_lean": int((outputs["lean"]["safe_tome_profile"] or {}).get("removed_tokens", 0)),
        "removed_tokens_debug": int((outputs["debug"]["safe_tome_profile"] or {}).get("removed_tokens", 0)),
    }


def build_timing_rows(variants: list[dict[str, Any]], quality_ref: dict[tuple[str, str], dict[str, float]]) -> list[dict[str, Any]]:
    rows = []
    quality_key_map = {
        "PureDistance-r192-lean": "PureDistance-r192",
        "GA-SPG-soft-r192-lean": "GA-SPG-soft-r192",
        "GA-SPG-soft-r192-debug": "GA-SPG-soft-r192",
    }
    for variant in variants:
        quality_row = quality_ref[(quality_key_map[variant["label"]], "10-60")]
        row = {
            "variant": variant["label"],
            "sample_wall_time_mean_sec": variant["timing_summary"]["sample_wall_time_sec"]["mean"],
            "sample_wall_time_std_sec": variant["timing_summary"]["sample_wall_time_sec"]["std"],
            "dataset_wall_time_mean_sec": variant["timing_summary"]["dataset_wall_time_sec"]["mean"],
            "dataset_wall_time_std_sec": variant["timing_summary"]["dataset_wall_time_sec"]["std"],
            "dataset_wall_time_per_image_mean_sec": variant["timing_summary"]["dataset_wall_time_per_image_sec"]["mean"],
            "dataset_wall_time_per_image_std_sec": variant["timing_summary"]["dataset_wall_time_per_image_sec"]["std"],
            "peak_memory_mean_mb": variant["timing_summary"]["peak_memory_allocated_mb"]["mean"],
            "peak_memory_std_mb": variant["timing_summary"]["peak_memory_allocated_mb"]["std"],
            "actual_removed_tokens_mean": variant["timing_summary"]["safe_profile"]["removed_tokens"]["mean"],
            "actual_compression_ratio_mean": variant["timing_summary"]["safe_profile"]["actual_compression_ratio"]["mean"],
            "PSNR_masked_mean_ref_phase5e": float(quality_row["PSNR_masked_mean"]),
            "PSNR_boundary_mean_ref_phase5e": float(quality_row["PSNR_boundary_mean"]),
            "LPIPS_masked_mean_ref_phase5e": float(quality_row["LPIPS_masked_mean"]),
            "LPIPS_boundary_mean_ref_phase5e": float(quality_row["LPIPS_boundary_mean"]),
        }
        for bucket in PHASE5F_BUCKETS:
            row[f"{bucket}_mean_ms"] = variant["timing_summary"]["timings_ms"][bucket]["mean"]
            row[f"{bucket}_std_ms"] = variant["timing_summary"]["timings_ms"][bucket]["std"]
        rows.append(row)
    return rows


def build_latency_rows(variants: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    pure = variants["pure_distance_r192_lean"]
    ga_lean = variants["ga_spg_soft_r192_lean"]
    ga_debug = variants["ga_spg_soft_r192_debug"]

    def mean(summary: dict[str, Any], key: str) -> float:
        return float(summary["timing_summary"][key]["mean"])

    def bucket(summary: dict[str, Any], bucket_name: str) -> float:
        return float(summary["timing_summary"]["timings_ms"][bucket_name]["mean"])

    rows = []
    for variant in (pure, ga_lean, ga_debug):
        row = {
            "variant": variant["label"],
            "dataset_wall_time_per_image_mean_sec": mean(variant, "dataset_wall_time_per_image_sec"),
            "dataset_wall_time_per_image_std_sec": float(variant["timing_summary"]["dataset_wall_time_per_image_sec"]["std"]),
            "sample_wall_time_mean_sec": mean(variant, "sample_wall_time_sec"),
            "sample_wall_time_std_sec": float(variant["timing_summary"]["sample_wall_time_sec"]["std"]),
            "total_inference_cuda_mean_ms": bucket(variant, "total_inference_time"),
            "total_inference_cuda_std_ms": float(variant["timing_summary"]["timings_ms"]["total_inference_time"]["std"]),
            "peak_memory_mean_mb": float(variant["timing_summary"]["peak_memory_allocated_mb"]["mean"]),
            "actual_removed_tokens_mean": float(variant["timing_summary"]["safe_profile"]["removed_tokens"]["mean"]),
            "delta_vs_puredistance_dataset_pct": 0.0,
            "delta_vs_puredistance_cuda_pct": 0.0,
            "delta_vs_ga_lean_dataset_pct": 0.0,
            "delta_vs_ga_lean_cuda_pct": 0.0,
        }
        if variant["name"] != pure["name"]:
            row["delta_vs_puredistance_dataset_pct"] = (
                mean(variant, "dataset_wall_time_per_image_sec") - mean(pure, "dataset_wall_time_per_image_sec")
            ) / max(mean(pure, "dataset_wall_time_per_image_sec"), 1.0e-8)
            row["delta_vs_puredistance_cuda_pct"] = (
                bucket(variant, "total_inference_time") - bucket(pure, "total_inference_time")
            ) / max(bucket(pure, "total_inference_time"), 1.0e-8)
        if variant["name"] != ga_lean["name"]:
            row["delta_vs_ga_lean_dataset_pct"] = (
                mean(variant, "dataset_wall_time_per_image_sec") - mean(ga_lean, "dataset_wall_time_per_image_sec")
            ) / max(mean(ga_lean, "dataset_wall_time_per_image_sec"), 1.0e-8)
            row["delta_vs_ga_lean_cuda_pct"] = (
                bucket(variant, "total_inference_time") - bucket(ga_lean, "total_inference_time")
            ) / max(bucket(ga_lean, "total_inference_time"), 1.0e-8)
        rows.append(row)
    return rows


def evaluate_conclusion(variants: dict[str, dict[str, Any]]) -> dict[str, Any]:
    pure = variants["pure_distance_r192_lean"]
    ga_lean = variants["ga_spg_soft_r192_lean"]
    ga_debug = variants["ga_spg_soft_r192_debug"]

    pure_wall = float(pure["timing_summary"]["dataset_wall_time_per_image_sec"]["mean"])
    ga_lean_wall = float(ga_lean["timing_summary"]["dataset_wall_time_per_image_sec"]["mean"])
    ga_debug_wall = float(ga_debug["timing_summary"]["dataset_wall_time_per_image_sec"]["mean"])
    pure_cuda = float(pure["timing_summary"]["timings_ms"]["total_inference_time"]["mean"])
    ga_lean_cuda = float(ga_lean["timing_summary"]["timings_ms"]["total_inference_time"]["mean"])
    ga_debug_cuda = float(ga_debug["timing_summary"]["timings_ms"]["total_inference_time"]["mean"])

    lean_overhead_dataset = (ga_lean_wall - pure_wall) / max(pure_wall, 1.0e-8)
    lean_overhead_cuda = (ga_lean_cuda - pure_cuda) / max(pure_cuda, 1.0e-8)
    debug_delta_dataset = (ga_debug_wall - ga_lean_wall) / max(ga_lean_wall, 1.0e-8)
    debug_delta_cuda = (ga_debug_cuda - ga_lean_cuda) / max(ga_lean_cuda, 1.0e-8)

    if lean_overhead_dataset < 0.03 and lean_overhead_cuda < 0.03:
        status = "Re-evaluate"
        reason = "Lean GA-SPG overhead dropped below 3%, so the Phase 5E system-level latency penalty is not stable."
    elif lean_overhead_dataset > 0.10 or lean_overhead_cuda > 0.10:
        status = "No-Go"
        reason = "Lean GA-SPG overhead remains above 10%, so the Phase 5E latency No-Go is attributable to model compute rather than debug-only side effects."
    else:
        status = "Blocked"
        reason = "Lean overhead improved but not enough to clear the optional-quality-mode threshold."

    return {
        "status": status,
        "reason": reason,
        "lean_overhead_vs_puredistance_dataset_pct": float(lean_overhead_dataset),
        "lean_overhead_vs_puredistance_cuda_pct": float(lean_overhead_cuda),
        "debug_delta_vs_lean_dataset_pct": float(debug_delta_dataset),
        "debug_delta_vs_lean_cuda_pct": float(debug_delta_cuda),
    }


def write_metrics_md(
    path: Path,
    variants: list[dict[str, Any]],
    quality_ref: dict[tuple[str, str], dict[str, float]],
    conclusion: dict[str, Any],
) -> None:
    quality_key_map = {
        "PureDistance-r192-lean": "PureDistance-r192",
        "GA-SPG-soft-r192-lean": "GA-SPG-soft-r192",
        "GA-SPG-soft-r192-debug": "GA-SPG-soft-r192",
    }
    lines = [
        "# Phase 5F GA-SPG Overhead Attribution",
        "",
        "## Lean Vs Debug",
        "",
        "| Variant | Dataset Wall / Image (s) | CUDA Total (ms) | Peak Mem (MiB) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for variant in variants:
        lines.append(
            "| {label} | {wall:.4f} +/- {wall_std:.4f} | {cuda:.3f} +/- {cuda_std:.3f} | {mem:.2f} |".format(
                label=variant["label"],
                wall=variant["timing_summary"]["dataset_wall_time_per_image_sec"]["mean"],
                wall_std=variant["timing_summary"]["dataset_wall_time_per_image_sec"]["std"],
                cuda=variant["timing_summary"]["timings_ms"]["total_inference_time"]["mean"],
                cuda_std=variant["timing_summary"]["timings_ms"]["total_inference_time"]["std"],
                mem=variant["timing_summary"]["peak_memory_allocated_mb"]["mean"],
            )
        )
    lines.extend(
        [
            "",
            "## Quality Reference (Inherited From Phase 5E, Same Algorithm / Checkpoint)",
            "",
            "| Variant | Masked PSNR | Boundary PSNR | Masked LPIPS | Boundary LPIPS |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for variant in variants:
        ref = quality_ref[(quality_key_map[variant["label"]], "10-60")]
        lines.append(
            "| {label} | {masked_psnr:.4f} | {boundary_psnr:.4f} | {masked_lpips:.4f} | {boundary_lpips:.4f} |".format(
                label=variant["label"],
                masked_psnr=float(ref["PSNR_masked_mean"]),
                boundary_psnr=float(ref["PSNR_boundary_mean"]),
                masked_lpips=float(ref["LPIPS_masked_mean"]),
                boundary_lpips=float(ref["LPIPS_boundary_mean"]),
            )
        )
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            f"- status: `{conclusion['status']}`",
            f"- lean_overhead_vs_puredistance_dataset_pct: `{conclusion['lean_overhead_vs_puredistance_dataset_pct']:.4%}`",
            f"- lean_overhead_vs_puredistance_cuda_pct: `{conclusion['lean_overhead_vs_puredistance_cuda_pct']:.4%}`",
            f"- debug_delta_vs_lean_dataset_pct: `{conclusion['debug_delta_vs_lean_dataset_pct']:.4%}`",
            f"- debug_delta_vs_lean_cuda_pct: `{conclusion['debug_delta_vs_lean_cuda_pct']:.4%}`",
            f"- reason: {conclusion['reason']}",
        ]
    )
    phase5a.ensure_parent(path)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    put_root = Path(args.put_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    phase5e_dir = Path(args.phase5e_dir).resolve()

    sample_records = phase5c.load_sample_records(Path(args.sample_manifest).resolve())
    input_res = phase5a.parse_hw(args.input_res)
    subset_image_dir = Path(sample_records[0]["resized_image_path"]).parent
    subset_mask_dir = Path(sample_records[0]["resized_mask_path"]).parent

    sys.path.insert(0, str(put_root))
    os.chdir(put_root)
    from scripts.inference import ImagePathDataset

    phase5a.set_seed(args.seed)
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    base_dataset = ImagePathDataset(str(subset_image_dir), str(subset_mask_dir), size=input_res)
    dataset = phase5e.FilteredDataset(base_dataset, [record["sample_id"] for record in sample_records])

    quality_ref = load_phase5e_quality(phase5e_dir)
    equivalence = run_equivalence_smoke(
        put_root=put_root,
        checkpoint=str(Path(args.checkpoint).resolve()),
        device=device,
        args=args,
        dataset=dataset,
    )

    arms = arm_specs()
    config_manifest = {
        "phase": "Phase 5F GA-SPG Overhead Attribution and Lean Inference Check",
        "seed": int(args.seed),
        "put_root": str(put_root),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "sample_manifest": str(Path(args.sample_manifest).resolve()),
        "phase5e_dir": str(phase5e_dir),
        "input_res": list(input_res),
        "warmup_runs": int(args.warmup_runs),
        "measure_runs": int(args.measure_runs),
        "num_token_per_iter": int(args.num_token_per_iter),
        "num_token_for_sampling": int(args.num_token_for_sampling),
        "boundary_ring_radius": int(args.boundary_ring_radius),
        "boundary_band_radius": int(args.boundary_band_radius),
        "ga_lambda_risk": float(args.ga_lambda_risk),
        "ga_weights": {
            "w1": float(args.ga_w1),
            "w2": float(args.ga_w2),
            "w3": float(args.ga_w3),
            "w4": float(args.ga_w4),
        },
        "ga_hard_threshold": float(args.ga_hard_threshold),
        "ga_hard_penalty": float(args.ga_hard_penalty),
        "equivalence_smoke": equivalence,
        "arms": arms,
    }
    (output_dir / "config_manifest.json").write_text(
        json.dumps(phase5a.to_serializable(config_manifest), indent=2),
        encoding="utf-8",
    )

    commands = []
    variants = []
    for spec in arms:
        env = arm_env(spec, args)
        apply_put_env(env)
        phase5a.set_seed(args.seed)
        model = phase5a.load_model(put_root, str(Path(args.checkpoint).resolve()), device)
        commands.append(" ".join([f"{key}={value}" for key, value in sorted(env.items()) if key.startswith("PUT_")]))

        for _ in range(args.warmup_runs):
            _records, _meta = run_dataset_pass_phase5f(
                model=model,
                dataset=dataset,
                device=device,
                args=args,
                debug_artifact_dir=None,
                collect_debug=spec["gaspg_debug"],
                write_debug_side_effects=False,
                save_pair_audit=False,
                save_overlay=False,
                save_error_map=False,
            )
            torch.cuda.synchronize(device)

        repeat_summaries = []
        for repeat_idx in range(args.measure_runs):
            phase5a.set_seed(args.seed)
            torch.cuda.synchronize(device)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(args.gpu)
            debug_artifact_dir = None
            if spec["gaspg_debug"]:
                debug_artifact_dir = output_dir / spec["name"] / f"repeat_{repeat_idx:02d}"
            records, pass_meta = run_dataset_pass_phase5f(
                model=model,
                dataset=dataset,
                device=device,
                args=args,
                debug_artifact_dir=debug_artifact_dir,
                collect_debug=spec["gaspg_debug"],
                write_debug_side_effects=spec["gaspg_debug"],
                save_pair_audit=spec["save_pair_audit"],
                save_overlay=spec["save_overlay"],
                save_error_map=spec["save_error_map"],
            )
            repeat_summaries.append(summarize_phase5f_repeat(records, pass_meta, args.gpu))
        del model
        torch.cuda.empty_cache()

        variants.append(
            {
                "name": spec["name"],
                "label": spec["label"],
                "env": env,
                "timing_summary": summarize_variant_repeats(repeat_summaries),
                "repeat_summaries": repeat_summaries,
            }
        )

    variants_by_name = {variant["name"]: variant for variant in variants}
    timing_rows = build_timing_rows(variants, quality_ref)
    latency_rows = build_latency_rows(variants_by_name)
    conclusion = evaluate_conclusion(variants_by_name)

    write_csv(output_dir / "phase5f_timing_table.csv", timing_rows)
    write_csv(output_dir / "lean_vs_debug_latency.csv", latency_rows)
    write_metrics_md(output_dir / "METRICS.md", variants, quality_ref, conclusion)
    (output_dir / "COMMANDS.md").write_text("\n".join(commands) + "\n", encoding="utf-8")

    summary = {
        "config_manifest": config_manifest,
        "commands": commands,
        "variants": phase5a.to_serializable(variants),
        "conclusion": conclusion,
        "artifacts": {
            "timing_table_csv": str((output_dir / "phase5f_timing_table.csv").resolve()),
            "lean_vs_debug_latency_csv": str((output_dir / "lean_vs_debug_latency.csv").resolve()),
            "metrics_md": str((output_dir / "METRICS.md").resolve()),
        },
    }
    (output_dir / "phase5f_timing_breakdown.json").write_text(
        json.dumps(phase5a.to_serializable(summary), indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
