#!/usr/bin/env python3
"""Phase 5E zero-shot GA-SPG probe."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw

import phase5a_standardized_comparison as phase5a
import phase5c_selective_finetune_pilot as phase5c


REPORT_GROUPS = ("20-40", "40-60", "10-60", "high-texture")
REPO_ROOT = Path(__file__).resolve().parents[1]


class FilteredDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, relative_paths: list[str]):
        index_map = {}
        for idx in range(len(base_dataset)):
            sample = base_dataset[idx]
            index_map[sample["relative_path"]] = idx
        missing = [path for path in relative_paths if path not in index_map]
        if missing:
            raise KeyError(f"Missing relative paths in dataset: {missing[:5]}")
        self.base_dataset = base_dataset
        self.relative_paths = list(relative_paths)
        self.indices = [index_map[path] for path in self.relative_paths]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        return self.base_dataset[self.indices[index]]


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
        "--output-dir",
        default=str(REPO_ROOT / "runs" / "ga_spg_probe"),
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
    parser.add_argument("--boundary-zoom-cell", type=int, default=192)
    parser.add_argument("--boundary-zoom-pad", type=int, default=48)
    parser.add_argument("--high-texture-topk", type=int, default=50)
    parser.add_argument("--visual-sample-count", type=int, default=4)
    parser.add_argument("--ga-lambda-risk", type=float, default=0.35)
    parser.add_argument("--ga-w1", type=float, default=0.35)
    parser.add_argument("--ga-w2", type=float, default=0.35)
    parser.add_argument("--ga-w3", type=float, default=0.15)
    parser.add_argument("--ga-w4", type=float, default=0.15)
    parser.add_argument("--ga-hard-threshold", type=float, default=0.65)
    parser.add_argument("--ga-hard-penalty", type=float, default=1000000.0)
    parser.add_argument("--run-optional-r224", action="store_true", default=False)
    parser.add_argument("--run-hardveto-r192", action="store_true", default=False)
    return parser.parse_args()


def arm_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    arms = [
        {
            "name": "baseline",
            "label": "Baseline",
            "safe_tome_r": 0,
            "boundary_split": False,
            "score_mode": None,
            "timing_debug": False,
        },
        {
            "name": "pure_distance_r128",
            "label": "PureDistance-r128",
            "safe_tome_r": 128,
            "boundary_split": True,
            "score_mode": "distance",
            "timing_debug": False,
        },
        {
            "name": "pure_distance_r192",
            "label": "PureDistance-r192",
            "safe_tome_r": 192,
            "boundary_split": True,
            "score_mode": "distance",
            "timing_debug": False,
        },
        {
            "name": "ga_spg_soft_r128",
            "label": "GA-SPG-soft-r128",
            "safe_tome_r": 128,
            "boundary_split": True,
            "score_mode": "ga_spg_soft",
            "timing_debug": False,
        },
        {
            "name": "ga_spg_soft_r192",
            "label": "GA-SPG-soft-r192",
            "safe_tome_r": 192,
            "boundary_split": True,
            "score_mode": "ga_spg_soft",
            "timing_debug": False,
        },
    ]
    if args.run_optional_r224:
        arms.extend(
            [
                {
                    "name": "pure_distance_r224",
                    "label": "PureDistance-r224",
                    "safe_tome_r": 224,
                    "boundary_split": True,
                    "score_mode": "distance",
                    "timing_debug": False,
                },
                {
                    "name": "ga_spg_soft_r224",
                    "label": "GA-SPG-soft-r224",
                    "safe_tome_r": 224,
                    "boundary_split": True,
                    "score_mode": "ga_spg_soft",
                    "timing_debug": False,
                },
            ]
        )
    if args.run_hardveto_r192:
        arms.append(
            {
                "name": "ga_spg_hardveto_r192",
                "label": "GA-SPG-hardveto-r192",
                "safe_tome_r": 192,
                "boundary_split": True,
                "score_mode": "ga_spg_hardveto",
                "timing_debug": False,
            }
        )
    return arms


def arm_env(spec: dict[str, Any], args: argparse.Namespace, debug_mode: bool) -> dict[str, str]:
    env = {
        "PUT_GLOBAL_TOME_R": "0",
        "PUT_PHASE4A_TIMING": "1",
        "PUT_BOUNDARY_SPLIT": "1" if spec["boundary_split"] else "0",
        "PUT_BOUNDARY_RING_RADIUS": str(args.boundary_ring_radius),
        "PUT_SAFE_TOME_R": str(spec["safe_tome_r"]),
        "PUT_SAFE_TOME_DEBUG": "1" if debug_mode and spec["safe_tome_r"] > 0 else "0",
        "PUT_SIMILARITY_SCORER": "0",
        "PUT_SIMILARITY_LEARNABLE": "0",
        "PUT_SIMILARITY_GUIDANCE_SOURCE": "codec_quantized_detached",
        "PUT_GA_SPG_AUDIT": "0",
        "PUT_GA_SPG_LAMBDA_RISK": str(args.ga_lambda_risk),
        "PUT_GA_SPG_W1": str(args.ga_w1),
        "PUT_GA_SPG_W2": str(args.ga_w2),
        "PUT_GA_SPG_W3": str(args.ga_w3),
        "PUT_GA_SPG_W4": str(args.ga_w4),
        "PUT_GA_SPG_HARD_THRESHOLD": str(args.ga_hard_threshold),
        "PUT_GA_SPG_HARD_PENALTY": str(args.ga_hard_penalty),
    }
    if spec["score_mode"] is not None:
        env["PUT_SAFE_TOME_SCORE_MODE"] = str(spec["score_mode"])
    if debug_mode and spec["safe_tome_r"] > 0:
        env["PUT_GA_SPG_AUDIT"] = "1"
    return env


def apply_put_env(env: dict[str, str]) -> None:
    for key in list(os.environ.keys()):
        if key.startswith("PUT_"):
            os.environ.pop(key)
    os.environ.update(env)


def compact_debug(debug: dict[str, Any] | None) -> dict[str, Any] | None:
    if debug is None:
        return None
    return {
        "eligible_mask": debug["eligible_mask"][0, 0].detach().cpu().numpy().astype(bool),
        "source_mask": debug["source_mask"][0, 0].detach().cpu().numpy().astype(bool),
        "destination_mask": debug["destination_mask"][0, 0].detach().cpu().numpy().astype(bool),
        "assignment_map": debug["assignment_map"][0, 0].detach().cpu().numpy().astype(np.int32),
        "source_indices": list(debug["source_indices"][0]),
        "destination_indices": list(debug["destination_indices"][0]),
        "compressed_tokens": int(debug["compressed_tokens"][0]),
        "removed_tokens": int(debug["removed_tokens"][0]),
        "eligible_tokens": int(debug["eligible_tokens"][0]),
        "selected_pair_record": debug["selected_pair_records"][0],
        "candidate_pair_stats": debug["candidate_pair_stats"][0],
    }


def compact_debug_summary(debug: dict[str, Any] | None) -> dict[str, Any] | None:
    if debug is None:
        return None
    pair_record = debug.get("selected_pair_record") or {}
    risk_values = [float(value) for value in pair_record.get("risk_penalty", [])]
    adjusted_values = [float(value) for value in pair_record.get("adjusted_metric", [])]
    base_values = [float(value) for value in pair_record.get("base_metric", [])]
    return {
        "compressed_tokens": int(debug.get("compressed_tokens", 0)),
        "removed_tokens": int(debug.get("removed_tokens", 0)),
        "eligible_tokens": int(debug.get("eligible_tokens", 0)),
        "num_selected_pairs": int(len(pair_record.get("pair_src_idx", []))),
        "candidate_pair_stats": phase5a.to_serializable(debug.get("candidate_pair_stats")),
        "selected_pair_stats": {
            "risk_penalty": phase5a.stats(risk_values),
            "adjusted_metric": phase5a.stats(adjusted_values),
            "base_metric": phase5a.stats(base_values),
        },
    }


def run_dataset_pass_extended(
    model,
    dataset,
    device: torch.device,
    args: argparse.Namespace,
    save_outputs: bool,
    completed_dir: Path | None,
    masked_dir: Path | None,
    collect_debug: bool,
) -> list[dict[str, Any]]:
    records = []
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
            elapsed = time.perf_counter() - start

            if save_outputs and completed_dir is not None and masked_dir is not None:
                phase5a.save_tensor_image(output["completed"][0], completed_dir / relative_path)
                phase5a.save_tensor_image(output["masked_gt"][0], masked_dir / relative_path)

            records.append(
                {
                    "relative_path": relative_path,
                    "wall_time_sec": float(elapsed),
                    "phase4a_timing_profile": output.get("phase4a_timing_profile"),
                    "safe_tome_profile": output.get("safe_tome_profile"),
                    "global_tome_profile": output.get("global_tome_profile"),
                    "boundary_split_profile": output.get("boundary_split_profile"),
                    "similarity_profile": output.get("similarity_profile"),
                    "ga_spg_profile": output.get("ga_spg_profile"),
                    "safe_tome_debug": compact_debug(output.get("safe_tome_debug")) if collect_debug else None,
                    "shape_or_sampling_crash": False,
                }
            )
    return records


def compact_variant_summary(variant: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": variant["name"],
        "label": variant["label"],
        "target_r": int(variant["target_r"]),
        "completed_dir": variant["completed_dir"],
        "masked_dir": variant["masked_dir"],
        "timing_summary": phase5a.to_serializable(variant["timing_summary"]),
        "group_timing_summary": phase5a.to_serializable(variant["group_timing_summary"]),
        "quality": phase5a.to_serializable(variant["quality"]),
        "complexity": phase5a.to_serializable(variant["complexity"]),
        "artifact_records": [
            {
                "relative_path": record["relative_path"],
                "wall_time_sec": float(record["wall_time_sec"]),
                "shape_or_sampling_crash": bool(record.get("shape_or_sampling_crash", False)),
                "phase4a_timing_profile": phase5a.to_serializable(record.get("phase4a_timing_profile")),
                "safe_tome_profile": phase5a.to_serializable(record.get("safe_tome_profile")),
                "ga_spg_profile": phase5a.to_serializable(record.get("ga_spg_profile")),
                "safe_tome_debug_summary": compact_debug_summary(record.get("safe_tome_debug")),
            }
            for record in variant["artifact_records"]
        ],
    }


def hole_mask_from_sample(sample: dict[str, Any]) -> np.ndarray:
    mask = sample["mask"][0].detach().cpu().numpy().astype(np.float32)
    if float(mask.max()) <= 1.0:
        return mask < 0.5
    return mask < 127.0


def boundary_band_from_hole(hole_mask: np.ndarray, radius: int) -> np.ndarray:
    kernel = np.ones((radius * 2 + 1, radius * 2 + 1), dtype=np.uint8)
    hole_u8 = hole_mask.astype(np.uint8)
    dilated = cv2.dilate(hole_u8, kernel, iterations=1)
    eroded = cv2.erode(hole_u8, kernel, iterations=1)
    return dilated != eroded


def manual_region_psnr(gt_rgb: torch.Tensor, pred_rgb: torch.Tensor, region: np.ndarray) -> float:
    mask = torch.from_numpy(region).to(torch.bool)
    if int(mask.sum()) == 0:
        return 0.0
    diff = (gt_rgb - pred_rgb).float()
    mse = float(diff[:, mask].pow(2).mean().detach().cpu())
    if mse <= 1.0e-12:
        return 99.0
    return float(20.0 * math.log10(255.0 / math.sqrt(mse)))


def region_isolated_pred(gt_rgb: torch.Tensor, pred_rgb: torch.Tensor, region: np.ndarray) -> torch.Tensor:
    out = gt_rgb.clone()
    mask = torch.from_numpy(region).to(torch.bool)
    out[:, mask] = pred_rgb[:, mask]
    return out


def rgb_tensor_to_gray(image: torch.Tensor) -> torch.Tensor:
    return phase5a.rgb_tensor_to_gray_batch(image)


def lpips_ready(image: torch.Tensor, device: torch.device) -> torch.Tensor:
    return (image.unsqueeze(0).to(device) / 255.0) * 2.0 - 1.0


def group_membership(sample_record: dict[str, Any], group: str, high_texture_ids: set[str]) -> bool:
    if group == "10-60":
        return True
    if group == "high-texture":
        return sample_record["sample_id"] in high_texture_ids
    return sample_record["report_group"] == group


def summarize_records(records: list[dict[str, Any]], sample_map: dict[str, dict[str, Any]], metric_keys: list[str], high_texture_ids: set[str]) -> dict[str, Any]:
    out = {}
    for group in REPORT_GROUPS:
        selected = [record for record in records if group_membership(sample_map[record["relative_path"]], group, high_texture_ids)]
        out[group] = {"num_images": len(selected)}
        for key in metric_keys:
            out[group][key] = phase5a.stats([float(record[key]) for record in selected]) if selected else phase5a.stats([])
    return out


def compute_texture_scores(
    dataset,
    sample_records: list[dict[str, Any]],
    boundary_band_radius: int,
) -> list[dict[str, Any]]:
    scores = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        sample_id = sample["relative_path"]
        rgb = sample["image"].permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        hole_mask = hole_mask_from_sample(sample)
        band = boundary_band_from_hole(hole_mask, boundary_band_radius)
        visible_band = band & (~hole_mask)
        if int(visible_band.sum()) < 64:
            visible_band = ~hole_mask

        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        local_mean = cv2.blur(gray, (9, 9))
        local_sq_mean = cv2.blur(gray ** 2, (9, 9))
        local_var = np.maximum(local_sq_mean - local_mean ** 2, 0.0)
        edge_density = float((grad_mag[visible_band] > 24.0).mean()) if int(visible_band.sum()) > 0 else 0.0
        local_var_mean = float(local_var[visible_band].mean() / (255.0 ** 2)) if int(visible_band.sum()) > 0 else 0.0
        scores.append(
            {
                "sample_id": sample_id,
                "edge_density_visible_boundary": edge_density,
                "local_variance_visible_boundary": local_var_mean,
                "texture_score": edge_density + local_var_mean,
            }
        )
    return sorted(scores, key=lambda item: item["texture_score"], reverse=True)


def compute_quality_metrics(
    dataset,
    completed_dir: Path,
    sample_map: dict[str, dict[str, Any]],
    high_texture_ids: set[str],
    lpips_model,
    device: torch.device,
    boundary_band_radius: int,
) -> dict[str, Any]:
    from image_synthesis.utils.cal_metrics import get_PSNR, get_SSIM

    records = []
    with torch.no_grad():
        for idx in range(len(dataset)):
            sample = dataset[idx]
            relative_path = sample["relative_path"]
            gt_rgb = sample["image"].float()
            pred_rgb = phase5a.tensor_from_png(completed_dir / relative_path).float()
            hole_mask = hole_mask_from_sample(sample)
            boundary_band = boundary_band_from_hole(hole_mask, boundary_band_radius)
            known_mask = ~hole_mask

            gt_gray = rgb_tensor_to_gray(gt_rgb)
            pred_gray = rgb_tensor_to_gray(pred_rgb)
            full_lpips = float(lpips_model(lpips_ready(gt_rgb, device), lpips_ready(pred_rgb, device)).mean().detach().cpu())

            masked_pred = region_isolated_pred(gt_rgb, pred_rgb, hole_mask)
            boundary_pred = region_isolated_pred(gt_rgb, pred_rgb, boundary_band)

            masked_psnr = manual_region_psnr(gt_rgb, pred_rgb, hole_mask)
            boundary_psnr = manual_region_psnr(gt_rgb, pred_rgb, boundary_band)
            masked_ssim = float(get_SSIM(rgb_tensor_to_gray(gt_rgb), rgb_tensor_to_gray(masked_pred), full=False, win_size=51))
            boundary_ssim = float(get_SSIM(rgb_tensor_to_gray(gt_rgb), rgb_tensor_to_gray(boundary_pred), full=False, win_size=51))
            masked_lpips = float(lpips_model(lpips_ready(gt_rgb, device), lpips_ready(masked_pred, device)).mean().detach().cpu())
            boundary_lpips = float(lpips_model(lpips_ready(gt_rgb, device), lpips_ready(boundary_pred, device)).mean().detach().cpu())

            diff = (pred_rgb - gt_rgb).abs()
            band_mask = torch.from_numpy(boundary_band).to(torch.bool)
            known_mask_t = torch.from_numpy(known_mask).to(torch.bool)
            boundary_mae = float(diff[:, band_mask].mean().detach().cpu()) if int(band_mask.sum()) > 0 else 0.0
            unmasked_max_error = float(diff[:, known_mask_t].max().detach().cpu()) if int(known_mask_t.sum()) > 0 else 0.0

            records.append(
                {
                    "relative_path": relative_path,
                    "full_psnr": float(get_PSNR(gt_gray, pred_gray, tool="skimage")),
                    "full_ssim": float(get_SSIM(gt_gray, pred_gray, full=False, win_size=51)),
                    "full_lpips": full_lpips,
                    "masked_psnr": masked_psnr,
                    "masked_ssim": masked_ssim,
                    "masked_lpips": masked_lpips,
                    "boundary_psnr": boundary_psnr,
                    "boundary_ssim": boundary_ssim,
                    "boundary_lpips": boundary_lpips,
                    "boundary_mae": boundary_mae,
                    "unmasked_max_error": unmasked_max_error,
                }
            )

    metric_keys = [
        "full_psnr",
        "full_ssim",
        "full_lpips",
        "masked_psnr",
        "masked_ssim",
        "masked_lpips",
        "boundary_psnr",
        "boundary_ssim",
        "boundary_lpips",
        "boundary_mae",
        "unmasked_max_error",
    ]
    return {
        "records": records,
        "groups": summarize_records(records, sample_map, metric_keys, high_texture_ids),
    }


def summarize_group_repeat_metrics_extended(
    records: list[dict[str, Any]],
    sample_map: dict[str, dict[str, Any]],
    high_texture_ids: set[str],
) -> dict[str, Any]:
    out = {}
    for group in REPORT_GROUPS:
        selected = [record for record in records if group_membership(sample_map[record["relative_path"]], group, high_texture_ids)]
        timing_stats = {}
        token_fields = [
            "original_tokens",
            "eligible_tokens",
            "removed_tokens",
            "merged_tokens",
            "restored_tokens",
            "compressed_tokens",
        ]
        extra_fields = [
            "protect_tokens",
            "safe_tokens",
            "protect_overlap_tokens",
            "under_compression",
            "invalid_pair_count",
            "high_risk_pair_count",
            "mean_risk_penalty_selected",
            "mean_base_metric_selected",
            "mean_adjusted_metric_selected",
            "actual_compression_ratio",
        ]
        token_profile = {field: [] for field in token_fields}
        safe_profile_fields = {field: [] for field in extra_fields}
        bad_restore = []
        for bucket in phase5a.TIMING_BUCKETS:
            timing_stats[bucket] = float(
                np.mean(
                    [
                        float(((record.get("phase4a_timing_profile") or {}).get("timings_ms") or {}).get(bucket, 0.0))
                        for record in selected
                    ]
                )
            ) if selected else 0.0
        wall_times = [float(record["wall_time_sec"]) for record in selected]
        for record in selected:
            profile = record.get("phase4a_timing_profile") or {}
            token = profile.get("token_profile") or {}
            safe_profile = record.get("safe_tome_profile") or {}
            for field in token_fields:
                token_profile[field].append(float(token.get(field, safe_profile.get(field, 0.0))))
            for field in extra_fields:
                safe_profile_fields[field].append(float(safe_profile.get(field, 0.0)))
            bad_restore.append(0.0 if bool(safe_profile.get("restore_alignment_ok", True)) else 1.0)
        out[group] = {
            "num_images": len(selected),
            "wall_time_sec_mean": float(np.mean(wall_times)) if wall_times else 0.0,
            "timings_ms_mean": timing_stats,
            "token_profile_mean": {field: float(np.mean(values)) if values else 0.0 for field, values in token_profile.items()},
            "safe_profile_mean": {field: float(np.mean(values)) if values else 0.0 for field, values in safe_profile_fields.items()},
            "bad_restore_mean": float(np.mean(bad_restore)) if bad_restore else 0.0,
        }
    return out


def aggregate_group_repeat_metrics_extended(group_repeat_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for group in REPORT_GROUPS:
        out[group] = {
            "num_images": int(group_repeat_summaries[0][group]["num_images"]),
            "wall_time_sec": phase5a.stats([float(item[group]["wall_time_sec_mean"]) for item in group_repeat_summaries]),
            "timings_ms": {
                bucket: phase5a.stats([float(item[group]["timings_ms_mean"][bucket]) for item in group_repeat_summaries])
                for bucket in phase5a.TIMING_BUCKETS
            },
            "token_profile": {
                field: phase5a.stats([float(item[group]["token_profile_mean"][field]) for item in group_repeat_summaries])
                for field in group_repeat_summaries[0][group]["token_profile_mean"].keys()
            },
            "safe_profile": {
                field: phase5a.stats([float(item[group]["safe_profile_mean"][field]) for item in group_repeat_summaries])
                for field in group_repeat_summaries[0][group]["safe_profile_mean"].keys()
            },
            "bad_restore": phase5a.stats([float(item[group]["bad_restore_mean"]) for item in group_repeat_summaries]),
        }
    return out


def compute_group_complexity_extended(
    artifact_records: list[dict[str, Any]],
    sample_map: dict[str, dict[str, Any]],
    high_texture_ids: set[str],
    dim: int,
    hidden_dim: int,
    num_layers: int,
) -> dict[str, Any]:
    per_image = []
    for record in artifact_records:
        timing_profile = record.get("phase4a_timing_profile") or {}
        token_profile = timing_profile.get("token_profile") or {}
        steps = int(timing_profile.get("num_sampling_steps", 0))
        tokens_in_blocks = int(token_profile.get("merged_tokens", token_profile.get("original_tokens", 0)))
        total_macs = phase5a.transformer_core_macs(tokens_in_blocks, dim=dim, hidden_dim=hidden_dim, num_layers=num_layers) * steps
        per_image.append(
            {
                "relative_path": record["relative_path"],
                "core_gmacs": float(total_macs / 1e9),
                "core_gflops": float((total_macs * 2) / 1e9),
                "tokens_in_blocks": float(tokens_in_blocks),
            }
        )
    out = {}
    for group in REPORT_GROUPS:
        selected = [item for item in per_image if group_membership(sample_map[item["relative_path"]], group, high_texture_ids)]
        out[group] = {
            "num_images": len(selected),
            "core_gmacs": phase5a.stats([item["core_gmacs"] for item in selected]),
            "core_gflops": phase5a.stats([item["core_gflops"] for item in selected]),
            "tokens_in_blocks": phase5a.stats([item["tokens_in_blocks"] for item in selected]),
        }
    return out


def build_budget_rows(
    variant: dict[str, Any],
    sample_map: dict[str, dict[str, Any]],
    target_r: int,
) -> list[dict[str, Any]]:
    rows = []
    for record in variant["artifact_records"]:
        safe = record.get("safe_tome_profile") or {}
        sample = sample_map[record["relative_path"]]
        rows.append(
            {
                "image_id": record["relative_path"],
                "mask_ratio": float(sample["mask_ratio"]),
                "mask_ratio_bin": sample["report_group"],
                "arm": variant["label"],
                "target_r": int(target_r),
                "input_tokens": int(safe.get("original_tokens", 1024)),
                "safe_tokens": int(safe.get("safe_tokens", 0)),
                "protect_tokens": int(safe.get("protect_tokens", 0)),
                "eligible_tokens": int(safe.get("eligible_tokens", 0)),
                "actual_removed_tokens": int(safe.get("removed_tokens", 0)),
                "compressed_tokens": int(safe.get("compressed_tokens", safe.get("merged_tokens", 0))),
                "restored_tokens": int(safe.get("restored_tokens", 0)),
                "actual_compression_ratio": float(safe.get("actual_compression_ratio", 0.0)),
                "protect_overlap_tokens": int(safe.get("protect_overlap_tokens", 0)),
                "invalid_pair_count": int(safe.get("invalid_pair_count", 0)),
                "high_risk_pair_count": int(safe.get("high_risk_pair_count", 0)),
                "mean_risk_penalty_selected": float(safe.get("mean_risk_penalty_selected", 0.0)),
                "mean_base_metric_selected": float(safe.get("mean_base_metric_selected", 0.0)),
                "mean_adjusted_metric_selected": float(safe.get("mean_adjusted_metric_selected", 0.0)),
                "fallback_used": bool(safe.get("fallback_used", False)),
                "fallback_reason": str(safe.get("fallback_reason", "")),
                "under_compression": bool(safe.get("under_compression", False)),
                "bad_restore": int(not bool(safe.get("restore_alignment_ok", True))),
                "shape_or_sampling_crash": bool(record.get("shape_or_sampling_crash", False)),
            }
        )
    return rows


def selected_pair_map(record: dict[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    debug = record.get("safe_tome_debug")
    if debug is None or debug.get("selected_pair_record") is None:
        return {}
    pair_record = debug["selected_pair_record"]
    out = {}
    total = len(pair_record["pair_src_idx"])
    for idx in range(total):
        key = (int(pair_record["pair_src_idx"][idx]), int(pair_record["pair_dst_idx"][idx]))
        out[key] = {name: pair_record[name][idx] for name in pair_record.keys() if isinstance(pair_record[name], list)}
    return out


def build_pair_gate_audit(
    pure_variant: dict[str, Any],
    ga_variant: dict[str, Any],
    sample_map: dict[str, dict[str, Any]],
    target_r: int,
) -> list[dict[str, Any]]:
    pure_records = {item["relative_path"]: item for item in pure_variant["artifact_records"]}
    ga_records = {item["relative_path"]: item for item in ga_variant["artifact_records"]}
    rows = []
    for image_id, pure_record in pure_records.items():
        ga_record = ga_records[image_id]
        pure_pairs = selected_pair_map(pure_record)
        ga_pairs = selected_pair_map(ga_record)
        union_keys = sorted(set(pure_pairs.keys()) | set(ga_pairs.keys()))
        for key in union_keys:
            source = ga_pairs.get(key) or pure_pairs.get(key)
            sample = sample_map[image_id]
            rows.append(
                {
                    "image_id": image_id,
                    "mask_ratio_bin": sample["report_group"],
                    "arm": ga_variant["label"],
                    "target_r": int(target_r),
                    "actual_removed_tokens": int((ga_record.get("safe_tome_profile") or {}).get("removed_tokens", 0)),
                    "actual_removed_tokens_puredistance": int((pure_record.get("safe_tome_profile") or {}).get("removed_tokens", 0)),
                    "pair_src_idx": int(key[0]),
                    "pair_dst_idx": int(key[1]),
                    "base_metric": float(source.get("base_metric", 0.0)),
                    "risk_penalty": float(source.get("risk_penalty", 0.0)),
                    "adjusted_metric": float(source.get("adjusted_metric", 0.0)),
                    "pair_cos_raw": float(source.get("pair_cos_raw", 0.0)),
                    "pair_cos_3x3": float(source.get("pair_cos_3x3", 0.0)),
                    "pair_cos_9x9": float(source.get("pair_cos_9x9", 0.0)),
                    "pair_feature_l2_raw": float(source.get("pair_feature_l2_raw", 0.0)),
                    "pair_feature_l2_3x3": float(source.get("pair_feature_l2_3x3", 0.0)),
                    "pair_feature_l2_9x9": float(source.get("pair_feature_l2_9x9", 0.0)),
                    "pair_local_variance_risk": float(source.get("pair_local_variance_risk", 0.0)),
                    "pair_context_variance_risk": float(source.get("pair_context_variance_risk", 0.0)),
                    "pair_boundary_distance_min": float(source.get("pair_boundary_distance_min", 0.0)),
                    "pair_spatial_distance": float(source.get("pair_spatial_distance", 0.0)),
                    "selected_by_puredistance": key in pure_pairs,
                    "selected_by_gaspg": key in ga_pairs,
                }
            )
    return rows


def build_routing_overlap_rows(
    pure_variant: dict[str, Any],
    ga_variant: dict[str, Any],
    sample_map: dict[str, dict[str, Any]],
    target_r: int,
    high_texture_ids: set[str],
) -> list[dict[str, Any]]:
    rows = []
    pure_records = {item["relative_path"]: item for item in pure_variant["artifact_records"]}
    ga_records = {item["relative_path"]: item for item in ga_variant["artifact_records"]}
    per_group = defaultdict(lambda: {"pair_jaccard": [], "token_jaccard": [], "source_jaccard": [], "destination_jaccard": []})
    for image_id, pure_record in pure_records.items():
        ga_record = ga_records[image_id]
        pure_debug = pure_record.get("safe_tome_debug") or {}
        ga_debug = ga_record.get("safe_tome_debug") or {}
        pure_pair_set = set(selected_pair_map(pure_record).keys())
        ga_pair_set = set(selected_pair_map(ga_record).keys())
        pure_token_set = set((pure_debug.get("source_indices") or []) + (pure_debug.get("destination_indices") or []))
        ga_token_set = set((ga_debug.get("source_indices") or []) + (ga_debug.get("destination_indices") or []))
        pure_source = set(pure_debug.get("source_indices") or [])
        ga_source = set(ga_debug.get("source_indices") or [])
        pure_dest = set(pure_debug.get("destination_indices") or [])
        ga_dest = set(ga_debug.get("destination_indices") or [])

        def jaccard(left: set[Any], right: set[Any]) -> float:
            if len(left | right) == 0:
                return 1.0
            return float(len(left & right) / len(left | right))

        sample = sample_map[image_id]
        keys = [sample["report_group"], "10-60"]
        if sample["sample_id"] in high_texture_ids:
            keys.append("high-texture")
        for group in keys:
            per_group[group]["pair_jaccard"].append(jaccard(pure_pair_set, ga_pair_set))
            per_group[group]["token_jaccard"].append(jaccard(pure_token_set, ga_token_set))
            per_group[group]["source_jaccard"].append(jaccard(pure_source, ga_source))
            per_group[group]["destination_jaccard"].append(jaccard(pure_dest, ga_dest))

    for group in REPORT_GROUPS:
        current = per_group[group]
        rows.append(
            {
                "mask_group": group,
                "target_r": int(target_r),
                "pair_jaccard_mean": phase5a.stats(current["pair_jaccard"])["mean"],
                "pair_jaccard_std": phase5a.stats(current["pair_jaccard"])["std"],
                "token_jaccard_mean": phase5a.stats(current["token_jaccard"])["mean"],
                "token_jaccard_std": phase5a.stats(current["token_jaccard"])["std"],
                "source_jaccard_mean": phase5a.stats(current["source_jaccard"])["mean"],
                "destination_jaccard_mean": phase5a.stats(current["destination_jaccard"])["mean"],
            }
        )
    return rows


def token_mask_overlay(image: Image.Image, source_mask: np.ndarray | None, destination_mask: np.ndarray | None) -> Image.Image:
    arr = np.array(image).copy()
    if source_mask is not None:
        src = Image.fromarray(source_mask.astype(np.uint8) * 255).resize((image.width, image.height), Image.Resampling.NEAREST)
        src_mask = np.array(src) > 0
        arr[src_mask] = (32, 144, 255)
    if destination_mask is not None:
        dst = Image.fromarray(destination_mask.astype(np.uint8) * 255).resize((image.width, image.height), Image.Resampling.NEAREST)
        dst_mask = np.array(dst) > 0
        arr[dst_mask] = (46, 204, 113)
    return Image.fromarray(arr)


def draw_pair_lines(image: Image.Image, debug: dict[str, Any], risk_values: list[float] | None) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    token_h, token_w = debug["source_mask"].shape
    scale_x = image.width / float(token_w)
    scale_y = image.height / float(token_h)
    sources = debug.get("selected_pair_record", {}).get("pair_src_idx", []) if debug.get("selected_pair_record") else []
    destinations = debug.get("selected_pair_record", {}).get("pair_dst_idx", []) if debug.get("selected_pair_record") else []
    for idx, (src, dst) in enumerate(zip(sources, destinations)):
        sy, sx = divmod(int(src), token_w)
        dy, dx = divmod(int(dst), token_w)
        color = (255, 128, 0)
        if risk_values is not None and idx < len(risk_values):
            risk = float(risk_values[idx])
            color = (
                int(min(255, 64 + 191 * risk)),
                int(max(0, 255 - 160 * risk)),
                64,
            )
        draw.line(
            (
                (sx + 0.5) * scale_x,
                (sy + 0.5) * scale_y,
                (dx + 0.5) * scale_x,
                (dy + 0.5) * scale_y,
            ),
            fill=color,
            width=2,
        )
    return out


def crop_hole_region(image: Image.Image, mask_path: Path, pad: int) -> Image.Image:
    hole = np.array(Image.open(mask_path).convert("L")) < 127
    ys, xs = np.where(hole)
    if len(xs) == 0:
        return image
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, image.width)
    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad + 1, image.height)
    return image.crop((x0, y0, x1, y1))


def build_selected_overlay_figure(
    output_path: Path,
    sample_names: list[str],
    image_dir: Path,
    mask_dir: Path,
    variant_outputs: list[dict[str, Any]],
    cell: int,
    pad: int,
) -> None:
    rows = []
    for name in sample_names:
        base_image = Image.open(image_dir / name).convert("RGB")
        row = [phase5a.resize_cell(crop_hole_region(base_image, mask_dir / name, pad), cell)]
        for variant in variant_outputs:
            record = variant["record_map"][name]
            debug = record.get("safe_tome_debug")
            overlay = token_mask_overlay(base_image, debug["source_mask"], debug["destination_mask"])
            row.append(phase5a.resize_cell(crop_hole_region(overlay, mask_dir / name, pad), cell))
        rows.append(row)
    canvas = Image.new("RGB", (cell * len(rows[0]), cell * len(rows)))
    for y, row in enumerate(rows):
        for x, image in enumerate(row):
            canvas.paste(image, (x * cell, y * cell))
    phase5a.ensure_parent(output_path)
    canvas.save(output_path)


def build_pair_risk_overlay_figure(
    output_path: Path,
    sample_names: list[str],
    image_dir: Path,
    mask_dir: Path,
    variant: dict[str, Any],
    cell: int,
    pad: int,
) -> None:
    rows = []
    for name in sample_names:
        base_image = Image.open(image_dir / name).convert("RGB")
        record = variant["record_map"][name]
        debug = record.get("safe_tome_debug")
        risk_values = debug.get("selected_pair_record", {}).get("risk_penalty", []) if debug and debug.get("selected_pair_record") else []
        overlay = draw_pair_lines(base_image, debug, risk_values)
        rows.append([phase5a.resize_cell(crop_hole_region(overlay, mask_dir / name, pad), cell)])
    canvas = Image.new("RGB", (cell, cell * len(rows)))
    for y, row in enumerate(rows):
        canvas.paste(row[0], (0, y * cell))
    phase5a.ensure_parent(output_path)
    canvas.save(output_path)


def build_masked_region_comparison(
    output_path: Path,
    sample_names: list[str],
    image_dir: Path,
    mask_dir: Path,
    variant_outputs: list[dict[str, Any]],
    cell: int,
    pad: int,
) -> None:
    rows = []
    for name in sample_names:
        base = Image.open(image_dir / name).convert("RGB")
        masked = Image.open(mask_dir / name).convert("RGB")
        row = [phase5a.resize_cell(crop_hole_region(masked, mask_dir / name, pad), cell)]
        for variant in variant_outputs:
            image = Image.open(Path(variant["completed_dir"]) / name).convert("RGB")
            row.append(phase5a.resize_cell(crop_hole_region(image, mask_dir / name, pad), cell))
        rows.append(row)
    canvas = Image.new("RGB", (cell * len(rows[0]), cell * len(rows)))
    for y, row in enumerate(rows):
        for x, image in enumerate(row):
            canvas.paste(image, (x * cell, y * cell))
    phase5a.ensure_parent(output_path)
    canvas.save(output_path)


def build_boundary_error_map(
    output_path: Path,
    sample_names: list[str],
    dataset,
    mask_dir: Path,
    variant_outputs: list[dict[str, Any]],
    cell: int,
    pad: int,
    boundary_band_radius: int,
) -> None:
    sample_lookup = {dataset[idx]["relative_path"]: dataset[idx] for idx in range(len(dataset))}
    rows = []
    for name in sample_names:
        sample = sample_lookup[name]
        gt = sample["image"].permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)
        hole = hole_mask_from_sample(sample)
        band = boundary_band_from_hole(hole, boundary_band_radius)
        row = []
        for variant in variant_outputs:
            pred = np.array(Image.open(Path(variant["completed_dir"]) / name).convert("RGB"), dtype=np.float32)
            error = np.linalg.norm(pred - gt, axis=-1)
            error = np.where(band, error, 0.0)
            error = np.clip(error / 64.0, 0.0, 1.0)
            heat = (plt.cm.inferno(error)[..., :3] * 255.0).astype(np.uint8)
            heat_image = Image.fromarray(heat)
            row.append(phase5a.resize_cell(crop_hole_region(heat_image, mask_dir / name, pad), cell))
        rows.append(row)
    canvas = Image.new("RGB", (cell * len(rows[0]), cell * len(rows)))
    for y, row in enumerate(rows):
        for x, image in enumerate(row):
            canvas.paste(image, (x * cell, y * cell))
    phase5a.ensure_parent(output_path)
    canvas.save(output_path)


def plot_histogram(output_path: Path, series: dict[str, list[float]], title: str, xlabel: str) -> None:
    plt.figure(figsize=(9, 5))
    for label, values in series.items():
        if len(values) == 0:
            continue
        plt.hist(values, bins=32, alpha=0.45, label=label, density=True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    phase5a.ensure_parent(output_path)
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_rd_curve(output_path: Path, variants: list[dict[str, Any]]) -> None:
    plt.figure(figsize=(8, 6))
    markers = {"20-40": "o", "40-60": "s", "10-60": "^", "high-texture": "D"}
    for variant in variants:
        for group in REPORT_GROUPS:
            x = variant["group_timing_summary"][group]["safe_profile"]["actual_compression_ratio"]["mean"]
            y = variant["quality"]["groups"][group]["masked_psnr"]["mean"]
            plt.scatter([x], [y], marker=markers[group], label=f"{variant['label']} {group}")
    plt.xlabel("Actual Compression Ratio")
    plt.ylabel("Masked-Region PSNR")
    plt.title("RD Curve: Actual Removed vs Masked PSNR")
    handles, labels = plt.gca().get_legend_handles_labels()
    dedup = dict(zip(labels, handles))
    plt.legend(dedup.values(), dedup.keys(), fontsize=7, loc="best")
    plt.tight_layout()
    phase5a.ensure_parent(output_path)
    plt.savefig(output_path, dpi=200)
    plt.close()


def build_comparison_rows(eval_summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for variant in eval_summary["variants"]:
        for group in REPORT_GROUPS:
            row = {
                "variant": variant["label"],
                "variant_name": variant["name"],
                "mask_group": group,
                "num_images": int(variant["group_timing_summary"][group]["num_images"]),
                "PSNR_full_mean": variant["quality"]["groups"][group]["full_psnr"]["mean"],
                "SSIM_full_mean": variant["quality"]["groups"][group]["full_ssim"]["mean"],
                "LPIPS_full_mean": variant["quality"]["groups"][group]["full_lpips"]["mean"],
                "PSNR_masked_mean": variant["quality"]["groups"][group]["masked_psnr"]["mean"],
                "SSIM_masked_mean": variant["quality"]["groups"][group]["masked_ssim"]["mean"],
                "LPIPS_masked_mean": variant["quality"]["groups"][group]["masked_lpips"]["mean"],
                "PSNR_boundary_mean": variant["quality"]["groups"][group]["boundary_psnr"]["mean"],
                "SSIM_boundary_mean": variant["quality"]["groups"][group]["boundary_ssim"]["mean"],
                "LPIPS_boundary_mean": variant["quality"]["groups"][group]["boundary_lpips"]["mean"],
                "MAE_boundary_mean": variant["quality"]["groups"][group]["boundary_mae"]["mean"],
                "unmasked_max_error_mean": variant["quality"]["groups"][group]["unmasked_max_error"]["mean"],
                "end_to_end_latency_mean_sec": variant["group_timing_summary"][group]["wall_time_sec"]["mean"],
                "end_to_end_latency_std_sec": variant["group_timing_summary"][group]["wall_time_sec"]["std"],
                "peak_memory_mean_mb": variant["timing_summary"]["peak_memory_allocated_mb"]["mean"],
                "peak_memory_std_mb": variant["timing_summary"]["peak_memory_allocated_mb"]["std"],
                "transformer_core_GMACs_mean": variant["complexity"][group]["core_gmacs"]["mean"],
                "transformer_core_GFLOPs_mean": variant["complexity"][group]["core_gflops"]["mean"],
                "actual_removed_tokens_mean": variant["group_timing_summary"][group]["token_profile"]["removed_tokens"]["mean"],
                "actual_removed_tokens_std": variant["group_timing_summary"][group]["token_profile"]["removed_tokens"]["std"],
                "actual_compression_ratio_mean": variant["group_timing_summary"][group]["safe_profile"]["actual_compression_ratio"]["mean"],
                "invalid_pair_count_mean": variant["group_timing_summary"][group]["safe_profile"]["invalid_pair_count"]["mean"],
                "high_risk_pair_count_mean": variant["group_timing_summary"][group]["safe_profile"]["high_risk_pair_count"]["mean"],
                "mean_risk_penalty_selected": variant["group_timing_summary"][group]["safe_profile"]["mean_risk_penalty_selected"]["mean"],
                "mean_base_metric_selected": variant["group_timing_summary"][group]["safe_profile"]["mean_base_metric_selected"]["mean"],
                "mean_adjusted_metric_selected": variant["group_timing_summary"][group]["safe_profile"]["mean_adjusted_metric_selected"]["mean"],
                "bad_restore_mean": variant["group_timing_summary"][group]["bad_restore"]["mean"],
                "shape_or_sampling_crash": any(bool(item.get("shape_or_sampling_crash", False)) for item in variant["artifact_records"]),
            }
            for bucket in phase5a.TIMING_BUCKETS:
                row[f"{bucket}_mean_ms"] = variant["group_timing_summary"][group]["timings_ms"][bucket]["mean"]
                row[f"{bucket}_std_ms"] = variant["group_timing_summary"][group]["timings_ms"][bucket]["std"]
            rows.append(row)
    return rows


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


def evaluate_acceptance(eval_summary: dict[str, Any]) -> dict[str, Any]:
    variants = {variant["name"]: variant for variant in eval_summary["variants"]}
    out = {
        "quality_vs_puredistance": {},
        "quality_vs_baseline": {},
        "bad_restore": 0,
        "shape_or_sampling_crash": False,
        "go": False,
    }
    baseline = variants["baseline"]
    for name in [item for item in variants.keys() if item.startswith("ga_spg_soft_")]:
        target_r = name.rsplit("_r", 1)[-1]
        pure = variants[f"pure_distance_r{target_r}"]
        ga = variants[name]
        for group in ("20-40", "40-60", "10-60", "high-texture"):
            ga_q = ga["quality"]["groups"][group]
            pure_q = pure["quality"]["groups"][group]
            baseline_q = baseline["quality"]["groups"][group]
            out["quality_vs_puredistance"][f"{name}:{group}"] = {
                "masked_psnr_delta": float(ga_q["masked_psnr"]["mean"] - pure_q["masked_psnr"]["mean"]),
                "boundary_psnr_delta": float(ga_q["boundary_psnr"]["mean"] - pure_q["boundary_psnr"]["mean"]),
                "masked_lpips_delta": float(ga_q["masked_lpips"]["mean"] - pure_q["masked_lpips"]["mean"]),
                "boundary_lpips_delta": float(ga_q["boundary_lpips"]["mean"] - pure_q["boundary_lpips"]["mean"]),
                "actual_removed_delta": float(
                    ga["group_timing_summary"][group]["token_profile"]["removed_tokens"]["mean"]
                    - pure["group_timing_summary"][group]["token_profile"]["removed_tokens"]["mean"]
                ),
                "latency_delta": float(
                    ga["group_timing_summary"][group]["wall_time_sec"]["mean"]
                    - pure["group_timing_summary"][group]["wall_time_sec"]["mean"]
                ),
            }
            out["quality_vs_baseline"][f"{name}:{group}"] = {
                "masked_psnr_delta": float(ga_q["masked_psnr"]["mean"] - baseline_q["masked_psnr"]["mean"]),
                "boundary_psnr_delta": float(ga_q["boundary_psnr"]["mean"] - baseline_q["boundary_psnr"]["mean"]),
                "masked_lpips_delta": float(ga_q["masked_lpips"]["mean"] - baseline_q["masked_lpips"]["mean"]),
                "boundary_lpips_delta": float(ga_q["boundary_lpips"]["mean"] - baseline_q["boundary_lpips"]["mean"]),
            }
        out["bad_restore"] += int(
            sum(1 for row in ga["artifact_records"] if not bool((row.get("safe_tome_profile") or {}).get("restore_alignment_ok", True)))
        )
        out["shape_or_sampling_crash"] = out["shape_or_sampling_crash"] or any(
            bool(row.get("shape_or_sampling_crash", False)) for row in ga["artifact_records"]
        )

    if "ga_spg_soft_r192" in variants:
        main = out["quality_vs_puredistance"]["ga_spg_soft_r192:10-60"]
        out["go"] = (
            main["masked_psnr_delta"] > 0.0
            and main["boundary_psnr_delta"] >= 0.0
            and main["masked_lpips_delta"] <= 0.0
            and main["actual_removed_delta"] >= -1.0
            and main["latency_delta"] <= 0.01
            and out["bad_restore"] == 0
            and not out["shape_or_sampling_crash"]
        )
    return out


def write_metrics_md(path: Path, eval_summary: dict[str, Any], acceptance: dict[str, Any]) -> None:
    lines = [
        "# Phase 5E GA-SPG Probe",
        "",
        "## Core Table (10-60)",
        "",
        "| Variant | Full PSNR | Masked PSNR | Boundary PSNR | Masked LPIPS | Boundary LPIPS | Latency (s/image) | Actual Removed | Peak Mem (MiB) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant in eval_summary["variants"]:
        group = "10-60"
        lines.append(
            "| {label} | {full_psnr:.4f} | {masked_psnr:.4f} | {boundary_psnr:.4f} | {masked_lpips:.4f} | {boundary_lpips:.4f} | {lat:.4f} +/- {lat_std:.4f} | {removed:.2f} | {mem:.2f} |".format(
                label=variant["label"],
                full_psnr=variant["quality"]["groups"][group]["full_psnr"]["mean"],
                masked_psnr=variant["quality"]["groups"][group]["masked_psnr"]["mean"],
                boundary_psnr=variant["quality"]["groups"][group]["boundary_psnr"]["mean"],
                masked_lpips=variant["quality"]["groups"][group]["masked_lpips"]["mean"],
                boundary_lpips=variant["quality"]["groups"][group]["boundary_lpips"]["mean"],
                lat=variant["group_timing_summary"][group]["wall_time_sec"]["mean"],
                lat_std=variant["group_timing_summary"][group]["wall_time_sec"]["std"],
                removed=variant["group_timing_summary"][group]["token_profile"]["removed_tokens"]["mean"],
                mem=variant["timing_summary"]["peak_memory_allocated_mb"]["mean"],
            )
        )
    lines.extend(
        [
            "",
            "## Acceptance",
            "",
            f"- go: `{acceptance['go']}`",
            f"- bad_restore: `{acceptance['bad_restore']}`",
            f"- shape_or_sampling_crash: `{acceptance['shape_or_sampling_crash']}`",
        ]
    )
    phase5a.ensure_parent(path)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def choose_visual_samples(
    sample_map: dict[str, dict[str, Any]],
    high_texture_order: list[dict[str, Any]],
    count: int,
) -> list[str]:
    report_groups = {"20-40": [], "40-60": []}
    for item in high_texture_order:
        sample_meta = sample_map.get(item["sample_id"])
        if sample_meta is None:
            continue
        group = sample_meta["report_group"]
        if group in report_groups:
            report_groups[group].append(item["sample_id"])
    selected = []
    per_group = max(count // 2, 1)
    for group in ("20-40", "40-60"):
        selected.extend(report_groups[group][:per_group])
    for item in high_texture_order:
        if item["sample_id"] not in selected:
            selected.append(item["sample_id"])
        if len(selected) >= count:
            break
    return selected[:count]


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    put_root = Path(args.put_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_records = phase5c.load_sample_records(Path(args.sample_manifest).resolve())
    sample_map = phase5a.build_sample_group_map(sample_records)
    subset_image_dir = Path(sample_records[0]["resized_image_path"]).parent
    subset_mask_dir = Path(sample_records[0]["resized_mask_path"]).parent
    input_res = phase5a.parse_hw(args.input_res)

    sys.path.insert(0, str(put_root))
    os.chdir(put_root)
    from scripts.inference import ImagePathDataset
    import lpips

    phase5a.set_seed(args.seed)
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    base_dataset = ImagePathDataset(str(subset_image_dir), str(subset_mask_dir), size=input_res)
    dataset = FilteredDataset(base_dataset, [record["sample_id"] for record in sample_records])

    texture_scores = compute_texture_scores(dataset, sample_records, boundary_band_radius=args.boundary_band_radius)
    high_texture_top = texture_scores[: int(args.high_texture_topk)]
    high_texture_ids = {item["sample_id"] for item in high_texture_top}
    visual_sample_names = choose_visual_samples(sample_map, texture_scores, count=args.visual_sample_count)

    arms = arm_specs(args)
    commands = []
    variants = []
    token_budget_rows = []
    pair_audit_rows = []
    routing_overlap_rows = []

    config_manifest = {
        "phase": "Phase 5E GA-SPG probe",
        "seed": int(args.seed),
        "put_root": str(put_root),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "sample_manifest": str(Path(args.sample_manifest).resolve()),
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
        "high_texture_rule": {
            "topk": int(args.high_texture_topk),
            "texture_score": "edge_density_visible_boundary + local_variance_visible_boundary",
        },
        "high_texture_sample_ids": [item["sample_id"] for item in high_texture_top],
        "visual_sample_ids": visual_sample_names,
        "arms": arms,
    }
    (output_dir / "config_manifest.json").write_text(json.dumps(config_manifest, indent=2), encoding="utf-8")

    lpips_model = lpips.LPIPS(net="vgg", spatial=True).to(device)

    for spec in arms:
        timing_env = arm_env(spec, args, debug_mode=False)
        apply_put_env(timing_env)
        phase5a.set_seed(args.seed)
        model = phase5a.load_model(put_root, str(Path(args.checkpoint).resolve()), device)
        commands.append(" ".join([f"{key}={value}" for key, value in sorted(timing_env.items()) if key.startswith("PUT_")]))

        for _ in range(args.warmup_runs):
            _ = run_dataset_pass_extended(
                model=model,
                dataset=dataset,
                device=device,
                args=args,
                save_outputs=False,
                completed_dir=None,
                masked_dir=None,
                collect_debug=False,
            )
            torch.cuda.synchronize(device)

        repeat_summaries = []
        group_repeat_summaries = []
        for _ in range(args.measure_runs):
            phase5a.set_seed(args.seed)
            torch.cuda.synchronize(device)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(args.gpu)
            timing_records = run_dataset_pass_extended(
                model=model,
                dataset=dataset,
                device=device,
                args=args,
                save_outputs=False,
                completed_dir=None,
                masked_dir=None,
                collect_debug=False,
            )
            repeat_summaries.append(phase5a.summarize_repeat(timing_records, args.gpu))
            group_repeat_summaries.append(
                summarize_group_repeat_metrics_extended(timing_records, sample_map, high_texture_ids)
            )
        del model
        torch.cuda.empty_cache()

        artifact_env = arm_env(spec, args, debug_mode=spec["safe_tome_r"] > 0)
        apply_put_env(artifact_env)
        phase5a.set_seed(args.seed)
        artifact_model = phase5a.load_model(put_root, str(Path(args.checkpoint).resolve()), device)
        variant_dir = output_dir / spec["name"]
        completed_dir = variant_dir / "completed_single"
        masked_dir = variant_dir / "masked_gt"
        completed_dir.mkdir(parents=True, exist_ok=True)
        masked_dir.mkdir(parents=True, exist_ok=True)
        artifact_records = run_dataset_pass_extended(
            model=artifact_model,
            dataset=dataset,
            device=device,
            args=args,
            save_outputs=True,
            completed_dir=completed_dir,
            masked_dir=masked_dir,
            collect_debug=spec["safe_tome_r"] > 0,
        )
        dim = int(artifact_model.dim)
        hidden_dim = int(artifact_model.blocks[0].mlp.fc1.out_features)
        num_layers = int(len(artifact_model.blocks))
        del artifact_model
        torch.cuda.empty_cache()

        quality = compute_quality_metrics(
            dataset=dataset,
            completed_dir=completed_dir,
            sample_map=sample_map,
            high_texture_ids=high_texture_ids,
            lpips_model=lpips_model,
            device=device,
            boundary_band_radius=args.boundary_band_radius,
        )
        variant = {
            "name": spec["name"],
            "label": spec["label"],
            "target_r": int(spec["safe_tome_r"]),
            "completed_dir": str(completed_dir.resolve()),
            "masked_dir": str(masked_dir.resolve()),
            "repeat_summaries": repeat_summaries,
            "timing_summary": phase5a.summarize_variant_repeats(repeat_summaries),
            "group_timing_summary": aggregate_group_repeat_metrics_extended(group_repeat_summaries),
            "quality": quality,
            "complexity": compute_group_complexity_extended(
                artifact_records=artifact_records,
                sample_map=sample_map,
                high_texture_ids=high_texture_ids,
                dim=dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
            ),
            "artifact_records": artifact_records,
            "record_map": {item["relative_path"]: item for item in artifact_records},
        }
        variants.append(variant)
        token_budget_rows.extend(build_budget_rows(variant, sample_map, target_r=spec["safe_tome_r"]))

    del lpips_model
    torch.cuda.empty_cache()

    variants_by_name = {variant["name"]: variant for variant in variants}
    for target_r in [128, 192]:
        pure_name = f"pure_distance_r{target_r}"
        ga_name = f"ga_spg_soft_r{target_r}"
        if pure_name in variants_by_name and ga_name in variants_by_name:
            pair_audit_rows.extend(
                build_pair_gate_audit(
                    pure_variant=variants_by_name[pure_name],
                    ga_variant=variants_by_name[ga_name],
                    sample_map=sample_map,
                    target_r=target_r,
                )
            )
            routing_overlap_rows.extend(
                build_routing_overlap_rows(
                    pure_variant=variants_by_name[pure_name],
                    ga_variant=variants_by_name[ga_name],
                    sample_map=sample_map,
                    target_r=target_r,
                    high_texture_ids=high_texture_ids,
                )
            )

    comparison_rows = build_comparison_rows({"variants": variants})
    write_csv(output_dir / "comparison_table.csv", comparison_rows)
    write_csv(output_dir / "token_budget_by_image.csv", token_budget_rows)
    write_csv(output_dir / "pair_gate_audit.csv", pair_audit_rows)
    write_csv(output_dir / "routing_overlap_by_mask_ratio.csv", routing_overlap_rows)

    histogram_distance = {}
    histogram_texture = {}
    for variant in variants:
        if variant["target_r"] <= 0:
            continue
        distance_values = []
        texture_values = []
        for record in variant["artifact_records"]:
            debug = record.get("safe_tome_debug")
            if debug is None or debug.get("selected_pair_record") is None:
                continue
            pair_record = debug["selected_pair_record"]
            distance_values.extend([float(x) for x in pair_record.get("selected_source_boundary_distance", [])])
            distance_values.extend([float(x) for x in pair_record.get("selected_destination_boundary_distance", [])])
            texture_values.extend([float(x) for x in pair_record.get("selected_source_context_risk", [])])
            texture_values.extend([float(x) for x in pair_record.get("selected_destination_context_risk", [])])
        histogram_distance[variant["label"]] = distance_values
        histogram_texture[variant["label"]] = texture_values

    plot_histogram(
        output_path=output_dir / "selected_token_distance_hist.png",
        series=histogram_distance,
        title="Selected Token Boundary Distance",
        xlabel="Boundary Distance",
    )
    plot_histogram(
        output_path=output_dir / "selected_token_texture_hist.png",
        series=histogram_texture,
        title="Selected Token Context Risk",
        xlabel="Context Variance Risk",
    )
    plot_rd_curve(output_dir / "rd_curve_actual_removed_vs_quality.png", variants)

    overlay_variants = [
        {
            "label": variants_by_name[name]["label"],
            "record_map": variants_by_name[name]["record_map"],
        }
        for name in ["pure_distance_r128", "ga_spg_soft_r128", "pure_distance_r192", "ga_spg_soft_r192"]
        if name in variants_by_name
    ]
    build_selected_overlay_figure(
        output_path=output_dir / "pure_distance_vs_gaspg_selected_overlay.png",
        sample_names=visual_sample_names,
        image_dir=subset_image_dir,
        mask_dir=subset_mask_dir,
        variant_outputs=overlay_variants,
        cell=args.boundary_zoom_cell,
        pad=args.boundary_zoom_pad,
    )
    if "ga_spg_soft_r192" in variants_by_name:
        build_pair_risk_overlay_figure(
            output_path=output_dir / "merge_pair_risk_overlay.png",
            sample_names=visual_sample_names,
            image_dir=subset_image_dir,
            mask_dir=subset_mask_dir,
            variant=variants_by_name["ga_spg_soft_r192"],
            cell=args.boundary_zoom_cell,
            pad=args.boundary_zoom_pad,
        )
    build_masked_region_comparison(
        output_path=output_dir / "masked_region_comparison.png",
        sample_names=visual_sample_names,
        image_dir=subset_image_dir,
        mask_dir=subset_mask_dir,
        variant_outputs=[
            {"label": variant["label"], "completed_dir": variant["completed_dir"]}
            for variant in variants
        ],
        cell=args.boundary_zoom_cell,
        pad=args.boundary_zoom_pad,
    )
    build_boundary_error_map(
        output_path=output_dir / "boundary_band_error_map.png",
        sample_names=visual_sample_names,
        dataset=dataset,
        mask_dir=subset_mask_dir,
        variant_outputs=[
            {"label": variant["label"], "completed_dir": variant["completed_dir"]}
            for variant in variants
        ],
        cell=args.boundary_zoom_cell,
        pad=args.boundary_zoom_pad,
        boundary_band_radius=args.boundary_band_radius,
    )

    acceptance = evaluate_acceptance({"variants": variants})
    write_metrics_md(output_dir / "METRICS.md", {"variants": variants}, acceptance)
    (output_dir / "COMMANDS.md").write_text("\n".join(commands) + "\n", encoding="utf-8")

    summary = {
        "config_manifest": config_manifest,
        "commands": commands,
        "acceptance": acceptance,
        "variants": [compact_variant_summary(variant) for variant in variants],
        "high_texture_scores": texture_scores,
        "visual_sample_ids": visual_sample_names,
        "artifacts": {
            "comparison_table_csv": str((output_dir / "comparison_table.csv").resolve()),
            "token_budget_by_image_csv": str((output_dir / "token_budget_by_image.csv").resolve()),
            "pair_gate_audit_csv": str((output_dir / "pair_gate_audit.csv").resolve()),
            "routing_overlap_by_mask_ratio_csv": str((output_dir / "routing_overlap_by_mask_ratio.csv").resolve()),
            "rd_curve_path": str((output_dir / "rd_curve_actual_removed_vs_quality.png").resolve()),
            "selected_token_distance_hist_path": str((output_dir / "selected_token_distance_hist.png").resolve()),
            "selected_token_texture_hist_path": str((output_dir / "selected_token_texture_hist.png").resolve()),
            "selected_overlay_path": str((output_dir / "pure_distance_vs_gaspg_selected_overlay.png").resolve()),
            "merge_pair_risk_overlay_path": str((output_dir / "merge_pair_risk_overlay.png").resolve()),
            "masked_region_comparison_path": str((output_dir / "masked_region_comparison.png").resolve()),
            "boundary_band_error_map_path": str((output_dir / "boundary_band_error_map.png").resolve()),
        },
    }
    (output_dir / "gate_probe_summary.json").write_text(json.dumps(phase5a.to_serializable(summary), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
