#!/usr/bin/env python3
"""Phase 5A standardized medium-scale comparison on Places2 natural scenes."""

from __future__ import annotations

import argparse
import csv
import gc
import itertools
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


MASK_BUCKET_SPECS = [
    {"bucket": "10-20", "min_ratio": 0.10, "max_ratio": 0.20, "report_group": "10-20"},
    {"bucket": "20-30", "min_ratio": 0.20, "max_ratio": 0.30, "report_group": "20-40"},
    {"bucket": "30-40", "min_ratio": 0.30, "max_ratio": 0.40, "report_group": "20-40"},
    {"bucket": "40-50", "min_ratio": 0.40, "max_ratio": 0.50, "report_group": "40-60"},
    {"bucket": "50-60", "min_ratio": 0.50, "max_ratio": 0.60, "report_group": "40-60"},
]
REPORT_GROUPS = ("20-40", "40-60", "10-60")
TIMING_BUCKETS = (
    "Time_Scoring",
    "Time_TopK_and_Pairing",
    "Time_Merge",
    "Time_Attention",
    "Time_Restore",
    "Time_Block_Total",
)


def parse_hw(value: str) -> tuple[int, int]:
    parts = [int(x) for x in value.split(",")]
    if len(parts) != 2:
        raise ValueError(f"expected H,W but got {value!r}")
    return parts[0], parts[1]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def stats(values: list[float]) -> dict[str, float]:
    if len(values) == 0:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "min": float(array.min()),
        "max": float(array.max()),
        "std": float(array.std(ddof=0)),
    }


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_tensor_image(image: torch.Tensor, path: Path) -> None:
    arr = image.detach().cpu().clamp(0, 255).to(torch.uint8).permute(1, 2, 0).numpy()
    ensure_parent(path)
    Image.fromarray(arr).save(path)


def save_resized_rgb(source_path: Path, target_path: Path, size: tuple[int, int]) -> None:
    image = Image.open(source_path).convert("RGB").resize((size[1], size[0]), Image.Resampling.BILINEAR)
    ensure_parent(target_path)
    image.save(target_path)


def save_put_mask_from_pconv(source_path: Path, target_path: Path, size: tuple[int, int]) -> float:
    source = Image.open(source_path).convert("L").resize((size[1], size[0]), Image.Resampling.NEAREST)
    source_arr = np.array(source, dtype=np.uint8)
    visible_arr = np.where(source_arr > 127, 0, 255).astype(np.uint8)
    ensure_parent(target_path)
    Image.fromarray(visible_arr).save(target_path)
    return float((visible_arr == 0).mean())


def parse_bucket_counts(args: argparse.Namespace) -> dict[str, int]:
    return {
        "10-20": int(args.count_10_20),
        "20-30": int(args.count_20_30),
        "30-40": int(args.count_30_40),
        "40-50": int(args.count_40_50),
        "50-60": int(args.count_50_60),
    }


def ratio_in_bucket(ratio: float, bucket: str) -> bool:
    spec = next(item for item in MASK_BUCKET_SPECS if item["bucket"] == bucket)
    lower = spec["min_ratio"]
    upper = spec["max_ratio"]
    if bucket == "50-60":
        return lower <= ratio <= upper + 1e-8
    return lower <= ratio < upper


def resolve_validation_images(validation_list: Path, image_root: Path) -> list[dict[str, Any]]:
    relpaths = [line.strip() for line in validation_list.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = []
    for relpath in relpaths:
        basename = Path(relpath).name
        source_path = image_root / basename
        if not source_path.is_file():
            raise FileNotFoundError(f"validation image not found: {source_path}")
        records.append(
            {
                "validation_relpath": relpath,
                "image_uid": basename,
                "source_image_path": str(source_path.resolve()),
            }
        )
    return records


def build_mask_catalog(mask_root: Path, input_res: tuple[int, int]) -> list[dict[str, Any]]:
    catalog = []
    for path in sorted(mask_root.glob("*.png")):
        mask = Image.open(path).convert("L").resize((input_res[1], input_res[0]), Image.Resampling.NEAREST)
        mask_arr = np.array(mask, dtype=np.uint8)
        missing_ratio = float((mask_arr > 127).mean())
        bucket = None
        report_group = None
        for spec in MASK_BUCKET_SPECS:
            if ratio_in_bucket(missing_ratio, spec["bucket"]):
                bucket = spec["bucket"]
                report_group = spec["report_group"]
                break
        catalog.append(
            {
                "mask_uid": path.stem,
                "source_mask_path": str(path.resolve()),
                "mask_ratio": missing_ratio,
                "bucket": bucket,
                "report_group": report_group,
            }
        )
    return catalog


def select_spaced(items: list[dict[str, Any]], count: int, key: str) -> list[dict[str, Any]]:
    if len(items) < count:
        raise RuntimeError(f"need {count} items but only found {len(items)}")
    ordered = sorted(items, key=lambda item: item[key])
    if count == 1:
        return [ordered[len(ordered) // 2]]
    positions = np.linspace(0, len(ordered) - 1, count)
    return [ordered[int(round(position))] for position in positions]


def build_standardized_subset(
    output_dir: Path,
    validation_images: list[dict[str, Any]],
    mask_catalog: list[dict[str, Any]],
    bucket_counts: dict[str, int],
    input_res: tuple[int, int],
    seed: int,
) -> tuple[list[dict[str, Any]], Path, Path]:
    subset_dir = output_dir / "subset"
    image_dir = subset_dir / "images"
    mask_dir = subset_dir / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    total_required = sum(bucket_counts.values())
    if total_required > len(validation_images):
        raise RuntimeError(
            f"need {total_required} validation images but only {len(validation_images)} are available"
        )

    rng = np.random.default_rng(seed)
    shuffled_indices = rng.permutation(len(validation_images)).tolist()
    selected_images = [validation_images[idx] for idx in shuffled_indices[:total_required]]

    records = []
    cursor = 0
    sample_index = 0
    for spec in MASK_BUCKET_SPECS:
        bucket = spec["bucket"]
        bucket_count = int(bucket_counts[bucket])
        bucket_masks = [item for item in mask_catalog if item["bucket"] == bucket]
        selected_masks = select_spaced(bucket_masks, bucket_count, key="mask_ratio")
        bucket_images = selected_images[cursor : cursor + bucket_count]
        cursor += bucket_count

        for image_info, mask_info in zip(bucket_images, selected_masks):
            sample_name = f"sample_{sample_index:04d}_{bucket.replace('-', '_')}.png"
            output_image_path = image_dir / sample_name
            output_mask_path = mask_dir / sample_name
            save_resized_rgb(Path(image_info["source_image_path"]), output_image_path, size=input_res)
            saved_mask_ratio = save_put_mask_from_pconv(
                Path(mask_info["source_mask_path"]),
                output_mask_path,
                size=input_res,
            )
            records.append(
                {
                    "sample_id": sample_name,
                    "image_uid": image_info["image_uid"],
                    "mask_uid": mask_info["mask_uid"],
                    "bucket": bucket,
                    "report_group": spec["report_group"],
                    "mask_ratio": saved_mask_ratio,
                    "validation_relpath": image_info["validation_relpath"],
                    "source_image_path": image_info["source_image_path"],
                    "source_mask_path": mask_info["source_mask_path"],
                    "resized_image_path": str(output_image_path.resolve()),
                    "resized_mask_path": str(output_mask_path.resolve()),
                }
            )
            sample_index += 1

    return records, image_dir, mask_dir


def write_sample_manifest(sample_records: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "sample_id",
        "image_uid",
        "mask_uid",
        "bucket",
        "report_group",
        "mask_ratio",
        "validation_relpath",
        "source_image_path",
        "source_mask_path",
        "resized_image_path",
        "resized_mask_path",
    ]
    ensure_parent(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in sample_records:
            writer.writerow(record)


def summarize_subset(sample_records: list[dict[str, Any]]) -> dict[str, Any]:
    group_counts = {group: 0 for group in REPORT_GROUPS}
    bucket_counts = {spec["bucket"]: 0 for spec in MASK_BUCKET_SPECS}
    ratios_all = []
    ratios_by_group = {group: [] for group in list(REPORT_GROUPS) + ["10-20"]}
    for record in sample_records:
        ratio = float(record["mask_ratio"])
        bucket_counts[record["bucket"]] += 1
        ratios_all.append(ratio)
        ratios_by_group["10-60"].append(ratio)
        ratios_by_group[record["report_group"]].append(ratio)
        if record["report_group"] in group_counts:
            group_counts[record["report_group"]] += 1
    group_counts["10-60"] = len(sample_records)
    return {
        "num_samples": len(sample_records),
        "bucket_counts": bucket_counts,
        "report_group_counts": group_counts,
        "mask_ratio_10_60": stats(ratios_all),
        "mask_ratio_20_40": stats(ratios_by_group["20-40"]),
        "mask_ratio_40_60": stats(ratios_by_group["40-60"]),
    }


def build_variants() -> list[dict[str, Any]]:
    return [
        {
            "name": "put_baseline",
            "label": "Baseline",
            "global_tome_r": 0,
            "boundary_split": False,
            "similarity_scorer": False,
            "safe_tome_r": 0,
            "safe_tome_score_mode": "similarity",
        },
        {
            "name": "global_tome_r128",
            "label": "Global ToMe r=128",
            "global_tome_r": 128,
            "boundary_split": False,
            "similarity_scorer": False,
            "safe_tome_r": 0,
            "safe_tome_score_mode": "similarity",
        },
        {
            "name": "pure_distance_r128",
            "label": "Pure Distance r=128",
            "global_tome_r": 0,
            "boundary_split": True,
            "similarity_scorer": False,
            "safe_tome_r": 128,
            "safe_tome_score_mode": "distance",
        },
        {
            "name": "sbvc_r128",
            "label": "SBVC r=128",
            "global_tome_r": 0,
            "boundary_split": True,
            "similarity_scorer": True,
            "safe_tome_r": 128,
            "safe_tome_score_mode": "similarity",
        },
        {
            "name": "sbvc_r192",
            "label": "SBVC r=192",
            "global_tome_r": 0,
            "boundary_split": True,
            "similarity_scorer": True,
            "safe_tome_r": 192,
            "safe_tome_score_mode": "similarity",
        },
    ]


def set_variant_env(spec: dict[str, Any], args: argparse.Namespace) -> None:
    os.environ["PUT_GLOBAL_TOME_R"] = str(spec["global_tome_r"])
    os.environ["PUT_BOUNDARY_SPLIT"] = "1" if spec["boundary_split"] else "0"
    os.environ["PUT_BOUNDARY_RING_RADIUS"] = str(args.boundary_ring_radius)
    os.environ["PUT_SIMILARITY_SCORER"] = "1" if spec["similarity_scorer"] else "0"
    os.environ["PUT_SIMILARITY_ALPHA"] = str(args.alpha)
    os.environ["PUT_SIMILARITY_BETA"] = str(args.beta)
    os.environ["PUT_SIMILARITY_KERNEL_SIZE"] = str(args.kernel_size)
    os.environ["PUT_SIMILARITY_TOPK_RATIO"] = str(args.topk_ratio)
    os.environ["PUT_SAFE_TOME_R"] = str(spec["safe_tome_r"])
    os.environ["PUT_SAFE_TOME_SCORE_MODE"] = spec["safe_tome_score_mode"]
    os.environ["PUT_SAFE_TOME_DEBUG"] = "0"
    os.environ["PUT_PHASE4A_TIMING"] = "1"


def load_model(put_root: Path, checkpoint: str, device: torch.device):
    from scripts.inference import get_model

    model_args = argparse.Namespace(name=checkpoint)
    info = get_model(args=model_args, model_name=checkpoint)
    model = info["model"].to(device).eval()
    return model


def run_dataset_pass(
    model,
    dataset,
    device: torch.device,
    args: argparse.Namespace,
    save_outputs: bool,
    completed_dir: Path | None,
    masked_dir: Path | None,
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
                save_tensor_image(output["completed"][0], completed_dir / relative_path)
                save_tensor_image(output["masked_gt"][0], masked_dir / relative_path)

            records.append(
                {
                    "relative_path": relative_path,
                    "wall_time_sec": float(elapsed),
                    "phase4a_timing_profile": output.get("phase4a_timing_profile"),
                    "safe_tome_profile": output.get("safe_tome_profile"),
                    "global_tome_profile": output.get("global_tome_profile"),
                    "boundary_split_profile": output.get("boundary_split_profile"),
                    "similarity_profile": output.get("similarity_profile"),
                }
            )
    return records


def summarize_repeat(records: list[dict[str, Any]], gpu: int) -> dict[str, Any]:
    wall_times = [record["wall_time_sec"] for record in records]
    timing_means = {}
    event_count_means = {}
    num_sampling_steps = []
    for bucket in TIMING_BUCKETS:
        values = []
        counts = []
        for record in records:
            profile = record["phase4a_timing_profile"] or {}
            timings_ms = profile.get("timings_ms", {})
            values.append(float(timings_ms.get(bucket, 0.0)))
            counts.append(int((profile.get("event_counts") or {}).get(bucket, 0)))
        timing_means[bucket] = float(np.mean(values))
        event_count_means[bucket] = float(np.mean(counts))
    for record in records:
        profile = record["phase4a_timing_profile"] or {}
        num_sampling_steps.append(int(profile.get("num_sampling_steps", 0)))
    return {
        "wall_time_sec_mean": float(np.mean(wall_times)),
        "wall_time_sec_std_within_repeat": float(np.std(wall_times, ddof=0)),
        "peak_memory_allocated_mb": float(torch.cuda.max_memory_allocated(gpu) / 1024 / 1024),
        "peak_memory_reserved_mb": float(torch.cuda.max_memory_reserved(gpu) / 1024 / 1024),
        "timings_ms_mean": timing_means,
        "event_count_mean": event_count_means,
        "num_sampling_steps_mean": float(np.mean(num_sampling_steps)),
        "records": records,
    }


def summarize_variant_repeats(repeats: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "wall_time_sec": stats([float(repeat["wall_time_sec_mean"]) for repeat in repeats]),
        "peak_memory_allocated_mb": stats([float(repeat["peak_memory_allocated_mb"]) for repeat in repeats]),
        "peak_memory_reserved_mb": stats([float(repeat["peak_memory_reserved_mb"]) for repeat in repeats]),
        "num_sampling_steps": stats([float(repeat["num_sampling_steps_mean"]) for repeat in repeats]),
        "timings_ms": {
            bucket: stats([float(repeat["timings_ms_mean"][bucket]) for repeat in repeats])
            for bucket in TIMING_BUCKETS
        },
        "event_count_mean": {
            bucket: stats([float(repeat["event_count_mean"][bucket]) for repeat in repeats])
            for bucket in TIMING_BUCKETS
        },
    }


def build_sample_group_map(sample_records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {record["sample_id"]: record for record in sample_records}


def group_membership(sample_record: dict[str, Any], report_group: str) -> bool:
    if report_group == "10-60":
        return True
    return sample_record["report_group"] == report_group


def summarize_group_repeat_metrics(
    records: list[dict[str, Any]],
    sample_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    group_summary = {}
    for group in REPORT_GROUPS:
        selected = [record for record in records if group_membership(sample_map[record["relative_path"]], group)]
        timing_stats = {}
        token_fields = {
            "original_tokens": [],
            "eligible_tokens": [],
            "removed_tokens": [],
            "merged_tokens": [],
            "restored_tokens": [],
        }
        for bucket in TIMING_BUCKETS:
            values = []
            for record in selected:
                profile = record["phase4a_timing_profile"] or {}
                values.append(float((profile.get("timings_ms") or {}).get(bucket, 0.0)))
            timing_stats[bucket] = float(np.mean(values))
        for record in selected:
            token_profile = (record.get("phase4a_timing_profile") or {}).get("token_profile") or {}
            for field in token_fields:
                token_fields[field].append(float(token_profile.get(field, 0.0)))
        group_summary[group] = {
            "num_images": len(selected),
            "wall_time_sec_mean": float(np.mean([record["wall_time_sec"] for record in selected])),
            "timings_ms_mean": timing_stats,
            "token_profile_mean": {
                field: float(np.mean(values)) if len(values) > 0 else 0.0
                for field, values in token_fields.items()
            },
        }
    return group_summary


def aggregate_group_repeat_metrics(group_repeat_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    aggregated = {}
    for group in REPORT_GROUPS:
        aggregated[group] = {
            "num_images": int(group_repeat_summaries[0][group]["num_images"]),
            "wall_time_sec": stats(
                [float(repeat[group]["wall_time_sec_mean"]) for repeat in group_repeat_summaries]
            ),
            "timings_ms": {
                bucket: stats(
                    [float(repeat[group]["timings_ms_mean"][bucket]) for repeat in group_repeat_summaries]
                )
                for bucket in TIMING_BUCKETS
            },
            "token_profile": {
                field: stats(
                    [float(repeat[group]["token_profile_mean"][field]) for repeat in group_repeat_summaries]
                )
                for field in group_repeat_summaries[0][group]["token_profile_mean"].keys()
            },
        }
    return aggregated


def tensor_from_png(path: Path) -> torch.Tensor:
    arr = np.array(Image.open(path).convert("RGB"), dtype=np.float32)
    return torch.from_numpy(arr).permute(2, 0, 1)


def rgb_tensor_to_gray_batch(image: torch.Tensor) -> torch.Tensor:
    arr = image.permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return torch.from_numpy(gray).unsqueeze(0).unsqueeze(0)


def summarize_metric_records(
    records: list[dict[str, Any]],
    sample_map: dict[str, dict[str, Any]],
    metric_keys: list[str],
) -> dict[str, Any]:
    summary = {}
    for group in REPORT_GROUPS:
        selected = [record for record in records if group_membership(sample_map[record["relative_path"]], group)]
        summary[group] = {"num_images": len(selected)}
        for key in metric_keys:
            summary[group][key] = stats([float(record[key]) for record in selected])
    return summary


def compute_single_output_metrics(
    dataset,
    completed_dir: Path,
    sample_map: dict[str, dict[str, Any]],
    lpips_model,
    device: torch.device,
) -> dict[str, Any]:
    from image_synthesis.utils.cal_metrics import get_PSNR, get_SSIM

    records = []
    with torch.no_grad():
        for idx in range(len(dataset)):
            sample = dataset[idx]
            relative_path = sample["relative_path"]
            gt_rgb = sample["image"].float()
            pred_rgb = tensor_from_png(completed_dir / relative_path).float()
            gt_gray = rgb_tensor_to_gray_batch(gt_rgb)
            pred_gray = rgb_tensor_to_gray_batch(pred_rgb)
            gt_lpips = (gt_rgb.unsqueeze(0).to(device) / 255.0) * 2.0 - 1.0
            pred_lpips = (pred_rgb.unsqueeze(0).to(device) / 255.0) * 2.0 - 1.0
            lpips_value = float(lpips_model(gt_lpips, pred_lpips).mean().detach().cpu())
            records.append(
                {
                    "relative_path": relative_path,
                    "psnr": float(get_PSNR(gt_gray, pred_gray, tool="skimage")),
                    "ssim": float(get_SSIM(gt_gray, pred_gray, full=False, win_size=51)),
                    "lpips_output_vs_gt": lpips_value,
                }
            )
    return {
        "records": records,
        "groups": summarize_metric_records(
            records=records,
            sample_map=sample_map,
            metric_keys=["psnr", "ssim", "lpips_output_vs_gt"],
        ),
    }


def generate_diversity_outputs(
    model,
    dataset,
    device: torch.device,
    args: argparse.Namespace,
    output_dir: Path,
    variant_index: int,
) -> None:
    cached_timing_state = getattr(model, "phase4a_timing_enabled", False)
    model.phase4a_timing_enabled = False
    try:
        with torch.no_grad():
            for image_index in range(len(dataset)):
                sample = dataset[image_index]
                relative_path = sample["relative_path"]
                sample_dir = output_dir / Path(relative_path).stem
                sample_dir.mkdir(parents=True, exist_ok=True)
                batch = {
                    "relative_path": [relative_path],
                    "image": sample["image"].unsqueeze(0),
                    "mask": sample["mask"].unsqueeze(0),
                }
                for sample_index in range(args.diversity_num_samples):
                    sample_seed = (
                        int(args.seed)
                        + int(variant_index) * 100000
                        + int(image_index) * 100
                        + int(sample_index)
                    )
                    set_seed(sample_seed)
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
                    save_tensor_image(output["completed"][0], sample_dir / f"completed_{sample_index:02d}.png")
                    torch.cuda.synchronize(device)
    finally:
        model.phase4a_timing_enabled = cached_timing_state


def lpips_tensor_from_png(path: Path, device: torch.device) -> torch.Tensor:
    arr = np.array(Image.open(path).convert("RGB"), dtype=np.float32)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    return (tensor / 255.0) * 2.0 - 1.0


def compute_diversity_metrics(
    diversity_dir: Path,
    sample_map: dict[str, dict[str, Any]],
    lpips_model,
    device: torch.device,
) -> dict[str, Any]:
    stem_to_name = {Path(name).stem: name for name in sample_map.keys()}
    records = []
    with torch.no_grad():
        for sample_dir in sorted(path for path in diversity_dir.iterdir() if path.is_dir()):
            image_paths = sorted(sample_dir.glob("*.png"))
            pair_values = []
            for left, right in itertools.combinations(image_paths, 2):
                value = float(
                    lpips_model(
                        lpips_tensor_from_png(left, device),
                        lpips_tensor_from_png(right, device),
                    ).mean().detach().cpu()
                )
                pair_values.append(value)
            relative_path = stem_to_name[sample_dir.name]
            records.append(
                {
                    "relative_path": relative_path,
                    "lpips_pairwise_diversity": float(np.mean(pair_values)) if pair_values else 0.0,
                    "num_pairs": len(pair_values),
                }
            )
    return {
        "records": records,
        "groups": summarize_metric_records(
            records=records,
            sample_map=sample_map,
            metric_keys=["lpips_pairwise_diversity"],
        ),
    }


def transformer_core_macs(num_tokens: int, dim: int, hidden_dim: int, num_layers: int) -> int:
    qkv = num_tokens * dim * (3 * dim)
    attn_qk = num_tokens * num_tokens * dim
    attn_v = num_tokens * num_tokens * dim
    proj = num_tokens * dim * dim
    fc1 = num_tokens * dim * hidden_dim
    fc2 = num_tokens * hidden_dim * dim
    return int(num_layers * (qkv + attn_qk + attn_v + proj + fc1 + fc2))


def compute_group_complexity(
    artifact_records: list[dict[str, Any]],
    sample_map: dict[str, dict[str, Any]],
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
        total_macs = transformer_core_macs(tokens_in_blocks, dim=dim, hidden_dim=hidden_dim, num_layers=num_layers) * steps
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
        selected = [record for record in per_image if group_membership(sample_map[record["relative_path"]], group)]
        out[group] = {
            "num_images": len(selected),
            "core_gmacs": stats([record["core_gmacs"] for record in selected]),
            "core_gflops": stats([record["core_gflops"] for record in selected]),
            "tokens_in_blocks": stats([record["tokens_in_blocks"] for record in selected]),
            "per_image": selected,
        }
    return out


def boundary_ring(mask_path: Path) -> np.ndarray:
    mask = np.array(Image.open(mask_path).convert("L")) > 127
    padded = np.pad(mask, 1, mode="edge")
    neighbors = []
    for y in range(3):
        for x in range(3):
            neighbors.append(padded[y : y + mask.shape[0], x : x + mask.shape[1]])
    stacked = np.stack(neighbors, axis=0)
    return stacked.max(axis=0) != stacked.min(axis=0)


def overlay_boundary(image: Image.Image, ring: np.ndarray) -> Image.Image:
    arr = np.array(image).copy()
    arr[ring] = (255, 32, 32)
    return Image.fromarray(arr)


def crop_around_boundary(image: Image.Image, ring: np.ndarray, pad: int) -> Image.Image:
    ys, xs = np.where(ring)
    if len(xs) == 0:
        return image
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, image.width)
    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad + 1, image.height)
    return image.crop((x0, y0, x1, y1))


def resize_cell(image: Image.Image, size: int) -> Image.Image:
    return image.resize((size, size), Image.Resampling.BICUBIC)


def add_banner(image: Image.Image, label: str) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    draw.rectangle((0, 0, out.width, 26), fill=(0, 0, 0))
    draw.text((6, 4), label, fill=(255, 255, 255), font=font)
    return out


def select_boundary_visual_samples(sample_records: list[dict[str, Any]]) -> list[str]:
    selected = []
    for bucket in [spec["bucket"] for spec in MASK_BUCKET_SPECS]:
        bucket_names = [record["sample_id"] for record in sample_records if record["bucket"] == bucket]
        if len(bucket_names) == 0:
            continue
        selected.append(bucket_names[len(bucket_names) // 2])
    return selected


def build_boundary_visuals(
    image_dir: Path,
    mask_dir: Path,
    boundary_dir: Path,
    final_path: Path,
    variant_outputs: list[dict[str, Any]],
    sample_names: list[str],
    cell: int,
    pad: int,
) -> None:
    rows = []
    boundary_dir.mkdir(parents=True, exist_ok=True)
    for name in sample_names:
        ring = boundary_ring(mask_dir / name)
        cells = [
            add_banner(
                resize_cell(
                    crop_around_boundary(
                        overlay_boundary(Image.open(image_dir / name).convert("RGB"), ring),
                        ring,
                        pad,
                    ),
                    cell,
                ),
                "GT crop",
            ),
            add_banner(
                resize_cell(crop_around_boundary(Image.open(mask_dir / name).convert("RGB"), ring, pad), cell),
                "Mask crop",
            ),
        ]
        for variant in variant_outputs:
            image = Image.open(Path(variant["completed_dir"]) / name).convert("RGB")
            crop = crop_around_boundary(overlay_boundary(image, ring), ring, pad)
            cells.append(add_banner(resize_cell(crop, cell), variant["label"]))
        row = Image.new("RGB", (cell * len(cells), cell), (255, 255, 255))
        for idx, cell_image in enumerate(cells):
            row.paste(cell_image, (idx * cell, 0))
        row_path = boundary_dir / f"{Path(name).stem}_boundary.png"
        row.save(row_path)
        rows.append(row)

    figure = Image.new("RGB", (rows[0].width, sum(row.height for row in rows)), (255, 255, 255))
    y_offset = 0
    for row in rows:
        figure.paste(row, (0, y_offset))
        y_offset += row.height
    ensure_parent(final_path)
    figure.save(final_path)


def to_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(key): to_serializable(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(value) for value in obj]
    if isinstance(obj, tuple):
        return [to_serializable(value) for value in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def build_comparison_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for variant in summary["variants"]:
        for group in REPORT_GROUPS:
            row = {
                "variant": variant["label"],
                "variant_name": variant["name"],
                "mask_group": group,
                "num_images": int(variant["group_timing_summary"][group]["num_images"]),
                "PSNR_mean": variant["quality"]["groups"][group]["psnr"]["mean"],
                "PSNR_min": variant["quality"]["groups"][group]["psnr"]["min"],
                "PSNR_max": variant["quality"]["groups"][group]["psnr"]["max"],
                "PSNR_std": variant["quality"]["groups"][group]["psnr"]["std"],
                "SSIM_mean": variant["quality"]["groups"][group]["ssim"]["mean"],
                "SSIM_min": variant["quality"]["groups"][group]["ssim"]["min"],
                "SSIM_max": variant["quality"]["groups"][group]["ssim"]["max"],
                "SSIM_std": variant["quality"]["groups"][group]["ssim"]["std"],
                "LPIPS_output_vs_gt_mean": variant["quality"]["groups"][group]["lpips_output_vs_gt"]["mean"],
                "LPIPS_output_vs_gt_min": variant["quality"]["groups"][group]["lpips_output_vs_gt"]["min"],
                "LPIPS_output_vs_gt_max": variant["quality"]["groups"][group]["lpips_output_vs_gt"]["max"],
                "LPIPS_output_vs_gt_std": variant["quality"]["groups"][group]["lpips_output_vs_gt"]["std"],
                "LPIPS_pairwise_diversity_mean": (
                    variant["diversity"]["groups"][group]["lpips_pairwise_diversity"]["mean"]
                    if variant["diversity"] is not None
                    else ""
                ),
                "LPIPS_pairwise_diversity_min": (
                    variant["diversity"]["groups"][group]["lpips_pairwise_diversity"]["min"]
                    if variant["diversity"] is not None
                    else ""
                ),
                "LPIPS_pairwise_diversity_max": (
                    variant["diversity"]["groups"][group]["lpips_pairwise_diversity"]["max"]
                    if variant["diversity"] is not None
                    else ""
                ),
                "LPIPS_pairwise_diversity_std": (
                    variant["diversity"]["groups"][group]["lpips_pairwise_diversity"]["std"]
                    if variant["diversity"] is not None
                    else ""
                ),
                "end_to_end_latency_mean_sec": variant["group_timing_summary"][group]["wall_time_sec"]["mean"],
                "end_to_end_latency_std_sec": variant["group_timing_summary"][group]["wall_time_sec"]["std"],
                "peak_memory_mean_mb": variant["timing_summary"]["peak_memory_allocated_mb"]["mean"],
                "peak_memory_std_mb": variant["timing_summary"]["peak_memory_allocated_mb"]["std"],
                "transformer_core_GMACs_mean": variant["complexity"][group]["core_gmacs"]["mean"],
                "transformer_core_GMACs_std": variant["complexity"][group]["core_gmacs"]["std"],
                "transformer_core_GFLOPs_mean": variant["complexity"][group]["core_gflops"]["mean"],
                "transformer_core_GFLOPs_std": variant["complexity"][group]["core_gflops"]["std"],
                "tokens_in_blocks_mean": variant["complexity"][group]["tokens_in_blocks"]["mean"],
                "tokens_in_blocks_std": variant["complexity"][group]["tokens_in_blocks"]["std"],
            }
            for bucket in TIMING_BUCKETS:
                row[f"{bucket}_mean_ms"] = variant["group_timing_summary"][group]["timings_ms"][bucket]["mean"]
                row[f"{bucket}_std_ms"] = variant["group_timing_summary"][group]["timings_ms"][bucket]["std"]
            rows.append(row)
    return rows


def write_comparison_csv(rows: list[dict[str, Any]], path: Path) -> None:
    ensure_parent(path)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--put-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--validation-list", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--mask-root", required=True)
    parser.add_argument("--input-res", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--measure-runs", type=int, default=3)
    parser.add_argument("--num-token-per-iter", type=int, default=20)
    parser.add_argument("--num-token-for-sampling", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260423)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.15)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--topk-ratio", type=float, default=0.25)
    parser.add_argument("--boundary-ring-radius", type=int, default=1)
    parser.add_argument("--count-10-20", type=int, default=50)
    parser.add_argument("--count-20-30", type=int, default=50)
    parser.add_argument("--count-30-40", type=int, default=50)
    parser.add_argument("--count-40-50", type=int, default=50)
    parser.add_argument("--count-50-60", type=int, default=50)
    parser.add_argument("--diversity-num-samples", type=int, default=4)
    parser.add_argument("--skip-diversity", action="store_true", default=False)
    parser.add_argument("--boundary-zoom-cell", type=int, default=192)
    parser.add_argument("--boundary-zoom-pad", type=int, default=48)
    args = parser.parse_args()

    put_root = Path(args.put_root).resolve()
    sys.path.insert(0, str(put_root))
    os.chdir(put_root)

    from scripts.inference import ImagePathDataset
    import lpips

    input_res = parse_hw(args.input_res)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")

    validation_images = resolve_validation_images(
        validation_list=Path(args.validation_list),
        image_root=Path(args.image_root),
    )
    mask_catalog = build_mask_catalog(mask_root=Path(args.mask_root), input_res=input_res)
    bucket_counts = parse_bucket_counts(args)
    sample_records, subset_image_dir, subset_mask_dir = build_standardized_subset(
        output_dir=output_dir,
        validation_images=validation_images,
        mask_catalog=mask_catalog,
        bucket_counts=bucket_counts,
        input_res=input_res,
        seed=args.seed,
    )
    sample_manifest_path = output_dir / "sample_manifest.csv"
    write_sample_manifest(sample_records, sample_manifest_path)
    sample_map = build_sample_group_map(sample_records)

    dataset = ImagePathDataset(str(subset_image_dir), str(subset_mask_dir), size=input_res)
    variants = build_variants()
    subset_summary = summarize_subset(sample_records)
    total_passes = args.warmup_runs + args.measure_runs + 1
    if not args.skip_diversity:
        total_passes += args.diversity_num_samples
    estimated_seconds = len(variants) * len(dataset) * max(total_passes, 1) * 0.22
    print(f"estimated runtime about {estimated_seconds / 60.0:.1f} minutes")

    config_manifest = {
        "phase": "Phase 5A standardized comparison",
        "seed": int(args.seed),
        "checkpoint": args.checkpoint,
        "put_root": str(put_root),
        "validation_list": str(Path(args.validation_list).resolve()),
        "image_root": str(Path(args.image_root).resolve()),
        "mask_root": str(Path(args.mask_root).resolve()),
        "mask_protocol": {
            "official_source": "NVIDIA PConv test_mask/testing_mask_dataset",
            "raw_mask_convention": "white=missing, black=visible",
            "saved_mask_convention": "white=visible, black=missing",
            "report_groups": list(REPORT_GROUPS),
            "bucket_counts": bucket_counts,
        },
        "input_res": list(input_res),
        "num_images": len(dataset),
        "warmup_runs": int(args.warmup_runs),
        "measure_runs": int(args.measure_runs),
        "num_token_per_iter": int(args.num_token_per_iter),
        "num_token_for_sampling": int(args.num_token_for_sampling),
        "diversity_num_samples": 0 if args.skip_diversity else int(args.diversity_num_samples),
        "subset_summary": subset_summary,
        "variants": variants,
    }
    (output_dir / "config_manifest.json").write_text(
        json.dumps(to_serializable(config_manifest), indent=2),
        encoding="utf-8",
    )

    all_results = {
        "config_manifest": config_manifest,
        "sample_manifest_csv": str(sample_manifest_path.resolve()),
        "variants": [],
    }

    lpips_model = lpips.LPIPS(net="vgg", spatial=True).to(device)
    for variant_index, spec in enumerate(variants):
        variant_dir = output_dir / spec["name"]
        completed_dir = variant_dir / "completed_single"
        masked_dir = variant_dir / "masked_gt"
        diversity_dir = variant_dir / "diversity"
        completed_dir.mkdir(parents=True, exist_ok=True)
        masked_dir.mkdir(parents=True, exist_ok=True)
        if not args.skip_diversity:
            diversity_dir.mkdir(parents=True, exist_ok=True)

        print(f"[variant] {spec['label']}")
        set_variant_env(spec, args)
        set_seed(args.seed)
        model = load_model(put_root, args.checkpoint, device)
        dim = int(model.dim)
        hidden_dim = int(model.blocks[0].mlp.fc1.out_features)
        num_layers = int(len(model.blocks))

        for warmup_index in range(args.warmup_runs):
            _ = run_dataset_pass(
                model=model,
                dataset=dataset,
                device=device,
                args=args,
                save_outputs=False,
                completed_dir=None,
                masked_dir=None,
            )
            torch.cuda.synchronize(device)
            print(f"  warmup {warmup_index + 1}/{args.warmup_runs}")

        repeat_summaries = []
        group_repeat_summaries = []
        for repeat_index in range(args.measure_runs):
            set_seed(args.seed)
            torch.cuda.synchronize(device)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(args.gpu)
            records = run_dataset_pass(
                model=model,
                dataset=dataset,
                device=device,
                args=args,
                save_outputs=False,
                completed_dir=None,
                masked_dir=None,
            )
            repeat_summary = summarize_repeat(records, args.gpu)
            repeat_summary["repeat_index"] = repeat_index
            repeat_summaries.append(repeat_summary)
            group_repeat_summaries.append(summarize_group_repeat_metrics(records, sample_map))
            print(
                f"  repeat {repeat_index + 1}/{args.measure_runs}: "
                f"wall={repeat_summary['wall_time_sec_mean']:.4f}s/im, "
                f"peak={repeat_summary['peak_memory_allocated_mb']:.2f}MiB"
            )

        set_seed(args.seed)
        artifact_records = run_dataset_pass(
            model=model,
            dataset=dataset,
            device=device,
            args=args,
            save_outputs=True,
            completed_dir=completed_dir,
            masked_dir=masked_dir,
        )
        quality = compute_single_output_metrics(
            dataset=dataset,
            completed_dir=completed_dir,
            sample_map=sample_map,
            lpips_model=lpips_model,
            device=device,
        )

        diversity = None
        if not args.skip_diversity:
            generate_diversity_outputs(
                model=model,
                dataset=dataset,
                device=device,
                args=args,
                output_dir=diversity_dir,
                variant_index=variant_index,
            )
            diversity = compute_diversity_metrics(
                diversity_dir=diversity_dir,
                sample_map=sample_map,
                lpips_model=lpips_model,
                device=device,
            )

        variant_result = {
            "name": spec["name"],
            "label": spec["label"],
            "completed_dir": str(completed_dir.resolve()),
            "masked_dir": str(masked_dir.resolve()),
            "diversity_dir": str(diversity_dir.resolve()) if diversity is not None else None,
            "repeat_summaries": repeat_summaries,
            "timing_summary": summarize_variant_repeats(repeat_summaries),
            "group_timing_summary": aggregate_group_repeat_metrics(group_repeat_summaries),
            "quality": quality,
            "diversity": diversity,
            "complexity": compute_group_complexity(
                artifact_records=artifact_records,
                sample_map=sample_map,
                dim=dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
            ),
            "artifact_records": artifact_records,
        }
        all_results["variants"].append(variant_result)
        del model
        torch.cuda.empty_cache()

    del lpips_model
    torch.cuda.empty_cache()

    boundary_names = select_boundary_visual_samples(sample_records)
    boundary_outputs = [
        {"label": variant["label"], "completed_dir": variant["completed_dir"]}
        for variant in all_results["variants"]
    ]
    build_boundary_visuals(
        image_dir=subset_image_dir,
        mask_dir=subset_mask_dir,
        boundary_dir=output_dir / "boundary_visuals",
        final_path=output_dir / "final_phase5a_boundary_comparison.png",
        variant_outputs=boundary_outputs,
        sample_names=boundary_names,
        cell=args.boundary_zoom_cell,
        pad=args.boundary_zoom_pad,
    )
    all_results["boundary_visual_sample_ids"] = boundary_names
    all_results["boundary_comparison_path"] = str(
        (output_dir / "final_phase5a_boundary_comparison.png").resolve()
    )

    comparison_rows = build_comparison_rows(all_results)
    write_comparison_csv(comparison_rows, output_dir / "comparison_table.csv")
    (output_dir / "phase5a_summary.json").write_text(
        json.dumps(to_serializable(all_results), indent=2),
        encoding="utf-8",
    )
    print(f"saved summary to {output_dir / 'phase5a_summary.json'}")


if __name__ == "__main__":
    main()
