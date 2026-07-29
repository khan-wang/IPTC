#!/usr/bin/env python3
"""Phase 7 generic 256x256 protocol evaluation for FFHQ / ImageNet / Places2."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

import phase5a_standardized_comparison as phase5a
import phase5e_ga_spg_probe as phase5e
import phase5f_ga_spg_overhead_attribution as phase5f
import phase7_main256_places2_full_eval as phase7_places2


REPO_ROOT = Path(__file__).resolve().parents[1]
MASK_BUCKET_SPECS = (
    {"bucket": "01-10", "min_ratio": 0.01, "max_ratio": 0.10, "report_group": "0.01-20"},
    {"bucket": "10-20", "min_ratio": 0.10, "max_ratio": 0.20, "report_group": "0.01-20"},
    {"bucket": "20-30", "min_ratio": 0.20, "max_ratio": 0.30, "report_group": "20-40"},
    {"bucket": "30-40", "min_ratio": 0.30, "max_ratio": 0.40, "report_group": "20-40"},
    {"bucket": "40-50", "min_ratio": 0.40, "max_ratio": 0.50, "report_group": "40-60"},
    {"bucket": "50-60", "min_ratio": 0.50, "max_ratio": 0.60, "report_group": "40-60"},
)
BUCKET_ORDER = tuple(spec["bucket"] for spec in MASK_BUCKET_SPECS)
REPORT_GROUPS = ("0.01-20", "20-40", "40-60")
OVERALL_GROUP = "0.01-60"
SUMMARY_GROUPS = REPORT_GROUPS + (OVERALL_GROUP,)
PAIRWISE_COMPARISONS = (
    {
        "pair_name": "SBVC vs PureDistance",
        "method_a_name": "sbvc_r224_zero_pad_fastpath",
        "method_b_name": "pure_distance_r224",
    },
    {
        "pair_name": "SBVC vs PUT Baseline",
        "method_a_name": "sbvc_r224_zero_pad_fastpath",
        "method_b_name": "put_baseline",
    },
    {
        "pair_name": "Global ToMe vs PUT Baseline",
        "method_a_name": "global_tome_r224",
        "method_b_name": "put_baseline",
    },
)
DELTA_METRICS = (
    ("psnr", True),
    ("ssim", True),
    ("lpips", False),
    ("latency_sec", False),
    ("cuda_total_ms", False),
    ("core_gmacs", False),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--put-root", default=str(REPO_ROOT / "third_party" / "PUT"))
    checkpoint_default = os.environ.get("PUT_CHECKPOINT", "")
    image_root_default = os.environ.get("SBVC_IMAGE_ROOT", "")
    mask_root_default = os.environ.get("SBVC_MASK_ROOT", "")
    parser.add_argument("--checkpoint", default=checkpoint_default, required=not checkpoint_default)
    parser.add_argument("--validation-list", default="")
    parser.add_argument("--image-root", default=image_root_default, required=not image_root_default)
    parser.add_argument("--mask-root", default=mask_root_default, required=not mask_root_default)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reuse-existing-subset-dir", default="")
    parser.add_argument("--freeze-only", action="store_true", default=False)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--validation-source-label", default="")
    parser.add_argument(
        "--image-selection-mode",
        choices=("auto", "validation_list", "recursive_image_root_scan"),
        default="auto",
    )
    parser.add_argument(
        "--mask-selection-mode",
        choices=("unique_spaced", "deterministic_cycle"),
        default="unique_spaced",
    )
    parser.add_argument("--input-res", default="256,256")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260428)
    parser.add_argument("--samples-per-bucket", type=int, default=0)
    parser.add_argument("--count-01-10", type=int, default=0)
    parser.add_argument("--count-10-20", type=int, default=0)
    parser.add_argument("--count-20-30", type=int, default=0)
    parser.add_argument("--count-30-40", type=int, default=0)
    parser.add_argument("--count-40-50", type=int, default=0)
    parser.add_argument("--count-50-60", type=int, default=0)
    parser.add_argument("--warmup-images", type=int, default=16)
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
    parser.add_argument("--methods", default="put_baseline")
    parser.add_argument("--skip-fid", action="store_true", default=False)
    parser.add_argument("--resume-existing", action="store_true", default=False)
    parser.add_argument("--fid-batch-size", type=int, default=64)
    parser.add_argument("--fid-num-workers", type=int, default=4)
    return parser.parse_args()


def read_nonempty_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def resolve_validation_images_generic(validation_list: Path, image_root: Path) -> list[dict[str, Any]]:
    relpaths = read_nonempty_lines(validation_list)
    records = []
    missing = []
    for relpath in relpaths:
        candidate_relpath = image_root / relpath
        candidate_basename = image_root / Path(relpath).name
        if candidate_relpath.is_file():
            source_path = candidate_relpath
            image_uid = relpath
        elif candidate_basename.is_file():
            source_path = candidate_basename
            image_uid = Path(relpath).name
        else:
            missing.append(relpath)
            continue
        records.append(
            {
                "validation_relpath": relpath,
                "image_uid": image_uid,
                "source_image_path": str(source_path.resolve()),
            }
        )
    if missing:
        preview = ", ".join(missing[:5])
        suffix = "" if len(missing) <= 5 else f" ... (+{len(missing) - 5} more)"
        raise FileNotFoundError(
            f"{validation_list} has {len(missing)} unresolved entries under {image_root}: {preview}{suffix}"
        )
    return records


def resolve_image_records(
    *,
    image_root: Path,
    validation_list: Path | None,
    validation_source_label: str,
    image_selection_mode: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    use_validation_list = False
    if image_selection_mode == "validation_list":
        use_validation_list = True
    elif image_selection_mode == "recursive_image_root_scan":
        use_validation_list = False
    else:
        use_validation_list = validation_list is not None and validation_list.is_file()

    if use_validation_list:
        if validation_list is None or not validation_list.is_file():
            raise FileNotFoundError(f"validation list is required but missing: {validation_list}")
        records = resolve_validation_images_generic(validation_list=validation_list, image_root=image_root)
        return records, {
            "source": validation_source_label or str(validation_list),
            "available_validation_images": len(records),
            "is_official_val": True,
            "is_local_subset": True,
            "selection_mode": "official_validation_list",
            "path_resolution": "relpath_then_basename_fallback",
        }

    records = phase7_places2.resolve_places2_images(image_root)
    return records, {
        "source": validation_source_label or str(image_root),
        "available_validation_images": len(records),
        "is_official_val": False,
        "is_local_subset": False,
        "selection_mode": "recursive_image_root_scan",
        "path_resolution": "full_relpath_scan",
    }


def bucket_spec_by_name(bucket: str) -> dict[str, Any]:
    for spec in MASK_BUCKET_SPECS:
        if spec["bucket"] == bucket:
            return spec
    raise KeyError(bucket)


def ratio_in_bucket(ratio: float, bucket: str) -> bool:
    spec = bucket_spec_by_name(bucket)
    lower = float(spec["min_ratio"])
    upper = float(spec["max_ratio"])
    if bucket == "50-60":
        return lower <= ratio <= upper + 1.0e-8
    return lower <= ratio < upper


def build_mask_catalog_rebinned(mask_root: Path, input_res: tuple[int, int]) -> list[dict[str, Any]]:
    catalog = []
    for path in sorted(mask_root.glob("*.png")):
        mask = phase5a.Image.open(path).convert("L").resize((input_res[1], input_res[0]), phase5a.Image.Resampling.NEAREST)
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


def parse_bucket_counts(args: argparse.Namespace, mask_catalog_counts: dict[str, int]) -> dict[str, int]:
    explicit = {
        "01-10": int(args.count_01_10),
        "10-20": int(args.count_10_20),
        "20-30": int(args.count_20_30),
        "30-40": int(args.count_30_40),
        "40-50": int(args.count_40_50),
        "50-60": int(args.count_50_60),
    }
    if any(value > 0 for value in explicit.values()):
        counts = {}
        for bucket in BUCKET_ORDER:
            requested = explicit[bucket]
            counts[bucket] = requested if requested > 0 else int(mask_catalog_counts[bucket])
        return counts
    if int(args.samples_per_bucket) > 0:
        return {bucket: int(args.samples_per_bucket) for bucket in BUCKET_ORDER}
    return {bucket: int(mask_catalog_counts[bucket]) for bucket in BUCKET_ORDER}


def summarize_subset_rebinned(sample_records: list[dict[str, Any]]) -> dict[str, Any]:
    group_counts = {group: 0 for group in SUMMARY_GROUPS}
    bucket_counts = {spec["bucket"]: 0 for spec in MASK_BUCKET_SPECS}
    ratios_all: list[float] = []
    ratios_by_group = {group: [] for group in SUMMARY_GROUPS}
    for record in sample_records:
        ratio = float(record["mask_ratio"])
        bucket = str(record["bucket"])
        group = str(record["report_group"])
        bucket_counts[bucket] += 1
        group_counts[group] += 1
        group_counts[OVERALL_GROUP] += 1
        ratios_all.append(ratio)
        ratios_by_group[group].append(ratio)
        ratios_by_group[OVERALL_GROUP].append(ratio)
    out = {
        "num_samples": len(sample_records),
        "bucket_counts": bucket_counts,
        "report_group_counts": group_counts,
    }
    for group in SUMMARY_GROUPS:
        key = f"mask_ratio_{group.replace('.', 'p')}"
        values = ratios_by_group[group] if group != OVERALL_GROUP else ratios_all
        out[key] = phase5a.stats(values)
    return out


def select_masks_for_bucket(
    *,
    bucket_masks: list[dict[str, Any]],
    bucket_count: int,
    mode: str,
) -> list[dict[str, Any]]:
    if not bucket_masks:
        raise RuntimeError("no masks available for requested bucket")
    ordered = sorted(bucket_masks, key=lambda item: item["mask_ratio"])
    if mode == "unique_spaced":
        return phase5a.select_spaced(ordered, bucket_count, key="mask_ratio")
    if mode != "deterministic_cycle":
        raise ValueError(f"unsupported mask selection mode: {mode}")
    if bucket_count <= len(ordered):
        return phase5a.select_spaced(ordered, bucket_count, key="mask_ratio")
    repeats = int(math.ceil(bucket_count / len(ordered)))
    tiled = ordered * repeats
    return tiled[:bucket_count]


def build_standardized_subset_with_mask_mode(
    *,
    output_dir: Path,
    validation_images: list[dict[str, Any]],
    mask_catalog: list[dict[str, Any]],
    bucket_counts: dict[str, int],
    input_res: tuple[int, int],
    seed: int,
    mask_selection_mode: str,
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
        selected_masks = select_masks_for_bucket(
            bucket_masks=bucket_masks,
            bucket_count=bucket_count,
            mode=mask_selection_mode,
        )
        bucket_images = selected_images[cursor : cursor + bucket_count]
        cursor += bucket_count

        for image_info, mask_info in zip(bucket_images, selected_masks):
            sample_name = f"sample_{sample_index:05d}_{bucket.replace('-', '_')}.png"
            output_image_path = image_dir / sample_name
            output_mask_path = mask_dir / sample_name
            phase5a.save_resized_rgb(Path(image_info["source_image_path"]), output_image_path, size=input_res)
            saved_mask_ratio = phase5a.save_put_mask_from_pconv(
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


def read_sample_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            row["mask_ratio"] = float(row["mask_ratio"])
            rows.append(dict(row))
    return rows


def selected_method_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    requested = [item.strip() for item in str(args.methods).split(",") if item.strip()]
    available = {spec["name"]: spec for spec in phase7_places2.method_specs(args)}
    if not requested:
        requested = ["put_baseline"]
    unknown = [name for name in requested if name not in available]
    if unknown:
        raise ValueError(f"unknown methods: {unknown}; available={sorted(available)}")
    return [available[name] for name in requested]


def write_dataset_scope_md(
    *,
    path: Path,
    dataset_name: str,
    image_root: Path,
    validation_list: Path | None,
    image_scope: dict[str, Any],
    sample_records: list[dict[str, Any]],
    subset_summary: dict[str, Any],
    mask_selection_mode: str,
    mask_catalog_counts: dict[str, int],
) -> None:
    excluded_mask_count = int(image_scope.get("excluded_mask_count", 0))
    lines = [
        "# Dataset Scope",
        "",
        f"- Dataset: `{dataset_name}`",
        f"- Actual image count: `{len(sample_records)}`",
        f"- Validation source: `{image_scope['source']}`",
        f"- Validation list path: `{validation_list}`" if validation_list is not None else "- Validation list path: `none`",
        f"- Image root: `{image_root}`",
        f"- Available validation images in list: `{image_scope['available_validation_images']}`",
        f"- Official val: `{'yes' if image_scope['is_official_val'] else 'no'}`",
        f"- Local available subset: `{'yes' if image_scope['is_local_subset'] else 'no'}`",
        f"- Selection mode: `{image_scope['selection_mode']}`",
        f"- Path resolution: `{image_scope['path_resolution']}`",
        f"- Reused subset from: `{image_scope.get('reused_subset_from', '') or 'none'}`",
        f"- Mask assignment mode: `{mask_selection_mode}`",
        f"- Bucket mask catalog counts: `01-10={mask_catalog_counts['01-10']}, 10-20={mask_catalog_counts['10-20']}, 20-30={mask_catalog_counts['20-30']}, 30-40={mask_catalog_counts['30-40']}, 40-50={mask_catalog_counts['40-50']}, 50-60={mask_catalog_counts['50-60']}`",
        f"- Excluded masks outside 0.01-60 after resize: `{excluded_mask_count}`",
        "",
        "## Mask Ratio Distribution",
        "",
        f"- 0.01-10: `{subset_summary['bucket_counts']['01-10']}`",
        f"- 10-20: `{subset_summary['bucket_counts']['10-20']}`",
        f"- 20-30: `{subset_summary['bucket_counts']['20-30']}`",
        f"- 30-40: `{subset_summary['bucket_counts']['30-40']}`",
        f"- 40-50: `{subset_summary['bucket_counts']['40-50']}`",
        f"- 50-60: `{subset_summary['bucket_counts']['50-60']}`",
        f"- 0.01-20 total: `{subset_summary['report_group_counts']['0.01-20']}`",
        f"- 20-40 total: `{subset_summary['report_group_counts']['20-40']}`",
        f"- 40-60 total: `{subset_summary['report_group_counts']['40-60']}`",
        f"- 0.01-60 total: `{subset_summary['report_group_counts']['0.01-60']}`",
    ]
    if mask_selection_mode == "deterministic_cycle":
        lines.extend(
            [
                "",
                "## Protocol Note",
                "",
                "This run explicitly accepts deterministic mask reuse / cycling within each mask-ratio bucket.",
                "Masks are ordered by realized ratio within each bucket and repeated deterministically until the requested sample count is reached.",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_sample_ids_by_group(sample_records: list[dict[str, Any]]) -> dict[str, list[str]]:
    out = {group: [] for group in SUMMARY_GROUPS}
    for sample in sample_records:
        sample_id = sample["sample_id"]
        report_group = sample["report_group"]
        if report_group in out:
            out[report_group].append(sample_id)
        out[OVERALL_GROUP].append(sample_id)
    return out


def write_file_list(path: Path, items: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(items) + "\n", encoding="utf-8")


def build_fid_file_lists(output_dir: Path, sample_records: list[dict[str, Any]]) -> dict[str, Path]:
    sample_ids_by_group = build_sample_ids_by_group(sample_records)
    fid_list_dir = output_dir / "fid_filelists"
    out = {}
    for group, sample_ids in sample_ids_by_group.items():
        path = fid_list_dir / f"{group}.txt"
        write_file_list(path, sample_ids)
        out[group] = path
    return out


def group_for_sample(sample: dict[str, Any]) -> list[str]:
    return [sample["report_group"], OVERALL_GROUP]


def compute_fid_for_method_groups(
    *,
    gt_dir: Path,
    pred_dir: Path,
    file_list_by_group: dict[str, Path],
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, dict[str, Any]]:
    if args.skip_fid:
        return {
            group: {"status": "skipped", "fid": None, "pids": None, "uids": None, "length1": 0, "length2": 0}
            for group in SUMMARY_GROUPS
        }
    from scripts.metrics.cal_fid import calculate_fid_given_paths

    out = {}
    for group in SUMMARY_GROUPS:
        file_list = str(file_list_by_group[group])
        result = calculate_fid_given_paths(
            path1=str(gt_dir),
            path_prefix1="",
            file_list1=file_list,
            path2=str(pred_dir),
            path_prefix2="",
            file_list2=file_list,
            batch_size=int(args.fid_batch_size),
            device=device,
            dims=2048,
            net="inception",
            num_workers=int(args.fid_num_workers),
            max_count1=None,
            max_count2=None,
            count1=None,
            count2=None,
            im_size1=[256, 256],
            im_size2=[256, 256],
            share_same_files=True,
        )
        out[group] = {"status": "computed", **result}
    return out


def attach_group_fid(
    summary_rows: list[dict[str, Any]],
    fid_by_method_group: dict[str, dict[str, dict[str, Any]]],
    *,
    group_key: str,
) -> None:
    for row in summary_rows:
        group = OVERALL_GROUP if group_key == "dataset" else row[group_key]
        fid = (fid_by_method_group.get(row["method"]) or {}).get(group, {})
        row["fid_scope"] = group
        row["fid_status"] = fid.get("status", "not_reported")
        row["fid_mean"] = float(fid.get("fid", 0.0)) if fid.get("fid") is not None else ""
        row["fid_std"] = 0.0
        row["fid_num_images"] = int(fid.get("length2", 0)) if fid.get("length2") is not None else 0
        row["pids_mean"] = float(fid.get("pids", 0.0)) if fid.get("pids") is not None else ""
        row["uids_mean"] = float(fid.get("uids", 0.0)) if fid.get("uids") is not None else ""


def latex_from_rows_rebinned(rows: list[dict[str, Any]]) -> str:
    lines = [
        r"\\begin{tabular}{llrrrrrrr}",
        r"Mask Group & Method & PSNR & SSIM & LPIPS & FID & GMACs & Latency(s) & Actual CR \\",
        r"\\hline",
    ]
    for row in rows:
        fid_value = row.get("fid_mean", "")
        fid_text = "{:.4f}".format(float(fid_value)) if fid_value != "" else "NA"
        lines.append(
            "{group} & {method} & {psnr:.4f} & {ssim:.4f} & {lpips:.6f} & {fid} & {gmacs:.3f} & {lat:.4f} & {cr:.6f} \\\\".format(
                group=row["mask_group"],
                method=row["method"],
                psnr=float(row["psnr_mean"]),
                ssim=float(row["ssim_mean"]),
                lpips=float(row["lpips_mean"]),
                fid=fid_text,
                gmacs=float(row["core_gmacs_mean"]),
                lat=float(row["latency_sec_mean"]),
                cr=float(row["actual_compression_ratio_mean"]),
            )
        )
    lines.append(r"\\end{tabular}")
    return "\n".join(lines) + "\n"


def summarize_delta(values: list[float], higher_is_better: bool) -> dict[str, float]:
    if not values:
        return {
            "num_images": 0,
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "win_rate": 0.0,
            "loss_rate": 0.0,
        }
    arr = np.asarray(values, dtype=np.float64)
    eps = 1.0e-12
    if higher_is_better:
        wins = arr > eps
        losses = arr < -eps
    else:
        wins = arr < -eps
        losses = arr > eps
    return {
        "num_images": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "win_rate": float(wins.mean()),
        "loss_rate": float(losses.mean()),
    }


def build_pair_rows(
    per_image_metrics: list[dict[str, Any]],
    per_image_compression: list[dict[str, Any]],
    method_specs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    method_labels = {spec["name"]: spec["label"] for spec in method_specs}
    metrics_by_image_method = {
        (row["image_id"], row["method"]): row for row in per_image_metrics
    }
    compression_by_image_method = {
        (row["image_id"], row["method"]): row for row in per_image_compression
    }
    pair_rows = []
    for pair_spec in PAIRWISE_COMPARISONS:
        if pair_spec["method_a_name"] not in method_labels or pair_spec["method_b_name"] not in method_labels:
            continue
        method_a = method_labels[pair_spec["method_a_name"]]
        method_b = method_labels[pair_spec["method_b_name"]]
        image_ids = sorted({row["image_id"] for row in per_image_metrics if row["method"] == method_a})
        for image_id in image_ids:
            metric_a = metrics_by_image_method[(image_id, method_a)]
            metric_b = metrics_by_image_method[(image_id, method_b)]
            compression_a = compression_by_image_method[(image_id, method_a)]
            compression_b = compression_by_image_method[(image_id, method_b)]
            row = {
                "image_id": image_id,
                "dataset": metric_a["dataset"],
                "mask_ratio": float(metric_a["mask_ratio"]),
                "mask_group": metric_a["mask_ratio_bin"],
                "pair_name": pair_spec["pair_name"],
                "method_a": method_a,
                "method_b": method_b,
                "method_a_actual_removed_tokens": int(compression_a["actual_removed_tokens"]),
                "method_b_actual_removed_tokens": int(compression_b["actual_removed_tokens"]),
                "actual_removed_tokens_delta": int(compression_a["actual_removed_tokens"]) - int(compression_b["actual_removed_tokens"]),
                "actual_removed_tokens_match": int(int(compression_a["actual_removed_tokens"]) == int(compression_b["actual_removed_tokens"])),
                "method_a_actual_compression_ratio": float(compression_a["actual_compression_ratio"]),
                "method_b_actual_compression_ratio": float(compression_b["actual_compression_ratio"]),
                "actual_compression_ratio_delta": float(compression_a["actual_compression_ratio"]) - float(compression_b["actual_compression_ratio"]),
                "actual_compression_ratio_match": int(abs(float(compression_a["actual_compression_ratio"]) - float(compression_b["actual_compression_ratio"])) < 1.0e-12),
                "method_a_under_compression": int(bool(compression_a["under_compression"])),
                "method_b_under_compression": int(bool(compression_b["under_compression"])),
                "under_compression_match": int(bool(compression_a["under_compression"]) == bool(compression_b["under_compression"])),
                "method_a_bad_restore": int(metric_a["bad_restore"]),
                "method_b_bad_restore": int(metric_b["bad_restore"]),
                "method_a_protect_overlap": int(metric_a["protect_overlap"]),
                "method_b_protect_overlap": int(metric_b["protect_overlap"]),
            }
            for metric_name, _higher_is_better in DELTA_METRICS:
                row[f"method_a_{metric_name}"] = float(metric_a[metric_name])
                row[f"method_b_{metric_name}"] = float(metric_b[metric_name])
                row[f"{metric_name}_delta"] = float(metric_a[metric_name]) - float(metric_b[metric_name])
            pair_rows.append(row)

    summary_rows = []
    for pair_spec in PAIRWISE_COMPARISONS:
        pair_name = pair_spec["pair_name"]
        pair_group_rows = [row for row in pair_rows if row["pair_name"] == pair_name]
        if not pair_group_rows:
            continue
        for group in SUMMARY_GROUPS:
            if group == OVERALL_GROUP:
                rows = pair_group_rows
            else:
                rows = [row for row in pair_group_rows if row["mask_group"] == group]
            if not rows:
                continue
            summary = {
                "pair_name": pair_name,
                "mask_group": group,
                "num_images": len(rows),
                "method_a": rows[0]["method_a"],
                "method_b": rows[0]["method_b"],
                "actual_removed_tokens_match_rate": float(np.mean([row["actual_removed_tokens_match"] for row in rows])),
                "actual_removed_tokens_mismatch_count": int(sum(1 - row["actual_removed_tokens_match"] for row in rows)),
                "actual_compression_ratio_match_rate": float(np.mean([row["actual_compression_ratio_match"] for row in rows])),
                "actual_compression_ratio_mismatch_count": int(sum(1 - row["actual_compression_ratio_match"] for row in rows)),
                "under_compression_match_rate": float(np.mean([row["under_compression_match"] for row in rows])),
                "bad_restore_method_a_sum": int(sum(row["method_a_bad_restore"] for row in rows)),
                "bad_restore_method_b_sum": int(sum(row["method_b_bad_restore"] for row in rows)),
                "protect_overlap_method_a_sum": int(sum(row["method_a_protect_overlap"] for row in rows)),
                "protect_overlap_method_b_sum": int(sum(row["method_b_protect_overlap"] for row in rows)),
            }
            for metric_name, higher_is_better in DELTA_METRICS:
                stats = summarize_delta([float(row[f"{metric_name}_delta"]) for row in rows], higher_is_better)
                for stat_name, stat_value in stats.items():
                    summary[f"{metric_name}_delta_{stat_name}"] = stat_value
            summary_rows.append(summary)
    return pair_rows, summary_rows


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    put_root = Path(args.put_root).resolve()
    checkpoint = str(Path(args.checkpoint).resolve())
    image_root = Path(args.image_root).resolve()
    validation_list = Path(args.validation_list).resolve() if args.validation_list else None
    mask_root = Path(args.mask_root).resolve()
    reuse_existing_subset_dir = Path(args.reuse_existing_subset_dir).resolve() if args.reuse_existing_subset_dir else None
    input_res = phase5a.parse_hw(args.input_res)
    dataset_name = str(args.dataset_name)
    method_specs = selected_method_specs(args)

    image_records, image_scope = resolve_image_records(
        image_root=image_root,
        validation_list=validation_list,
        validation_source_label=str(args.validation_source_label or validation_list or image_root),
        image_selection_mode=str(args.image_selection_mode),
    )
    mask_catalog = build_mask_catalog_rebinned(mask_root, input_res)
    mask_catalog_counts = {
        bucket: len([item for item in mask_catalog if item["bucket"] == bucket])
        for bucket in BUCKET_ORDER
    }
    image_scope["excluded_mask_count"] = int(sum(1 for item in mask_catalog if item["bucket"] is None))
    bucket_counts = parse_bucket_counts(args, mask_catalog_counts)
    if any(count <= 0 for count in bucket_counts.values()):
        raise RuntimeError(f"bucket counts must be positive for all buckets: {bucket_counts}")
    sample_manifest_path = output_dir / "sample_manifest.csv"
    if reuse_existing_subset_dir is not None:
        reuse_manifest_path = reuse_existing_subset_dir / "sample_manifest.csv"
        subset_image_dir = reuse_existing_subset_dir / "subset" / "images"
        subset_mask_dir = reuse_existing_subset_dir / "subset" / "masks"
        if not reuse_manifest_path.is_file():
            raise FileNotFoundError(f"reuse manifest not found: {reuse_manifest_path}")
        if not subset_image_dir.is_dir() or not subset_mask_dir.is_dir():
            raise FileNotFoundError(
                f"reuse subset missing images or masks: images={subset_image_dir} masks={subset_mask_dir}"
            )
        sample_records = read_sample_manifest(reuse_manifest_path)
        phase5a.write_sample_manifest(sample_records, sample_manifest_path)
        subset_summary = summarize_subset_rebinned(sample_records)
        image_scope["reused_subset_from"] = str(reuse_existing_subset_dir)
    else:
        sample_records, subset_image_dir, subset_mask_dir = build_standardized_subset_with_mask_mode(
            output_dir=output_dir,
            validation_images=image_records,
            mask_catalog=mask_catalog,
            bucket_counts=bucket_counts,
            input_res=input_res,
            seed=int(args.seed),
            mask_selection_mode=str(args.mask_selection_mode),
        )
        phase5a.write_sample_manifest(sample_records, sample_manifest_path)
        subset_summary = summarize_subset_rebinned(sample_records)
        image_scope["reused_subset_from"] = ""
    image_scope["is_local_subset"] = len(sample_records) < len(image_records)
    if image_scope["is_official_val"]:
        image_scope["selection_mode"] = (
            "official_validation_list_seeded_subset"
            if image_scope["is_local_subset"]
            else "official_validation_list_full"
        )
    else:
        image_scope["selection_mode"] = (
            "recursive_image_root_scan_subset"
            if image_scope["is_local_subset"]
            else "recursive_image_root_scan_full"
        )

    config = {
        "phase": f"Phase 7 rebinned 256 protocol evaluation: {dataset_name}",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "seed": int(args.seed),
        "put_root": str(put_root),
        "checkpoint": checkpoint,
        "validation_list": str(validation_list),
        "image_root": str(image_root),
        "mask_root": str(mask_root),
        "sample_manifest": str(sample_manifest_path.resolve()),
        "dataset_name": dataset_name,
        "num_images": len(sample_records),
        "bucket_counts": bucket_counts,
        "input_res": list(input_res),
        "warmup_images": int(args.warmup_images),
        "num_token_per_iter": int(args.num_token_per_iter),
        "num_token_for_sampling": int(args.num_token_for_sampling),
        "methods": method_specs,
        "image_scope": image_scope,
        "subset_summary": subset_summary,
        "mask_selection_mode": str(args.mask_selection_mode),
        "mask_catalog_counts": mask_catalog_counts,
        "reused_subset_from": str(reuse_existing_subset_dir) if reuse_existing_subset_dir is not None else "",
        "protocol_bins": {
            "bucket_order": list(BUCKET_ORDER),
            "report_groups": list(REPORT_GROUPS),
            "overall_group": OVERALL_GROUP,
        },
        "freeze_only": bool(args.freeze_only),
        "resume_existing": bool(args.resume_existing),
        "fid_policy": (
            "not_computed_freeze_only"
            if args.freeze_only
            else "not_computed_lt_1k_by_flag"
            if args.skip_fid and len(sample_records) < 1000
            else "skipped_by_flag"
            if args.skip_fid
            else "computed_by_mask_group_and_0p01_60"
        ),
    }
    (output_dir / "CONFIG.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    write_dataset_scope_md(
        path=output_dir / "DATASET_SCOPE.md",
        dataset_name=dataset_name,
        image_root=image_root,
        validation_list=validation_list,
        image_scope=image_scope,
        sample_records=sample_records,
        subset_summary=subset_summary,
        mask_selection_mode=str(args.mask_selection_mode),
        mask_catalog_counts=mask_catalog_counts,
    )

    main_command = [
        f"PYTHONPATH={REPO_ROOT / 'tools'}:{REPO_ROOT / 'third_party' / 'PUT'}",
        sys.executable,
        "tools/phase7_main256_rebinned_protocol_eval.py",
        f"--put-root {put_root}",
        f"--checkpoint {checkpoint}",
        f"--validation-list {validation_list}" if validation_list is not None else "",
        f"--image-root {image_root}",
        f"--mask-root {mask_root}",
        f"--output-dir {output_dir}",
        f"--dataset-name '{dataset_name}'",
        f"--validation-source-label '{args.validation_source_label}'" if args.validation_source_label else "",
        f"--reuse-existing-subset-dir {reuse_existing_subset_dir}" if reuse_existing_subset_dir is not None else "",
        f"--image-selection-mode {args.image_selection_mode}",
        f"--mask-selection-mode {args.mask_selection_mode}",
        f"--count-01-10 {bucket_counts['01-10']}",
        f"--count-10-20 {bucket_counts['10-20']}",
        f"--count-20-30 {bucket_counts['20-30']}",
        f"--count-30-40 {bucket_counts['30-40']}",
        f"--count-40-50 {bucket_counts['40-50']}",
        f"--count-50-60 {bucket_counts['50-60']}",
        f"--warmup-images {args.warmup_images}",
        f"--num-token-per-iter {args.num_token_per_iter}",
        f"--num-token-for-sampling {args.num_token_for_sampling}",
        f"--methods {args.methods}",
        f"--seed {args.seed}",
        f"--gpu {args.gpu}",
        "--freeze-only" if args.freeze_only else "",
        "--skip-fid" if args.skip_fid else "",
        "--resume-existing" if args.resume_existing else "",
    ]
    main_command_text = " ".join(part for part in main_command if part)
    (output_dir / "COMMANDS.md").write_text(
        "# Commands\n\n"
        f"- `{main_command_text}`\n",
        encoding="utf-8",
    )
    if args.freeze_only:
        readme_lines = [
            f"# Phase 7 ReBinned Frozen Population: {dataset_name}",
            "",
            "Frozen evaluation population for the rebinned mask protocol.",
            "",
            f"- Actual image count: `{len(sample_records)}`",
            f"- Validation source: `{validation_list}`" if validation_list is not None else f"- Validation source: `{image_root}`",
            f"- Output directory: `{output_dir}`",
            f"- Bucket counts: `01-10={bucket_counts['01-10']}, 10-20={bucket_counts['10-20']}, 20-30={bucket_counts['20-30']}, 30-40={bucket_counts['30-40']}, 40-50={bucket_counts['40-50']}, 50-60={bucket_counts['50-60']}`",
            f"- Report groups: `0.01-20={subset_summary['report_group_counts']['0.01-20']}, 20-40={subset_summary['report_group_counts']['20-40']}, 40-60={subset_summary['report_group_counts']['40-60']}`",
            f"- Excluded >60 masks after resize: `{image_scope['excluded_mask_count']}`",
        ]
        (output_dir / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
        (output_dir / "metrics.md").write_text(
            "# Metrics\n\n- Status: `freeze_only`\n- No model inference was run in this directory.\n",
            encoding="utf-8",
        )
        print(json.dumps({"status": "ok", "freeze_only": True, "output_dir": str(output_dir), "num_images": len(sample_records)}, indent=2))
        return

    phase5a.set_seed(args.seed)
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    sys.path.insert(0, str(put_root))
    os.chdir(put_root)
    from scripts.inference import ImagePathDataset
    import lpips

    base_dataset = ImagePathDataset(str(subset_image_dir), str(subset_mask_dir), size=input_res)
    dataset = phase5e.FilteredDataset(base_dataset, [record["sample_id"] for record in sample_records])
    sample_map = phase5a.build_sample_group_map(sample_records)
    fid_file_lists = build_fid_file_lists(output_dir=output_dir, sample_records=sample_records)
    shared_masked_dir = output_dir / "shared_masked_gt"
    shared_masked_dir.mkdir(parents=True, exist_ok=True)

    lpips_model = lpips.LPIPS(net="vgg", spatial=True).to(device)
    high_texture_ids: set[str] = set()
    per_image_metrics: list[dict[str, Any]] = []
    per_image_compression: list[dict[str, Any]] = []
    per_image_timing: list[dict[str, Any]] = []
    outputs_manifest: list[dict[str, Any]] = []
    commands: list[str] = []
    fid_by_method_group: dict[str, dict[str, dict[str, Any]]] = {}

    for spec in method_specs:
        env = phase7_places2.env_for_method(spec, args)
        phase5f.apply_put_env(env)
        phase5a.set_seed(args.seed)
        model = phase5a.load_model(put_root, checkpoint, device)
        params_m = float(sum(parameter.numel() for parameter in model.parameters()) / 1e6)
        commands.append(" ".join([f"{key}={value}" for key, value in sorted(env.items()) if key.startswith("PUT_")]))

        phase7_places2.run_warmup(model=model, dataset=dataset, device=device, args=args)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(args.gpu)

        variant_dir = output_dir / spec["name"]
        completed_dir = variant_dir / "completed_single"
        completed_dir.mkdir(parents=True, exist_ok=True)
        artifact_records, pass_meta = phase7_places2.run_full_pass(
            model=model,
            dataset=dataset,
            device=device,
            args=args,
            completed_dir=completed_dir,
            masked_dir=shared_masked_dir,
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
        fid_by_method_group[spec["label"]] = compute_fid_for_method_groups(
            gt_dir=subset_image_dir,
            pred_dir=completed_dir,
            file_list_by_group=fid_file_lists,
            device=device,
            args=args,
        )

        quality_by_id = {record["relative_path"]: record for record in quality["records"]}
        artifact_by_id = {record["relative_path"]: record for record in artifact_records}

        for sample in sample_records:
            sample_id = sample["sample_id"]
            q = quality_by_id[sample_id]
            artifact = artifact_by_id[sample_id]
            route = phase7_places2.route_profile(artifact, spec)
            timings_ms = ((artifact.get("phase5f_timing_profile") or {}).get("timings_ms")) or {}
            cuda_total_ms = float(timings_ms.get("total_inference_time", 0.0))
            gmacs = phase7_places2.core_gmacs(artifact, spec, dim=dim, hidden_dim=hidden_dim, num_layers=num_layers)
            base = {
                "image_id": sample_id,
                "dataset": dataset_name,
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
                    "latency_sec": float(artifact["wall_time_sec"]),
                    "cuda_total_ms": cuda_total_ms,
                    "peak_memory_mb": float(pass_meta["peak_memory_allocated_mb"]),
                    "actual_removed_tokens": route["actual_removed_tokens"],
                    "actual_compression_ratio": route["actual_compression_ratio"],
                    "under_compression": int(route["under_compression"]),
                    "bad_restore": int(route["bad_restore"]),
                    "protect_overlap": int(route["protect_overlap"]),
                }
            )
            per_image_compression.append({**base, **route})
            per_image_timing.append(
                {
                    **base,
                    "latency_sec": float(artifact["wall_time_sec"]),
                    "cuda_total_ms": cuda_total_ms,
                    "transformer_blocks_ms": float(timings_ms.get("transformer_blocks_time", 0.0)),
                    "matching_ms": float(timings_ms.get("topk_matching_time", 0.0)),
                    "merge_ms": float(timings_ms.get("merge_time", 0.0)),
                    "restore_ms": float(timings_ms.get("restore_time", 0.0)),
                    "token_risk_compute_ms": float(timings_ms.get("token_risk_compute_time", 0.0)),
                    "peak_memory_mb": float(pass_meta["peak_memory_allocated_mb"]),
                }
            )
            outputs_manifest.append(
                {
                    **base,
                    "input_path": sample["resized_image_path"],
                    "mask_path": sample["resized_mask_path"],
                    "gt_path": sample["resized_image_path"],
                    "completed_output_path": str((completed_dir / sample_id).resolve()),
                    "masked_gt_path": str((shared_masked_dir / sample_id).resolve()),
                }
            )

    del lpips_model
    torch.cuda.empty_cache()

    phase7_places2.write_csv(output_dir / "per_image_metrics.csv", per_image_metrics)
    phase7_places2.write_csv(output_dir / "per_image_compression.csv", per_image_compression)
    phase7_places2.write_csv(output_dir / "per_image_timing.csv", per_image_timing)
    phase7_places2.write_csv(output_dir / "per_image_outputs_manifest.csv", outputs_manifest)
    paired_delta_rows, paired_delta_summary_rows = build_pair_rows(
        per_image_metrics=per_image_metrics,
        per_image_compression=per_image_compression,
        method_specs=method_specs,
    )
    phase7_places2.write_csv(output_dir / "paired_delta_per_image.csv", paired_delta_rows)
    phase7_places2.write_csv(output_dir / "paired_delta_summary.csv", paired_delta_summary_rows)

    summary_by_dataset = phase7_places2.summarize_rows(per_image_metrics, "dataset")
    expanded_for_groups = []
    for row in per_image_metrics:
        sample = sample_map[row["image_id"]]
        for group in group_for_sample(sample):
            expanded = dict(row)
            expanded["mask_group"] = group
            expanded_for_groups.append(expanded)
    summary_by_mask = phase7_places2.summarize_rows(expanded_for_groups, "mask_group")
    summary_main = [row for row in summary_by_mask if row["mask_group"] in REPORT_GROUPS]
    attach_group_fid(summary_by_dataset, fid_by_method_group, group_key="dataset")
    attach_group_fid(summary_by_mask, fid_by_method_group, group_key="mask_group")
    attach_group_fid(summary_main, fid_by_method_group, group_key="mask_group")
    phase7_places2.write_csv(output_dir / "summary_by_dataset.csv", summary_by_dataset)
    phase7_places2.write_csv(output_dir / "summary_by_mask_ratio.csv", summary_by_mask)
    phase7_places2.write_csv(output_dir / "summary_main_table.csv", summary_main)
    (output_dir / "summary_main_table_latex.txt").write_text(latex_from_rows_rebinned(summary_main), encoding="utf-8")

    fid_rows = []
    pilot_fid = len(sample_records) < 1000
    for method, result_by_group in fid_by_method_group.items():
        for group in SUMMARY_GROUPS:
            result = result_by_group.get(group, {})
            fid_rows.append(
                {
                    "method": method,
                    "mask_group": group,
                    "status": result.get("status", "missing"),
                    "pilot_fid": pilot_fid,
                    "fid": result.get("fid"),
                    "pids": result.get("pids"),
                    "uids": result.get("uids"),
                    "length1": result.get("length1"),
                    "length2": result.get("length2"),
                }
            )
    phase7_places2.write_csv(output_dir / "fid_summary.csv", fid_rows)
    (output_dir / "COMMANDS.md").write_text(
        "# Commands\n\n"
        f"- `{main_command_text}`\n\n"
        "## Method Environments\n\n"
        + "\n".join(f"- `{command}`" for command in commands)
        + "\n",
        encoding="utf-8",
    )
    readme_lines = [
        f"# Phase 7 ReBinned 256 Protocol Evaluation: {dataset_name}",
        "",
        f"{dataset_name} 256x256 rebinned protocol evaluation for the selected PUT-family methods.",
        "",
        f"- Actual image count: `{len(sample_records)}`",
        f"- Validation source: `{validation_list}`" if validation_list is not None else f"- Validation source: `{image_root}`",
        f"- Output directory: `{output_dir}`",
        f"- Reused subset from: `{reuse_existing_subset_dir}`" if reuse_existing_subset_dir is not None else "- Reused subset from: `none`",
        f"- Methods: `{', '.join(spec['label'] for spec in method_specs)}`",
        f"- Mask assignment mode: `{args.mask_selection_mode}`",
        f"- FID policy: `{config['fid_policy']}`",
    ]
    (output_dir / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
    metric_lines = [
        "# Metrics",
        "",
        f"- Dataset: `{dataset_name}`",
        f"- Images: `{len(sample_records)}`",
        f"- Split: `{image_scope['selection_mode']}`",
        f"- FID: `{'computed' if not args.skip_fid else 'skipped'}`",
        f"- pilot_fid: `{'true' if pilot_fid else 'false'}`",
        "",
        "| Mask Group | Method | PSNR | SSIM | LPIPS | FID | GMACs | Latency(s) | Actual CR | bad_restore | protect_overlap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_main:
        metric_lines.append(
            "| {group} | {method} | {psnr:.4f} | {ssim:.4f} | {lpips:.6f} | {fid} | {gmacs:.3f} | {lat:.4f} | {cr:.6f} | {bad:.3f} | {protect:.3f} |".format(
                group=row["mask_group"],
                method=row["method"],
                psnr=float(row["psnr_mean"]),
                ssim=float(row["ssim_mean"]),
                lpips=float(row["lpips_mean"]),
                fid=("{:.4f}".format(float(row["fid_mean"])) if row.get("fid_mean", "") != "" else row.get("fid_status", "")),
                gmacs=float(row["core_gmacs_mean"]),
                lat=float(row["latency_sec_mean"]),
                cr=float(row["actual_compression_ratio_mean"]),
                bad=float(row["bad_restore_mean"]),
                protect=float(row["protect_overlap_mean"]),
            )
        )
    metric_lines.extend(
        [
            "",
            "## Required Readout",
            "",
        ]
    )
    for row in paired_delta_summary_rows:
        if row["mask_group"] != OVERALL_GROUP:
            continue
        metric_lines.append(
            "- `{pair}`: `PSNR {psnr:+.4f}dB`, `LPIPS {lpips:+.6f}`, `latency {lat:+.4f}s`, `CUDA total {cuda:+.3f}ms`, `Core GMACs {gmacs:+.3f}`, `actual_CR_match_rate {cr_match:.3f}`, `actual_removed_match_rate {removed_match:.3f}`, `bad_restore {bad_a}/{bad_b}`, `protect_overlap {protect_a}/{protect_b}`".format(
                pair=row["pair_name"],
                psnr=float(row["psnr_delta_mean"]),
                lpips=float(row["lpips_delta_mean"]),
                lat=float(row["latency_sec_delta_mean"]),
                cuda=float(row["cuda_total_ms_delta_mean"]),
                gmacs=float(row["core_gmacs_delta_mean"]),
                cr_match=float(row["actual_compression_ratio_match_rate"]),
                removed_match=float(row["actual_removed_tokens_match_rate"]),
                bad_a=int(row["bad_restore_method_a_sum"]),
                bad_b=int(row["bad_restore_method_b_sum"]),
                protect_a=int(row["protect_overlap_method_a_sum"]),
                protect_b=int(row["protect_overlap_method_b_sum"]),
            )
        )
    (output_dir / "metrics.md").write_text("\n".join(metric_lines) + "\n", encoding="utf-8")

    summary = {
        "config": config,
        "fid_by_method_group": fid_by_method_group,
        "artifacts": {
            "sample_manifest": str(sample_manifest_path.resolve()),
            "per_image_metrics": str((output_dir / "per_image_metrics.csv").resolve()),
            "per_image_compression": str((output_dir / "per_image_compression.csv").resolve()),
            "per_image_timing": str((output_dir / "per_image_timing.csv").resolve()),
            "per_image_outputs_manifest": str((output_dir / "per_image_outputs_manifest.csv").resolve()),
            "paired_delta_per_image": str((output_dir / "paired_delta_per_image.csv").resolve()),
            "paired_delta_summary": str((output_dir / "paired_delta_summary.csv").resolve()),
            "summary_main_table": str((output_dir / "summary_main_table.csv").resolve()),
            "fid_summary": str((output_dir / "fid_summary.csv").resolve()),
            "dataset_scope_md": str((output_dir / "DATASET_SCOPE.md").resolve()),
            "metrics_md": str((output_dir / "metrics.md").resolve()),
        },
    }
    (output_dir / "phase7_main256_rebinned_protocol_summary.json").write_text(
        json.dumps(phase5a.to_serializable(summary), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "ok", "output_dir": str(output_dir), "num_images": len(sample_records)}, indent=2))


if __name__ == "__main__":
    main()
