#!/usr/bin/env python3
"""Phase 7B official-style Places2/NaturalScene 256 validation evaluation."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

import phase5a_standardized_comparison as phase5a
import phase5e_ga_spg_probe as phase5e
import phase5f_ga_spg_overhead_attribution as phase5f


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_GROUPS = ("20-40", "40-60", "10-60")
BUCKET_ORDER = ("10-20", "20-30", "30-40", "40-50", "50-60")
IMAGE_EXTS = {".bmp", ".jpg", ".jpeg", ".pgm", ".png", ".ppm", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--put-root", default=str(REPO_ROOT / "third_party" / "PUT"))
    checkpoint_default = os.environ.get("PUT_CHECKPOINT", "")
    image_root_default = os.environ.get("SBVC_IMAGE_ROOT", "")
    mask_root_default = os.environ.get("SBVC_MASK_ROOT", "")
    parser.add_argument(
        "--checkpoint",
        default=checkpoint_default,
        required=not checkpoint_default,
    )
    parser.add_argument("--validation-list", default=str(REPO_ROOT / "third_party" / "PUT" / "data" / "naturalscenevalidation.txt"))
    parser.add_argument("--image-root", default=image_root_default, required=not image_root_default)
    parser.add_argument("--mask-root", default=mask_root_default, required=not mask_root_default)
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "runs" / "put_full_protocol"))
    parser.add_argument("--input-res", default="256,256")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260428)
    parser.add_argument("--samples-per-bucket", type=int, default=0)
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
    parser.add_argument("--skip-fid", action="store_true", default=False)
    parser.add_argument("--resume-existing", action="store_true", default=False)
    parser.add_argument("--fid-batch-size", type=int, default=64)
    parser.add_argument("--fid-num-workers", type=int, default=4)
    return parser.parse_args()


def resolve_places2_images(image_root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(image_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        relpath = path.relative_to(image_root).as_posix()
        records.append(
            {
                "validation_relpath": relpath,
                "image_uid": relpath,
                "source_image_path": str(path.resolve()),
            }
        )
    if not records:
        raise RuntimeError(f"no images found under {image_root}")
    return records


def resolve_image_records(image_root: Path, validation_list: Path | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if validation_list is not None and validation_list.is_file():
        records = phase5a.resolve_validation_images(validation_list=validation_list, image_root=image_root)
        return records, {
            "source": "third_party/PUT/data/naturalscenevalidation.txt",
            "is_official_val": True,
            "is_local_subset": True,
            "selection_mode": "official_validation_list",
        }
    records = resolve_places2_images(image_root)
    return records, {
        "source": str(image_root),
        "is_official_val": False,
        "is_local_subset": True,
        "selection_mode": "recursive_image_root_scan",
    }


def build_places2_subset(
    output_dir: Path,
    image_records: list[dict[str, Any]],
    mask_catalog: list[dict[str, Any]],
    samples_per_bucket: int,
    input_res: tuple[int, int],
    seed: int,
) -> tuple[list[dict[str, Any]], Path, Path]:
    subset_dir = output_dir / "subset"
    image_dir = subset_dir / "images"
    mask_dir = subset_dir / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    total_required = samples_per_bucket * len(BUCKET_ORDER)
    if total_required > len(image_records):
        raise RuntimeError(f"need {total_required} images but only found {len(image_records)}")

    rng = np.random.default_rng(seed)
    shuffled_indices = rng.permutation(len(image_records)).tolist()
    selected_images = [image_records[idx] for idx in shuffled_indices[:total_required]]

    records = []
    cursor = 0
    sample_index = 0
    for bucket in BUCKET_ORDER:
        bucket_masks = [item for item in mask_catalog if item["bucket"] == bucket]
        selected_masks = phase5a.select_spaced(bucket_masks, samples_per_bucket, key="mask_ratio")
        bucket_images = selected_images[cursor : cursor + samples_per_bucket]
        cursor += samples_per_bucket
        report_group = next(spec["report_group"] for spec in phase5a.MASK_BUCKET_SPECS if spec["bucket"] == bucket)
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
                    "report_group": report_group,
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
        {
            "name": "global_tome_r224",
            "label": f"PUT + Global ToMe-r{args.global_tome_r}",
            "kind": "global_tome",
            "r": args.global_tome_r,
        },
        {
            "name": "pure_distance_r224",
            "label": f"PUT + PureDistance-r{args.safe_tome_r}",
            "kind": "pure_distance",
            "r": args.safe_tome_r,
        },
        {
            "name": "sbvc_r224_zero_pad_fastpath",
            "label": f"PUT + SBVC-r{args.safe_tome_r}-zero-pad-fastpath",
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
    return [sample["report_group"], "10-60"]


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
        out.append(row)
    return out


def latex_from_rows(rows: list[dict[str, Any]]) -> str:
    lines = [
        r"\\begin{tabular}{lrrrrrrr}",
        r"Method & PSNR & SSIM & LPIPS & FID & GMACs & Latency(s) & Actual CR \\",
        r"\\hline",
    ]
    for row in rows:
        if row.get("mask_group") != "10-60":
            continue
        fid_value = row.get("fid_mean", "")
        fid_text = "{:.4f}".format(float(fid_value)) if fid_value != "" else "NA"
        lines.append(
            "{method} & {psnr:.4f} & {ssim:.4f} & {lpips:.6f} & {fid} & {gmacs:.3f} & {lat:.4f} & {cr:.6f} \\\\".format(
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


def build_batch(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "relative_path": [sample["relative_path"]],
        "image": sample["image"].unsqueeze(0),
        "mask": sample["mask"].unsqueeze(0),
    }


def run_warmup(
    *,
    model,
    dataset,
    device: torch.device,
    args: argparse.Namespace,
) -> int:
    limit = min(int(args.warmup_images), len(dataset))
    with torch.no_grad():
        for idx in range(limit):
            sample = dataset[idx]
            _ = model.generate_content(
                batch=build_batch(sample),
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
    return limit


def run_full_pass(
    *,
    model,
    dataset,
    device: torch.device,
    args: argparse.Namespace,
    completed_dir: Path,
    masked_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = []
    dataset_start = time.perf_counter()
    artifact_dir = completed_dir.parent / "artifact_records"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for idx in range(len(dataset)):
            sample = dataset[idx]
            relative_path = sample["relative_path"]
            completed_path = completed_dir / relative_path
            masked_path = masked_dir / relative_path
            artifact_path = artifact_dir / f"{Path(relative_path).stem}.json"
            if (
                args.resume_existing
                and completed_path.is_file()
                and masked_path.is_file()
                and artifact_path.is_file()
            ):
                with artifact_path.open("r", encoding="utf-8") as handle:
                    records.append(json.load(handle))
                if idx == 0 or (idx + 1) % 100 == 0:
                    print(f"[run_full_pass][resume] {idx + 1}/{len(dataset)} {relative_path}", flush=True)
                continue
            start = time.perf_counter()
            output = model.generate_content(
                batch=build_batch(sample),
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
            phase5a.save_tensor_image(output["completed"][0], completed_path)
            phase5a.save_tensor_image(output["masked_gt"][0], masked_path)
            record = {
                "relative_path": relative_path,
                "wall_time_sec": float(elapsed),
                "phase4a_timing_profile": output.get("phase4a_timing_profile"),
                "phase5f_timing_profile": output.get("phase5f_timing_profile"),
                "safe_tome_profile": output.get("safe_tome_profile"),
                "global_tome_profile": output.get("global_tome_profile"),
                "boundary_split_profile": output.get("boundary_split_profile"),
                "similarity_profile": output.get("similarity_profile"),
                "ga_spg_profile": output.get("ga_spg_profile"),
                "shape_or_sampling_crash": False,
            }
            artifact_path.write_text(
                json.dumps(phase5a.to_serializable(record), ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            records.append(record)
            if idx == 0 or (idx + 1) % 100 == 0:
                print(
                    f"[run_full_pass] {idx + 1}/{len(dataset)} {relative_path} elapsed={elapsed:.4f}s",
                    flush=True,
                )
    dataset_elapsed = time.perf_counter() - dataset_start
    return records, {
        "dataset_wall_time_sec": float(dataset_elapsed),
        "dataset_wall_time_per_image_sec": float(dataset_elapsed / max(len(dataset), 1)),
        "peak_memory_allocated_mb": float(torch.cuda.max_memory_allocated(args.gpu) / 1024 / 1024),
        "peak_memory_reserved_mb": float(torch.cuda.max_memory_reserved(args.gpu) / 1024 / 1024),
    }


def compute_fid_for_method(
    *,
    gt_dir: Path,
    pred_dir: Path,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if args.skip_fid:
        return {"status": "skipped", "fid": None, "pids": None, "uids": None, "length1": 0, "length2": 0}
    from scripts.metrics.cal_fid import calculate_fid_given_paths

    result = calculate_fid_given_paths(
        path1=str(gt_dir),
        path_prefix1="",
        file_list1="",
        path2=str(pred_dir),
        path_prefix2="",
        file_list2="",
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
    return {"status": "computed", **result}


def attach_fid(summary_rows: list[dict[str, Any]], fid_by_method: dict[str, dict[str, Any]]) -> None:
    for row in summary_rows:
        fid = fid_by_method.get(row["method"], {})
        row["fid_status"] = fid.get("status", "missing")
        row["fid_mean"] = float(fid.get("fid", 0.0)) if fid.get("fid") is not None else ""
        row["fid_std"] = 0.0
        row["fid_num_images"] = int(fid.get("length2", 0)) if fid.get("length2") is not None else 0
        row["pids_mean"] = float(fid.get("pids", 0.0)) if fid.get("pids") is not None else ""
        row["uids_mean"] = float(fid.get("uids", 0.0)) if fid.get("uids") is not None else ""


def write_dataset_scope_md(
    *,
    path: Path,
    image_root: Path,
    validation_list: Path | None,
    image_scope: dict[str, Any],
    sample_records: list[dict[str, Any]],
    subset_summary: dict[str, Any],
) -> None:
    actual_images = len(sample_records)
    lines = [
        "# Dataset Scope",
        "",
        f"- Actual image count: `{actual_images}`",
        f"- Data source: `{image_scope['source']}`",
        f"- Image root: `{image_root}`",
        f"- Official val: `{'yes' if image_scope['is_official_val'] else 'no'}`",
        f"- Local available subset: `{'yes' if image_scope['is_local_subset'] else 'no'}`",
        f"- Selection mode: `{image_scope['selection_mode']}`",
        f"- Validation list: `{validation_list}`" if validation_list is not None else "- Validation list: `none`",
        "",
        "## Mask Ratio Distribution",
        "",
        f"- 10-20: `{subset_summary['bucket_counts']['10-20']}`",
        f"- 20-30: `{subset_summary['bucket_counts']['20-30']}`",
        f"- 30-40: `{subset_summary['bucket_counts']['30-40']}`",
        f"- 40-50: `{subset_summary['bucket_counts']['40-50']}`",
        f"- 50-60: `{subset_summary['bucket_counts']['50-60']}`",
        f"- 20-40 total: `{subset_summary['report_group_counts']['20-40']}`",
        f"- 40-60 total: `{subset_summary['report_group_counts']['40-60']}`",
        f"- 10-60 total: `{subset_summary['report_group_counts']['10-60']}`",
        "",
        "## Note",
        "",
        "This run follows the current local PUT Naturalscene validation protocol. "
        "If the image count is below the full Places365 validation set, treat it as "
        "an official-style local validation subset rather than a full Places365 sweep.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    put_root = Path(args.put_root).resolve()
    checkpoint = str(Path(args.checkpoint).resolve())
    image_root = Path(args.image_root).resolve()
    validation_list = Path(args.validation_list).resolve() if args.validation_list else None
    mask_root = Path(args.mask_root).resolve()
    input_res = phase5a.parse_hw(args.input_res)

    image_records, image_scope = resolve_image_records(image_root=image_root, validation_list=validation_list)
    samples_per_bucket = int(args.samples_per_bucket)
    if samples_per_bucket <= 0:
        if len(image_records) % len(BUCKET_ORDER) != 0:
            raise RuntimeError(
                f"image count {len(image_records)} is not divisible by {len(BUCKET_ORDER)}; "
                "set --samples-per-bucket explicitly"
            )
        samples_per_bucket = len(image_records) // len(BUCKET_ORDER)
    mask_catalog = phase5a.build_mask_catalog(mask_root, input_res)
    sample_records, subset_image_dir, subset_mask_dir = build_places2_subset(
        output_dir=output_dir,
        image_records=image_records,
        mask_catalog=mask_catalog,
        samples_per_bucket=samples_per_bucket,
        input_res=input_res,
        seed=int(args.seed),
    )
    sample_manifest_path = output_dir / "sample_manifest.csv"
    phase5a.write_sample_manifest(sample_records, sample_manifest_path)
    subset_summary = phase5a.summarize_subset(sample_records)

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
    shared_masked_dir = output_dir / "shared_masked_gt"
    shared_masked_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "phase": "Phase 7B official-style Places2/NaturalScene 256 validation",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "seed": int(args.seed),
        "put_root": str(put_root),
        "checkpoint": checkpoint,
        "validation_list": str(validation_list) if validation_list is not None else None,
        "image_root": str(image_root),
        "mask_root": str(mask_root),
        "sample_manifest": str(sample_manifest_path.resolve()),
        "num_images": len(sample_records),
        "samples_per_bucket": samples_per_bucket,
        "input_res": list(input_res),
        "warmup_images": int(args.warmup_images),
        "num_token_per_iter": int(args.num_token_per_iter),
        "num_token_for_sampling": int(args.num_token_for_sampling),
        "methods": method_specs(args),
        "image_scope": image_scope,
        "subset_summary": subset_summary,
        "fid_policy": (
            "not_computed_lt_1k_by_flag"
            if args.skip_fid and len(sample_records) < 1000
            else "skipped_by_flag"
            if args.skip_fid
            else "computed"
        ),
    }
    (output_dir / "CONFIG.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lpips_model = lpips.LPIPS(net="vgg", spatial=True).to(device)
    high_texture_ids: set[str] = set()
    per_image_metrics = []
    per_image_compression = []
    per_image_timing = []
    outputs_manifest = []
    commands = []
    fid_by_method: dict[str, dict[str, Any]] = {}

    for spec in method_specs(args):
        env = env_for_method(spec, args)
        phase5f.apply_put_env(env)
        phase5a.set_seed(args.seed)
        model = phase5a.load_model(put_root, checkpoint, device)
        params_m = float(sum(parameter.numel() for parameter in model.parameters()) / 1e6)
        commands.append(" ".join([f"{key}={value}" for key, value in sorted(env.items()) if key.startswith("PUT_")]))

        run_warmup(model=model, dataset=dataset, device=device, args=args)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(args.gpu)

        variant_dir = output_dir / spec["name"]
        completed_dir = variant_dir / "completed_single"
        completed_dir.mkdir(parents=True, exist_ok=True)
        artifact_records, pass_meta = run_full_pass(
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
        fid_by_method[spec["label"]] = compute_fid_for_method(
            gt_dir=subset_image_dir,
            pred_dir=completed_dir,
            device=device,
            args=args,
        )

        quality_by_id = {record["relative_path"]: record for record in quality["records"]}
        artifact_by_id = {record["relative_path"]: record for record in artifact_records}

        for sample in sample_records:
            sample_id = sample["sample_id"]
            q = quality_by_id[sample_id]
            artifact = artifact_by_id[sample_id]
            route = route_profile(artifact, spec)
            timings_ms = ((artifact.get("phase5f_timing_profile") or {}).get("timings_ms")) or {}
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
    attach_fid(summary_by_dataset, fid_by_method)
    attach_fid(summary_by_mask, fid_by_method)
    attach_fid(summary_main, fid_by_method)
    write_csv(output_dir / "summary_by_dataset.csv", summary_by_dataset)
    write_csv(output_dir / "summary_by_mask_ratio.csv", summary_by_mask)
    write_csv(output_dir / "summary_main_table.csv", summary_main)
    (output_dir / "summary_main_table_latex.txt").write_text(latex_from_rows(summary_main), encoding="utf-8")

    fid_rows = []
    pilot_fid = len(sample_records) < 1000
    for method, result in fid_by_method.items():
        fid_rows.append(
            {
                "method": method,
                "status": result.get("status", "missing"),
                "pilot_fid": pilot_fid,
                "fid": result.get("fid"),
                "pids": result.get("pids"),
                "uids": result.get("uids"),
                "length1": result.get("length1"),
                "length2": result.get("length2"),
            }
        )
    write_csv(output_dir / "fid_summary.csv", fid_rows)

    write_dataset_scope_md(
        path=output_dir / "DATASET_SCOPE.md",
        image_root=image_root,
        validation_list=validation_list,
        image_scope=image_scope,
        sample_records=sample_records,
        subset_summary=subset_summary,
    )

    main_command = [
        f"PYTHONPATH={REPO_ROOT / 'tools'}:{REPO_ROOT / 'third_party' / 'PUT'}",
        sys.executable,
        "tools/phase7_main256_places2_full_eval.py",
        f"--put-root {put_root}",
        f"--checkpoint {checkpoint}",
        f"--validation-list {validation_list}" if validation_list is not None else "",
        f"--image-root {image_root}",
        f"--mask-root {mask_root}",
        f"--output-dir {output_dir}",
        f"--samples-per-bucket {samples_per_bucket}",
        f"--warmup-images {args.warmup_images}",
        f"--num-token-per-iter {args.num_token_per_iter}",
        f"--num-token-for-sampling {args.num_token_for_sampling}",
        f"--seed {args.seed}",
        f"--gpu {args.gpu}",
        "--skip-fid" if args.skip_fid else "",
        "--resume-existing" if args.resume_existing else "",
    ]
    main_command_text = " ".join(part for part in main_command if part)
    (output_dir / "COMMANDS.md").write_text(
        "# Commands\n\n"
        f"- `{main_command_text}`\n\n"
        "## Method Environments\n\n"
        + "\n".join(f"- `{command}`" for command in commands)
        + "\n",
        encoding="utf-8",
    )
    readme_lines = [
        "# Phase 7B Full Places2/NaturalScene Validation",
        "",
        "Official-style Places2/NaturalScene 256x256 validation evaluation for "
        "Baseline / Global ToMe / PureDistance-r224 / SBVC-r224.",
        "",
        f"- Actual image count: `{len(sample_records)}`",
        f"- Validation source: `{validation_list}`" if validation_list is not None else "- Validation source: `none`",
        f"- Output directory: `{output_dir}`",
        f"- FID policy: `{config['fid_policy']}`",
    ]
    (output_dir / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
    metric_lines = [
        "# Metrics",
        "",
        "- Dataset: `Places2/NaturalScene`",
        f"- Images: `{len(sample_records)}`",
        f"- Split: `{'official-style validation list' if image_scope['is_official_val'] else 'local image root scan'}`",
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
    (output_dir / "metrics.md").write_text("\n".join(metric_lines) + "\n", encoding="utf-8")

    summary = {
        "config": config,
        "fid_by_method": fid_by_method,
        "artifacts": {
            "sample_manifest": str(sample_manifest_path.resolve()),
            "per_image_metrics": str((output_dir / "per_image_metrics.csv").resolve()),
            "per_image_compression": str((output_dir / "per_image_compression.csv").resolve()),
            "per_image_timing": str((output_dir / "per_image_timing.csv").resolve()),
            "per_image_outputs_manifest": str((output_dir / "per_image_outputs_manifest.csv").resolve()),
            "summary_main_table": str((output_dir / "summary_main_table.csv").resolve()),
            "fid_summary": str((output_dir / "fid_summary.csv").resolve()),
            "dataset_scope_md": str((output_dir / "DATASET_SCOPE.md").resolve()),
            "metrics_md": str((output_dir / "metrics.md").resolve()),
        },
    }
    (output_dir / "phase7_main256_full_summary.json").write_text(
        json.dumps(phase5a.to_serializable(summary), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "ok", "output_dir": str(output_dir), "num_images": len(sample_records)}, indent=2))


if __name__ == "__main__":
    main()
