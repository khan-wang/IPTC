#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import lpips
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "tools"
PUT_ROOT = REPO_ROOT / "third_party" / "PUT"
for import_path in (TOOLS_DIR, PUT_ROOT):
    path_str = str(import_path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from common_places2_fv import BUCKET_RANK, load_manifest, materialize_manifest_links
import phase5a_standardized_comparison as phase5a
from scripts.inference import ImagePathDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--completed-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--fid-batch-size", type=int, default=64)
    parser.add_argument("--fid-num-workers", type=int, default=4)
    parser.add_argument("--skip-fid", action="store_true", default=False)
    return parser.parse_args()


def build_sample_map(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {
        row["sample_id"]: {
            "sample_id": row["sample_id"],
            "bucket": row["bucket"],
            "report_group": row["report_group"],
            "mask_ratio": float(row["mask_ratio"]),
        }
        for row in rows
    }


def report_group_sort_key(group: str) -> tuple[int, str]:
    order = {
        "0.01-20": 0,
        "10-20": 1,
        "20-40": 2,
        "40-60": 3,
        "10-60": 4,
        "0.01-60": 5,
    }
    return (order.get(group, 999), group)


def detect_report_groups(rows: list[dict[str, str]]) -> tuple[list[str], str]:
    groups = sorted({row["report_group"] for row in rows}, key=report_group_sort_key)
    has_low = any(group.startswith("0.01-") for group in groups)
    overall_group = "0.01-60" if has_low else "10-60"
    return groups, overall_group


def build_fid_file_lists(
    output_dir: Path,
    rows: list[dict[str, str]],
    report_groups: list[str],
    overall_group: str,
) -> dict[str, Path]:
    by_group = {group: [] for group in list(report_groups) + [overall_group]}
    for row in rows:
        sample_id = row["sample_id"]
        by_group[row["report_group"]].append(sample_id)
        by_group[overall_group].append(sample_id)
    fid_dir = output_dir / "fid_filelists"
    fid_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for group, sample_ids in by_group.items():
        path = fid_dir / f"{group}.txt"
        path.write_text("\n".join(sample_ids) + "\n", encoding="utf-8")
        out[group] = path
    return out


def compute_fid_for_groups(
    *,
    gt_dir: Path,
    pred_dir: Path,
    file_list_by_group: dict[str, Path],
    report_groups: list[str],
    overall_group: str,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, dict]:
    if args.skip_fid:
        return {
            group: {"status": "skipped", "fid": None, "pids": None, "uids": None, "length1": 0, "length2": 0}
            for group in list(report_groups) + [overall_group]
        }
    from scripts.metrics.cal_fid import calculate_fid_given_paths

    out = {}
    for group in list(report_groups) + [overall_group]:
        result = calculate_fid_given_paths(
            path1=str(gt_dir),
            path_prefix1="",
            file_list1=str(file_list_by_group[group]),
            path2=str(pred_dir),
            path_prefix2="",
            file_list2=str(file_list_by_group[group]),
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


def stats_or_blank(summary: dict, key: str) -> tuple[float | str, float | str]:
    stats = summary.get(key) or {}
    mean = stats.get("mean")
    std = stats.get("std")
    return (float(mean) if mean is not None else "", float(std) if std is not None else "")


def summarize_records_by_group(
    rows: list[dict[str, str]],
    quality_records: list[dict[str, float]],
    report_groups: list[str],
    overall_group: str,
) -> dict[str, dict]:
    metrics_by_id = {record["relative_path"]: record for record in quality_records}
    grouped = {group: [] for group in list(report_groups) + [overall_group]}
    for row in rows:
        sample_id = row["sample_id"]
        grouped[row["report_group"]].append(metrics_by_id[sample_id])
        grouped[overall_group].append(metrics_by_id[sample_id])
    out = {}
    for group, records in grouped.items():
        out[group] = {
            "num_images": len(records),
            "psnr": phase5a.stats([float(record["psnr"]) for record in records]),
            "ssim": phase5a.stats([float(record["ssim"]) for record in records]),
            "lpips_output_vs_gt": phase5a.stats([float(record["lpips_output_vs_gt"]) for record in records]),
        }
    return out


def write_per_image_metrics(
    rows: list[dict[str, str]],
    quality_records: list[dict[str, float]],
    completed_dir: Path,
    out_path: Path,
) -> None:
    quality_by_id = {record["relative_path"]: record for record in quality_records}
    fieldnames = [
        "sample_id",
        "bucket",
        "report_group",
        "mask_ratio",
        "psnr",
        "ssim",
        "lpips",
        "input_path",
        "mask_path",
        "gt_path",
        "completed_output_path",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            metrics = quality_by_id[row["sample_id"]]
            writer.writerow(
                {
                    "sample_id": row["sample_id"],
                    "bucket": row["bucket"],
                    "report_group": row["report_group"],
                    "mask_ratio": row["mask_ratio"],
                    "psnr": metrics["psnr"],
                    "ssim": metrics["ssim"],
                    "lpips": metrics["lpips_output_vs_gt"],
                    "input_path": row["resized_image_path"],
                    "mask_path": row["resized_mask_path"],
                    "gt_path": row["resized_image_path"],
                    "completed_output_path": str((completed_dir / row["sample_id"]).resolve()),
                }
            )


def write_metrics_by_bin(
    group_summaries: dict[str, dict],
    fid_summary: dict[str, dict],
    report_groups: list[str],
    overall_group: str,
    out_path: Path,
) -> None:
    fieldnames = [
        "mask_ratio_group",
        "num_images",
        "psnr_mean",
        "psnr_std",
        "ssim_mean",
        "ssim_std",
        "lpips_mean",
        "lpips_std",
        "fid_status",
        "fid_mean",
        "pids_mean",
        "uids_mean",
        "fid_num_images",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for group in list(report_groups) + [overall_group]:
            summary = group_summaries[group]
            psnr_mean, psnr_std = stats_or_blank(summary, "psnr")
            ssim_mean, ssim_std = stats_or_blank(summary, "ssim")
            lpips_mean, lpips_std = stats_or_blank(summary, "lpips_output_vs_gt")
            fid = fid_summary[group]
            writer.writerow(
                {
                    "mask_ratio_group": group,
                    "num_images": summary["num_images"],
                    "psnr_mean": psnr_mean,
                    "psnr_std": psnr_std,
                    "ssim_mean": ssim_mean,
                    "ssim_std": ssim_std,
                    "lpips_mean": lpips_mean,
                    "lpips_std": lpips_std,
                    "fid_status": fid.get("status", "not_reported"),
                    "fid_mean": fid.get("fid", ""),
                    "pids_mean": fid.get("pids", ""),
                    "uids_mean": fid.get("uids", ""),
                    "fid_num_images": fid.get("length2", 0),
                }
            )


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    completed_dir = Path(args.completed_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_manifest(manifest_path)
    sample_map = build_sample_map(rows)
    report_groups, overall_group = detect_report_groups(rows)

    eval_subset_root = output_dir / "eval_subset"
    image_dir = eval_subset_root / "images"
    mask_dir = eval_subset_root / "masks"
    materialize_manifest_links(rows, image_dir=image_dir, mask_dir=mask_dir)

    dataset = ImagePathDataset(str(image_dir), mask_dir=str(mask_dir), size=(256, 256))
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    lpips_model = lpips.LPIPS(net="vgg").to(device)
    quality = phase5a.compute_single_output_metrics(
        dataset=dataset,
        completed_dir=completed_dir,
        sample_map=sample_map,
        lpips_model=lpips_model,
        device=device,
    )
    group_summaries = summarize_records_by_group(rows, quality["records"], report_groups, overall_group)
    fid_lists = build_fid_file_lists(output_dir, rows, report_groups, overall_group)
    fid_summary = compute_fid_for_groups(
        gt_dir=image_dir,
        pred_dir=completed_dir,
        file_list_by_group=fid_lists,
        report_groups=report_groups,
        overall_group=overall_group,
        device=device,
        args=args,
    )

    write_per_image_metrics(rows, quality["records"], completed_dir, output_dir / "per_image_metrics.csv")
    write_metrics_by_bin(
        group_summaries,
        fid_summary,
        report_groups=report_groups,
        overall_group=overall_group,
        out_path=output_dir / "metrics_by_bin.csv",
    )

    summary = {
        "status": "ok",
        "manifest": str(manifest_path.resolve()),
        "completed_dir": str(completed_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "num_images": len(rows),
        "report_groups": report_groups,
        "overall_group": overall_group,
        "bucket_order_present": sorted({row["bucket"] for row in rows}, key=lambda bucket: BUCKET_RANK.get(bucket, 999)),
        "fid_status_by_group": {
            group: fid_summary[group]["status"] for group in list(report_groups) + [overall_group]
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
