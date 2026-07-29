#!/usr/bin/env python3
"""Prepare a fixed Places2 5K subset from extended36500 and run r sweep."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import phase5a_standardized_comparison as phase5a
import phase7_main256_places2_full_eval as phase7_places2
import phase7_main256_protocol_eval as phase7_eval


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_RUN_DIR = REPO_ROOT / "exp_data" / "01_main_256_places2_extended36500_retry1"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "exp_data" / "02_r_sweep"
DEFAULT_CHECKPOINT = os.environ.get("PUT_CHECKPOINT", "")
BUCKET_ORDER = ("10-20", "20-30", "30-40", "40-50", "50-60")
R_VALUES = (128, 160, 192, 224, 256)
METHODS = ("pure_distance_r224", "sbvc_r224_zero_pad_fastpath")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run-dir", default=str(DEFAULT_SOURCE_RUN_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--put-root", default=str(REPO_ROOT / "third_party" / "PUT"))
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, required=not DEFAULT_CHECKPOINT)
    image_root_default = os.environ.get("SBVC_IMAGE_ROOT", "")
    mask_root_default = os.environ.get("SBVC_MASK_ROOT", "")
    parser.add_argument("--image-root", default=image_root_default, required=not image_root_default)
    parser.add_argument("--mask-root", default=mask_root_default, required=not mask_root_default)
    parser.add_argument("--dataset-name", default="Places2/NaturalScene Subset-5K r-sweep")
    parser.add_argument("--input-res", default="256,256")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260429)
    parser.add_argument("--samples-per-bucket", type=int, default=1000)
    parser.add_argument("--warmup-images", type=int, default=16)
    parser.add_argument("--num-token-per-iter", type=int, default=20)
    parser.add_argument("--num-token-for-sampling", type=int, default=200)
    parser.add_argument("--boundary-ring-radius", type=int, default=1)
    parser.add_argument("--boundary-band-radius", type=int, default=5)
    parser.add_argument("--lite-lambda", type=float, default=0.20)
    parser.add_argument("--lite-w-texture", type=float, default=0.45)
    parser.add_argument("--lite-w-boundary", type=float, default=0.35)
    parser.add_argument("--lite-w-smoothness", type=float, default=0.20)
    parser.add_argument("--lite-w-image-border", type=float, default=0.20)
    parser.add_argument("--fid-num-workers", type=int, default=8)
    parser.add_argument("--rerun-existing", action="store_true", default=False)
    return parser.parse_args()


def ensure_symlink(link_path: Path, target_path: Path) -> None:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.exists() or link_path.is_symlink():
        if link_path.resolve() == target_path.resolve():
            return
        raise FileExistsError(f"existing path conflicts with required symlink: {link_path} -> {target_path}")
    link_path.symlink_to(target_path)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def select_subset_records(source_manifest: Path, seed: int, samples_per_bucket: int) -> list[dict[str, Any]]:
    rows = phase7_eval.read_sample_manifest(source_manifest)
    rows_by_bucket = {bucket: [] for bucket in BUCKET_ORDER}
    for row in rows:
        rows_by_bucket[str(row["bucket"])].append(dict(row))
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for bucket in BUCKET_ORDER:
        bucket_rows = sorted(rows_by_bucket[bucket], key=lambda item: item["sample_id"])
        if len(bucket_rows) < samples_per_bucket:
            raise RuntimeError(
                f"bucket {bucket} has only {len(bucket_rows)} samples in {source_manifest}, "
                f"need {samples_per_bucket}"
            )
        picked = rng.sample(bucket_rows, samples_per_bucket)
        selected.extend(sorted(picked, key=lambda item: item["sample_id"]))
    return selected


def materialize_subset(
    *,
    subset_dir: Path,
    sample_records: list[dict[str, Any]],
) -> Path:
    image_dir = subset_dir / "subset" / "images"
    mask_dir = subset_dir / "subset" / "masks"
    for record in sample_records:
        image_target = Path(str(record["resized_image_path"])).resolve()
        mask_target = Path(str(record["resized_mask_path"])).resolve()
        image_link = image_dir / str(record["sample_id"])
        mask_link = mask_dir / str(record["sample_id"])
        ensure_symlink(image_link, image_target)
        ensure_symlink(mask_link, mask_target)
        record["resized_image_path"] = str(image_link.resolve())
        record["resized_mask_path"] = str(mask_link.resolve())
    manifest_path = subset_dir / "sample_manifest.csv"
    phase5a.write_sample_manifest(sample_records, manifest_path)
    return manifest_path


def already_completed(run_dir: Path) -> bool:
    summary_json = run_dir / "phase7_main256_protocol_summary.json"
    if not summary_json.is_file():
        return False
    try:
        payload = json.loads(summary_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return payload.get("status") == "ok"


def run_single_eval(
    *,
    args: argparse.Namespace,
    subset_dir: Path,
    output_dir: Path,
    r_value: int,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str((REPO_ROOT / "tools" / "phase7_main256_protocol_eval.py").resolve()),
        "--put-root",
        str(Path(args.put_root).resolve()),
        "--checkpoint",
        str(Path(args.checkpoint).resolve()),
        "--image-root",
        str(Path(args.image_root).resolve()),
        "--mask-root",
        str(Path(args.mask_root).resolve()),
        "--output-dir",
        str(output_dir.resolve()),
        "--reuse-existing-subset-dir",
        str(subset_dir.resolve()),
        "--dataset-name",
        args.dataset_name,
        "--validation-source-label",
        "subset extracted from Places2 extended36500 sample_manifest",
        "--image-selection-mode",
        "recursive_image_root_scan",
        "--mask-selection-mode",
        "deterministic_cycle",
        "--samples-per-bucket",
        str(int(args.samples_per_bucket)),
        "--warmup-images",
        str(int(args.warmup_images)),
        "--num-token-per-iter",
        str(int(args.num_token_per_iter)),
        "--num-token-for-sampling",
        str(int(args.num_token_for_sampling)),
        "--methods",
        ",".join(METHODS),
        "--safe-tome-r",
        str(int(r_value)),
        "--boundary-ring-radius",
        str(int(args.boundary_ring_radius)),
        "--boundary-band-radius",
        str(int(args.boundary_band_radius)),
        "--lite-lambda",
        str(float(args.lite_lambda)),
        "--lite-w-texture",
        str(float(args.lite_w_texture)),
        "--lite-w-boundary",
        str(float(args.lite_w_boundary)),
        "--lite-w-smoothness",
        str(float(args.lite_w_smoothness)),
        "--lite-w-image-border",
        str(float(args.lite_w_image_border)),
        "--seed",
        str(int(args.seed)),
        "--gpu",
        str(int(args.gpu)),
        "--skip-fid",
        "--fid-num-workers",
        str(int(args.fid_num_workers)),
    ]
    env = dict(os.environ)
    py_path = [
        str((REPO_ROOT / "tools").resolve()),
        str((REPO_ROOT / "third_party" / "PUT").resolve()),
    ]
    env["PYTHONPATH"] = os.pathsep.join(py_path + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    log_path = output_dir / "run.log"
    with log_path.open("w", encoding="utf-8") as handle:
        subprocess.run(command, check=True, stdout=handle, stderr=subprocess.STDOUT, env=env)
    return command


def summarize_run_rows(
    *,
    run_dir: Path,
    nominal_r: int,
    subset_seed: int,
    subset_size: int,
) -> list[dict[str, Any]]:
    rows = read_csv_rows(run_dir / "summary_main_table.csv")
    out = []
    for row in rows:
        out.append(
            {
                "dataset": "Places2/NaturalScene Subset-5K",
                "subset_seed": subset_seed,
                "subset_size": subset_size,
                "nominal_r": nominal_r,
                "mask_group": row["mask_group"],
                "method": row["method"],
                "psnr_mean": row["psnr_mean"],
                "lpips_mean": row["lpips_mean"],
                "latency_sec_mean": row["latency_sec_mean"],
                "core_gmacs_mean": row["core_gmacs_mean"],
                "actual_compression_ratio_mean": row["actual_compression_ratio_mean"],
                "actual_removed_tokens_mean": row["actual_removed_tokens_mean"],
                "under_compression_mean": row["under_compression_mean"],
                "bad_restore_mean": row["bad_restore_mean"],
                "protect_overlap_mean": row["protect_overlap_mean"],
                "fid_status": row.get("fid_status", "skipped"),
                "fid_mean": row.get("fid_mean", ""),
                "source_run_dir": str(run_dir.resolve()),
            }
        )
    return out


def write_top_level_artifacts(
    *,
    output_dir: Path,
    args: argparse.Namespace,
    subset_dir: Path,
    subset_records: list[dict[str, Any]],
    commands: list[list[str]],
    summary_rows: list[dict[str, Any]],
) -> None:
    summary_path = output_dir / "places2_5k_r_sweep_summary.csv"
    phase7_places2.write_csv(summary_path, summary_rows)
    subset_summary = phase5a.summarize_subset(subset_records)
    config = {
        "phase": "Phase 7 Places2 Subset-5K r sweep",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_run_dir": str(Path(args.source_run_dir).resolve()),
        "subset_dir": str(subset_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "seed": int(args.seed),
        "samples_per_bucket": int(args.samples_per_bucket),
        "r_values": list(R_VALUES),
        "methods": list(METHODS),
        "skip_fid": True,
        "fid_status": "skipped",
        "subset_summary": subset_summary,
    }
    (output_dir / "CONFIG.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    command_lines = [
        "# Commands",
        "",
        f"- Driver: `{' '.join(sys.argv)}`",
        "",
        "## Per-r Evaluation Commands",
        "",
    ]
    command_lines.extend(f"- `{' '.join(command)}`" for command in commands)
    (output_dir / "COMMANDS.md").write_text("\n".join(command_lines) + "\n", encoding="utf-8")
    readme_lines = [
        "# Places2 5K r Sweep",
        "",
        "- Source population: `Places2 extended36500` sample manifest",
        f"- Fixed subset seed: `{args.seed}`",
        f"- Subset size: `{len(subset_records)}`",
        f"- Bucket allocation: `10-20/20-30/30-40/40-50/50-60 = {args.samples_per_bucket} each`",
        f"- Methods: `{', '.join(METHODS)}`",
        f"- r values: `{', '.join(str(value) for value in R_VALUES)}`",
        "- FID: `skipped`",
        f"- Summary CSV: `{summary_path.resolve()}`",
    ]
    (output_dir / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
    metric_lines = [
        "# Metrics",
        "",
        "- Primary plotting rows are `mask_group=10-60`.",
        "- FID is intentionally skipped on the 5K subset to avoid small-sample distortion claims.",
        "",
        "| r | Method | Mask Group | PSNR | LPIPS | Latency(s) | Core GMACs | Actual CR | FID status |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    key_rows = [
        row for row in summary_rows
        if row["mask_group"] == "10-60"
    ]
    key_rows.sort(key=lambda row: (int(row["nominal_r"]), row["method"]))
    for row in key_rows:
        metric_lines.append(
            "| {r} | {method} | {group} | {psnr:.4f} | {lpips:.6f} | {lat:.4f} | {gmacs:.3f} | {cr:.6f} | {fid_status} |".format(
                r=int(row["nominal_r"]),
                method=row["method"],
                group=row["mask_group"],
                psnr=float(row["psnr_mean"]),
                lpips=float(row["lpips_mean"]),
                lat=float(row["latency_sec_mean"]),
                gmacs=float(row["core_gmacs_mean"]),
                cr=float(row["actual_compression_ratio_mean"]),
                fid_status=row["fid_status"],
            )
        )
    (output_dir / "metrics.md").write_text("\n".join(metric_lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_run_dir = Path(args.source_run_dir).resolve()
    source_manifest = source_run_dir / "sample_manifest.csv"
    if not source_manifest.is_file():
        raise FileNotFoundError(f"source manifest not found: {source_manifest}")

    subset_dir = output_dir / "subset_5k"
    subset_records = select_subset_records(
        source_manifest=source_manifest,
        seed=int(args.seed),
        samples_per_bucket=int(args.samples_per_bucket),
    )
    materialize_subset(subset_dir=subset_dir, sample_records=subset_records)

    all_commands: list[list[str]] = []
    summary_rows: list[dict[str, Any]] = []
    for r_value in R_VALUES:
        run_dir = output_dir / f"r{r_value}"
        if args.rerun_existing or not already_completed(run_dir):
            command = run_single_eval(
                args=args,
                subset_dir=subset_dir,
                output_dir=run_dir,
                r_value=r_value,
            )
        else:
            command = ["SKIPPED_EXISTING_OK", str(run_dir.resolve())]
        all_commands.append(command)
        summary_rows.extend(
            summarize_run_rows(
                run_dir=run_dir,
                nominal_r=r_value,
                subset_seed=int(args.seed),
                subset_size=len(subset_records),
            )
        )

    write_top_level_artifacts(
        output_dir=output_dir,
        args=args,
        subset_dir=subset_dir,
        subset_records=subset_records,
        commands=all_commands,
        summary_rows=summary_rows,
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "output_dir": str(output_dir.resolve()),
                "summary_csv": str((output_dir / "places2_5k_r_sweep_summary.csv").resolve()),
                "subset_size": len(subset_records),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
