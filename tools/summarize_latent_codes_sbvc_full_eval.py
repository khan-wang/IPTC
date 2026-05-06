#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--config-id", default="qkv_r64_last20")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def fmt(value: str | float | int, digits: int = 6) -> str:
    if value in ("", None):
        return "NA"
    return f"{float(value):.{digits}f}"


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    run_dir = Path(args.run_dir).resolve()
    eval_dir = Path(args.eval_dir).resolve()
    output_md = Path(args.output_md).resolve()
    output_json = Path(args.output_json).resolve() if args.output_json else output_md.with_suffix(".json")

    manifest_rows = read_csv(manifest_path)
    metric_rows = read_csv(eval_dir / "metrics_by_bin.csv")
    runtime_summary = json.loads((run_dir / "runtime_summary.json").read_text(encoding="utf-8"))
    sbvc_summary = runtime_summary.get("latent_sbvc_summary", {})

    payload = {
        "config_id": args.config_id,
        "manifest": str(manifest_path),
        "run_dir": str(run_dir),
        "eval_dir": str(eval_dir),
        "num_manifest_rows": len(manifest_rows),
        "runtime_summary": runtime_summary,
        "metrics_by_bin": metric_rows,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# Latent Codes-SBVC Full Places2 ReBinned36500 Evaluation",
        "",
        f"- Config: `{args.config_id}`",
        f"- Manifest: `{manifest_path}`",
        f"- Run dir: `{run_dir}`",
        f"- Eval dir: `{eval_dir}`",
        f"- Num images: `{len(manifest_rows)}`",
        f"- Completed dir: `{runtime_summary.get('completed_dir', '')}`",
        f"- Processed new: `{runtime_summary.get('processed_new', '')}`",
        f"- Skipped existing: `{runtime_summary.get('skipped_existing', '')}`",
        "",
        "## Quality By Mask Group",
        "",
        "| Mask group | N | PSNR | SSIM | LPIPS | FID status | FID |",
        "|---|---:|---:|---:|---:|---|---:|",
    ]
    for row in metric_rows:
        lines.append(
            "| {group} | {n} | {psnr} | {ssim} | {lpips} | {fid_status} | {fid} |".format(
                group=row["mask_ratio_group"],
                n=row["num_images"],
                psnr=fmt(row["psnr_mean"], 4),
                ssim=fmt(row["ssim_mean"], 6),
                lpips=fmt(row["lpips_mean"], 6),
                fid_status=row.get("fid_status", ""),
                fid=fmt(row.get("fid_mean", ""), 4),
            )
        )

    lines.extend(
        [
            "",
            "## SBVC Core Statistics",
            "",
            f"- Active stat images: `{sbvc_summary.get('num_active_stat_images', 0)}` / `{sbvc_summary.get('num_stat_images', 0)}`",
            f"- Core GMACs before total: `{fmt(sbvc_summary.get('core_gmacs_before_sum', 0.0), 6)}`",
            f"- Core GMACs after total: `{fmt(sbvc_summary.get('core_gmacs_after_sum', 0.0), 6)}`",
            f"- Core GMACs reduction total: `{fmt(sbvc_summary.get('core_gmacs_reduction_pct_total', 0.0), 4)}%`",
            f"- Actual compression ratio total: `{fmt(sbvc_summary.get('actual_compression_ratio_total', 0.0), 6)}`",
            f"- Removed tokens total: `{fmt(sbvc_summary.get('removed_tokens_sum_sum', 0.0), 0)}`",
            f"- Route cache hits total: `{fmt(sbvc_summary.get('route_cache_hits_sum', 0.0), 0)}`",
            "",
            "## Caveat",
            "",
            "Wall-time is recorded as an execution diagnostic only. This Latent Codes-SBVC run is evaluated for quality and Transformer-core GMACs, not for latency win.",
        ]
    )
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "output_md": str(output_md), "output_json": str(output_json)}, indent=2))


if __name__ == "__main__":
    main()
