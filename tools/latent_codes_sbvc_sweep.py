#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = REPO_ROOT / "tools"
DEFAULT_MANIFEST = REPO_ROOT / "outputs" / "places2_fv_reproduction" / "rebinned_full_11997" / "LatentCodes" / "selected_manifest.csv"
DEFAULT_ROW_INDICES = "2001,4005,8007"

if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import latent_codes_readonly_probe as probe  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Small Latent Codes SBVC smoke/R-sweep.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=str(probe.latent_adapter.DEFAULT_CONFIG))
    parser.add_argument("--row-indices", default=DEFAULT_ROW_INDICES)
    parser.add_argument("--r-values", default="0,16,32,64")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260502)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--sampling-ratio", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--temperature-degradation", type=float, default=0.9)
    parser.add_argument("--clamp-ratio", type=float, default=0.25)
    parser.add_argument("--sample", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sbvc-mode", choices=["qkv", "kv"], default="qkv")
    parser.add_argument("--layer-ids", default="all")
    parser.add_argument("--route-cache", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--boundary-radius", type=int, default=3)
    return parser.parse_args()


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


def cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def to_01(tensor: torch.Tensor) -> torch.Tensor:
    return ((tensor.detach().float() + 1.0) * 0.5).clamp(0.0, 1.0)


def final_composite(x: torch.Tensor, rec: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return x * mask + rec * (1.0 - mask)


def boundary_ring(mask: torch.Tensor, radius: int) -> torch.Tensor:
    known = mask > 0.5
    missing = ~known
    kernel = radius * 2 + 1
    missing_near = F.max_pool2d(missing.float(), kernel_size=kernel, stride=1, padding=radius) > 0
    known_near = F.max_pool2d(known.float(), kernel_size=kernel, stride=1, padding=radius) > 0
    return missing_near & known_near


def region_mae(a: torch.Tensor, b: torch.Tensor, region: torch.Tensor) -> float:
    region = region.expand_as(a)
    if not bool(region.any()):
        return float("nan")
    return float((a - b).abs()[region].mean().item())


def full_mae(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().mean().item())


def summarize_stats(stats: list[dict[str, Any]]) -> dict[str, float | int]:
    if not stats:
        return {
            "calls": 0,
            "enabled_calls": 0,
            "avg_original_tokens": 0.0,
            "avg_compact_tokens": 0.0,
            "avg_removed_tokens": 0.0,
            "avg_safe_tokens": 0.0,
            "avg_protect_tokens": 0.0,
            "score_elements_before": 0,
            "score_elements_after": 0,
            "score_reduction_pct": 0.0,
            "pair_ms": 0.0,
            "merge_ms": 0.0,
            "attention_ms": 0.0,
            "restore_ms": 0.0,
            "route_cache_hits": 0,
        }

    before = int(sum(int(row["score_elements_before"]) for row in stats))
    after = int(sum(int(row["score_elements_after"]) for row in stats))
    calls = len(stats)
    return {
        "calls": calls,
        "enabled_calls": int(sum(1 for row in stats if row["enabled"])),
        "avg_original_tokens": float(np.mean([row["original_tokens"] for row in stats])),
        "avg_compact_tokens": float(np.mean([row["compact_tokens"] for row in stats])),
        "avg_removed_tokens": float(np.mean([row["removed_tokens"] for row in stats])),
        "avg_safe_tokens": float(np.mean([row["safe_tokens"] for row in stats])),
        "avg_protect_tokens": float(np.mean([row["protect_tokens"] for row in stats])),
        "score_elements_before": before,
        "score_elements_after": after,
        "score_reduction_pct": float((1.0 - after / before) * 100.0) if before else 0.0,
        "pair_ms": float(sum(float(row["pair_ms"]) for row in stats)),
        "merge_ms": float(sum(float(row["merge_ms"]) for row in stats)),
        "attention_ms": float(sum(float(row["attention_ms"]) for row in stats)),
        "restore_ms": float(sum(float(row["restore_ms"]) for row in stats)),
        "route_cache_hits": int(sum(1 for row in stats if row.get("route_cache_hit"))),
    }


def run_with_env(args: argparse.Namespace, row: dict[str, str], model_parts, device: torch.device, mingpt, r: int, profile_enabled: bool):
    env = {
        "LATENT_SBVC_ENABLE": "1" if r > 0 else "0",
        "LATENT_SBVC_R": str(r),
        "LATENT_SBVC_PROFILE": "1" if profile_enabled else "0",
        "LATENT_SBVC_LAYER_IDS": args.layer_ids,
        "LATENT_SBVC_ROUTE_CACHE": "1" if args.route_cache else "0",
        "LATENT_SBVC_MODE": args.sbvc_mode,
    }
    with patched_env(env):
        if profile_enabled:
            mingpt.reset_latent_sbvc_stats()
            timer = probe.SectionTimer(device)
            token_info, outputs = probe.run_pipeline(args, row, model_parts, device, timer=timer)
            section_ms = timer.summary_ms()
            stats = mingpt.get_latent_sbvc_stats()
            return token_info, outputs, section_ms, stats, None

        cuda_sync(device)
        start = time.perf_counter()
        token_info, outputs = probe.run_pipeline(args, row, model_parts, device, timer=None)
        cuda_sync(device)
        wall_ms = (time.perf_counter() - start) * 1000.0
        return token_info, outputs, {}, [], wall_ms


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, summary_rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    lines = [
        "# Latent Codes SBVC Smoke Sweep",
        "",
        "## Scope",
        "- Default-off inference-time prototype.",
        "- Host: `core.modules.transformer.mingpt.CausalSelfAttention` only.",
        f"- Strategy: attention-local safe `{args.sbvc_mode}` merge; output leaves attention as `T=257`.",
        "- Protected tokens: SOS condition token and current unknown/masked latent tokens.",
        f"- Route cache: `{args.route_cache}`.",
        "",
        "## Summary By r",
        "| r | wall ms/img | attention ms/img | pair ms/img | merge ms/img | restore ms/img | removed | safe/protect | score reduction | raw known MAE | raw boundary MAE | final output MAE |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['r']} | {row['wall_ms_mean']:.3f} | {row['attention_ms_mean']:.3f} | "
            f"{row['pair_ms_mean']:.3f} | {row['merge_ms_mean']:.3f} | {row['restore_ms_mean']:.3f} | "
            f"{row['avg_removed_tokens']:.2f} | {row['avg_safe_tokens']:.2f}/{row['avg_protect_tokens']:.2f} | "
            f"{row['score_reduction_pct']:.2f}% | {row['raw_known_mae']:.8f} | "
            f"{row['raw_boundary_mae']:.8f} | {row['final_output_mae']:.8f} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "- MAE values are in `0~1` image space.",
            "- Final composite pastes known pixels from the input, so final known-region MAE is expected to be `0`; raw known MAE is computed before that final paste.",
            "- Profile timings use synchronized CUDA timing inside attention and are for diagnosis; wall time is measured in a separate profile-off pass.",
            f"- Command row indices: `{args.row_indices}`; r-values: `{args.r_values}`; layer ids: `{args.layer_ids}`.",
            f"- Route cache: `{args.route_cache}`.",
            f"- SBVC mode: `{args.sbvc_mode}`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_all = probe.latent_adapter.load_manifest(Path(args.manifest).resolve())
    row_indices = parse_int_list(args.row_indices)
    r_values = parse_int_list(args.r_values)
    if 0 not in r_values:
        r_values = [0] + r_values
    else:
        r_values = [0] + [value for value in r_values if value != 0]
    selected_rows = [rows_all[index] for index in row_indices]

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model_parts = probe.load_latent_codes_model(args, device)
    import core.modules.transformer.mingpt as mingpt

    if args.warmup > 0:
        for r in r_values:
            for _ in range(args.warmup):
                run_with_env(args, selected_rows[0], model_parts, device, mingpt, r, profile_enabled=False)

    per_image_rows: list[dict[str, Any]] = []
    baseline_cache: dict[str, dict[str, torch.Tensor]] = {}

    with torch.no_grad():
        for row in selected_rows:
            sample_id = row["sample_id"]
            for r in r_values:
                token_info, profile_outputs, section_ms, stats, _ = run_with_env(
                    args, row, model_parts, device, mingpt, r, profile_enabled=True
                )
                stats_summary = summarize_stats(stats)
                _token_info_wall, outputs, _section_wall, _stats_wall, wall_ms = run_with_env(
                    args, row, model_parts, device, mingpt, r, profile_enabled=False
                )

                x = outputs["x"]
                mask = outputs["mask"]
                raw = outputs["rec"]
                final = final_composite(x, raw, mask)
                if r == 0:
                    baseline_cache[sample_id] = {
                        "raw": raw.detach(),
                        "final": final.detach(),
                        "x": x.detach(),
                        "mask": mask.detach(),
                    }
                    first_vis = output_dir / f"{sample_id}_baseline_final.png"
                    probe.latent_adapter.to_uint8_image(final).save(first_vis)

                baseline = baseline_cache[sample_id]
                raw_01 = to_01(raw)
                final_01 = to_01(final)
                base_raw_01 = to_01(baseline["raw"])
                base_final_01 = to_01(baseline["final"])
                known = mask > 0.5
                ring = boundary_ring(mask, args.boundary_radius)
                final_known_mae = region_mae(final_01, base_final_01, known)
                raw_known_mae = region_mae(raw_01, base_raw_01, known)
                raw_boundary_mae = region_mae(raw_01, base_raw_01, ring)
                final_boundary_mae = region_mae(final_01, base_final_01, ring)
                final_output_mae = full_mae(final_01, base_final_01)
                raw_output_mae = full_mae(raw_01, base_raw_01)

                if r > 0 and sample_id == selected_rows[0]["sample_id"]:
                    probe.latent_adapter.to_uint8_image(final).save(output_dir / f"{sample_id}_r{r}_final.png")

                per_image_rows.append(
                    {
                        "sample_id": sample_id,
                        "mask_bin": row.get("bucket", ""),
                        "r": r,
                        "output_shape": list(raw.shape),
                        "z_tokens": token_info["z_token_length"],
                        "transformer_tokens": token_info["transformer_token_length"],
                        "known_latent_tokens": token_info["known_latent_tokens"],
                        "unknown_latent_tokens": token_info["unknown_latent_tokens"],
                        "sample_iterations": token_info["sample_iterations"],
                        "wall_ms": float(wall_ms),
                        "model_sections_profile_ms": float(sum(section_ms.values())),
                "sample_loop_profile_ms": float(section_ms.get("sample_loop_ms", 0.0)),
                **stats_summary,
                        "final_output_mae": final_output_mae,
                        "raw_output_mae": raw_output_mae,
                        "final_known_mae": final_known_mae,
                        "raw_known_mae": raw_known_mae,
                        "final_boundary_mae": final_boundary_mae,
                        "raw_boundary_mae": raw_boundary_mae,
                    }
                )

    summary_rows: list[dict[str, Any]] = []
    for r in r_values:
        rows_r = [row for row in per_image_rows if row["r"] == r]
        score_before = sum(int(row["score_elements_before"]) for row in rows_r)
        score_after = sum(int(row["score_elements_after"]) for row in rows_r)
        summary_rows.append(
            {
                "r": r,
                "images": len(rows_r),
                "wall_ms_mean": float(np.mean([row["wall_ms"] for row in rows_r])),
                "attention_ms_mean": float(np.mean([row["attention_ms"] for row in rows_r])),
                "pair_ms_mean": float(np.mean([row["pair_ms"] for row in rows_r])),
                "merge_ms_mean": float(np.mean([row["merge_ms"] for row in rows_r])),
                "restore_ms_mean": float(np.mean([row["restore_ms"] for row in rows_r])),
                "route_cache_hits_mean": float(np.mean([row["route_cache_hits"] for row in rows_r])),
                "avg_removed_tokens": float(np.mean([row["avg_removed_tokens"] for row in rows_r])),
                "avg_safe_tokens": float(np.mean([row["avg_safe_tokens"] for row in rows_r])),
                "avg_protect_tokens": float(np.mean([row["avg_protect_tokens"] for row in rows_r])),
                "score_elements_before": int(score_before),
                "score_elements_after": int(score_after),
                "score_reduction_pct": float((1.0 - score_after / score_before) * 100.0) if score_before else 0.0,
                "final_output_mae": float(np.mean([row["final_output_mae"] for row in rows_r])),
                "raw_output_mae": float(np.mean([row["raw_output_mae"] for row in rows_r])),
                "final_known_mae": float(np.mean([row["final_known_mae"] for row in rows_r])),
                "raw_known_mae": float(np.mean([row["raw_known_mae"] for row in rows_r])),
                "final_boundary_mae": float(np.mean([row["final_boundary_mae"] for row in rows_r])),
                "raw_boundary_mae": float(np.mean([row["raw_boundary_mae"] for row in rows_r])),
            }
        )

    write_csv(output_dir / "per_image_metrics.csv", per_image_rows)
    write_csv(output_dir / "summary_by_r.csv", summary_rows)
    summary = {
        "status": "ok",
        "manifest": str(Path(args.manifest).resolve()),
        "row_indices": row_indices,
        "r_values": r_values,
        "device": str(device),
        "seed": args.seed,
        "sbvc_mode": args.sbvc_mode,
        "summary_by_r": summary_rows,
        "paths": {
            "per_image_metrics": str(output_dir / "per_image_metrics.csv"),
            "summary_by_r": str(output_dir / "summary_by_r.csv"),
            "report": str(output_dir / "REPORT.md"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_report(output_dir / "REPORT.md", summary_rows, args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
