#!/usr/bin/env python3
"""Phase 5C pilot: controlled selective fine-tune and standardized re-evaluation."""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from phase5a_standardized_comparison import (
    REPORT_GROUPS,
    TIMING_BUCKETS,
    aggregate_group_repeat_metrics,
    build_boundary_visuals,
    build_comparison_rows,
    build_sample_group_map,
    compute_diversity_metrics,
    compute_group_complexity,
    compute_single_output_metrics,
    ensure_parent,
    generate_diversity_outputs,
    load_model,
    parse_hw,
    run_dataset_pass,
    set_seed,
    stats,
    summarize_group_repeat_metrics,
    summarize_repeat,
    summarize_variant_repeats,
    to_serializable,
    write_comparison_csv,
)


IMAGE_EXTENSIONS = "bmp,jpg,jpeg,pgm,png,ppm,tif,tiff,webp,JPEG"
DEFAULT_TRAIN_STROKE = {
    "maxVertex": 10,
    "minVertex": 5,
    "maxLength": 100,
    "maxBrushWidth": 30,
    "minBrushWidth": 10,
    "keep_ratio": [0.2, 0.6],
    "min_area": 64,
    "minRectangle": 0,
    "maxRectangle": 3,
}
DEFAULT_VAL_STROKE = {
    "maxVertex": 10,
    "minVertex": 5,
    "maxLength": 100,
    "maxBrushWidth": 30,
    "minBrushWidth": 10,
    "keep_ratio": [0.2, 0.6],
    "min_area": 64,
    "minRectangle": 0,
    "maxRectangle": 3,
}
LEGACY_TRAIN_ITER_PATTERN = re.compile(
    r"Iter (?P<global_iter>\d+)/(?P<global_total>\d+)"
    r"(?: epoch (?P<epoch>\d+) iter (?P<iter>\d+)/(?P<iter_total>\d+))?"
)
EPOCH_ITER_PATTERN = re.compile(
    r"Epoch (?P<epoch>\d+)/(?P<epoch_total>\d+) iter (?P<iter>\d+)/(?P<iter_total>\d+)"
)
EPOCH_ONLY_PATTERN = re.compile(r"Epoch (?P<epoch>\d+)/(?P<epoch_total>\d+)")
KV_PATTERN = re.compile(r"([A-Za-z0-9_]+): ([^|]+)")
REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--put-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--train-root",
        default=os.environ.get("SBVC_TRAIN_ROOT", ""),
    )
    parser.add_argument(
        "--mask-root",
        default=os.environ.get("SBVC_MASK_ROOT", ""),
    )
    parser.add_argument(
        "--sample-manifest",
        default=str(REPO_ROOT / "runs" / "protocol" / "sample_manifest.csv"),
    )
    parser.add_argument(
        "--phase5a-summary",
        default=str(REPO_ROOT / "runs" / "protocol" / "phase5a_summary.json"),
    )
    parser.add_argument(
        "--official-train-list",
        default=str(REPO_ROOT / "third_party" / "PUT" / "data" / "naturalscenetrain.txt"),
    )
    parser.add_argument(
        "--official-transformer-ckpt",
        default=os.environ.get("PUT_CHECKPOINT", ""),
    )
    parser.add_argument(
        "--official-pvqvae-ckpt",
        default=os.environ.get("PUT_PVQVAE_CHECKPOINT", ""),
    )
    parser.add_argument("--input-res", default="256,256")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260423)
    parser.add_argument("--pilot-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--base-lr", type=float, default=1.0e-5)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--measure-runs", type=int, default=3)
    parser.add_argument("--num-token-per-iter", type=int, default=20)
    parser.add_argument("--num-token-for-sampling", type=int, default=200)
    parser.add_argument("--diversity-num-samples", type=int, default=4)
    parser.add_argument("--boundary-ring-radius", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.15)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--topk-ratio", type=float, default=0.25)
    parser.add_argument("--boundary-zoom-cell", type=int, default=192)
    parser.add_argument("--boundary-zoom-pad", type=int, default=48)
    parser.add_argument("--log-frequency", type=int, default=50)
    parser.add_argument("--smoke-iterations", type=int, default=0)
    parser.add_argument("--smoke-arm", default="sbvc_ft_r192")
    parser.add_argument("--skip-training", action="store_true", default=False)
    parser.add_argument("--eval-only", action="store_true", default=False)
    return parser.parse_args()


def ensure_symlink(link_path: Path, target_path: Path) -> None:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.is_symlink() or link_path.exists():
        resolved = link_path.resolve()
        if resolved == target_path.resolve():
            return
        if link_path.is_symlink() or link_path.is_file():
            link_path.unlink()
        else:
            shutil.rmtree(link_path)
    os.symlink(target_path, link_path)


def load_sample_records(sample_manifest_path: Path) -> list[dict[str, Any]]:
    with sample_manifest_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        records = []
        for row in reader:
            row["mask_ratio"] = float(row["mask_ratio"])
            records.append(row)
    return records


def map_official_naturalscene_relpath(relpath: str) -> str:
    parts = relpath.split("/")
    if len(parts) < 3:
        raise ValueError(f"unexpected naturalscene relpath: {relpath}")
    cls_name = "-".join(parts[1:-1])
    return f"{cls_name}/{parts[-1]}"


def build_train_overlap_manifest(
    official_train_list: Path,
    train_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    relpaths = [line.strip() for line in official_train_list.read_text(encoding="utf-8").splitlines() if line.strip()]
    kept_relpaths: list[str] = []
    missing_examples: list[dict[str, str]] = []
    class_counter: Counter[str] = Counter()
    for relpath in relpaths:
        mapped = map_official_naturalscene_relpath(relpath)
        if (train_root / mapped).is_file():
            kept_relpaths.append(mapped)
            class_counter[mapped.split("/")[0]] += 1
        elif len(missing_examples) < 50:
            missing_examples.append({"official_relpath": relpath, "mapped_relpath": mapped})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(kept_relpaths) + "\n", encoding="utf-8")
    return {
        "official_total": len(relpaths),
        "local_overlap": len(kept_relpaths),
        "coverage_ratio": float(len(kept_relpaths) / max(len(relpaths), 1)),
        "class_counts": dict(sorted(class_counter.items())),
        "missing_examples": missing_examples,
        "overlap_manifest": str(output_path.resolve()),
    }


def prepare_data_root(
    output_dir: Path,
    train_root: Path,
    mask_root: Path,
    sample_records: list[dict[str, Any]],
) -> dict[str, str]:
    data_root = output_dir / "data_root"
    train_link = data_root / "train"
    mask_link = data_root / "irregular-mask" / "testing_mask_dataset"
    val_image_link = data_root / "val250" / "images"
    val_mask_link = data_root / "val250" / "masks"
    ensure_symlink(train_link, train_root)
    ensure_symlink(mask_link, mask_root)
    ensure_symlink(val_image_link, Path(sample_records[0]["resized_image_path"]).parent)
    ensure_symlink(val_mask_link, Path(sample_records[0]["resized_mask_path"]).parent)
    return {
        "data_root": str(data_root.resolve()),
        "train_link": str(train_link.resolve()),
        "mask_link": str(mask_link.resolve()),
        "val_image_link": str(val_image_link.resolve()),
        "val_mask_link": str(val_mask_link.resolve()),
    }


def build_base_config(
    args: argparse.Namespace,
    data_root: Path,
    train_manifest: Path,
) -> dict[str, Any]:
    return {
        "model": {
            "target": "image_synthesis.modeling.models.masked_image_inpainting_transformer.MaskedImageInpaintingTransformer",
            "params": {
                "content_seq_len": 1024,
                "n_layer": 12,
                "dim": 768,
                "num_heads": 12,
                "learn_mask_emb": True,
                "num_token": 8192,
                "act_layer": "GELU2",
                "input_feature_type": "origin",
                "attn_content_with_mask": False,
                "drop_path": 0.1,
                "random_quantize": 0.3,
                "weight_decay": 0.05,
                "ckpt_path": str(Path(args.official_transformer_ckpt).resolve()),
                "loss_config": {
                    "target": "image_synthesis.modeling.modules.losses.poly_loss.PolyLoss",
                    "params": {"epsilons": [1]},
                },
                "content_codec_config": {
                    "target": "image_synthesis.modeling.codecs.image_codec.patch_vqgan.PatchVQGAN",
                    "params": {
                        "ckpt_path": str(Path(args.official_pvqvae_ckpt).resolve()),
                        "trainable": False,
                        "token_shape": [32, 32],
                        "quantizer_config": {
                            "target": "image_synthesis.modeling.codecs.image_codec.patch_vqgan.VectorQuantizer",
                            "params": {
                                "n_e": 9216,
                                "e_dim": 256,
                                "masked_embed_start": 8192,
                                "embed_ema": True,
                                "get_embed_type": "retrive",
                            },
                        },
                        "encoder_config": {
                            "target": "image_synthesis.modeling.codecs.image_codec.patch_vqgan.PatchEncoder2",
                            "params": {
                                "in_ch": 3,
                                "res_ch": 256,
                                "out_ch": 256,
                                "num_res_block": 8,
                                "stride": 8,
                            },
                        },
                        "decoder_config": {
                            "target": "image_synthesis.modeling.codecs.image_codec.patch_vqgan.PatchConvDecoder2",
                            "params": {
                                "in_ch": 256,
                                "out_ch": 3,
                                "res_ch": 256,
                                "num_res_block": 2,
                                "num_res_block_after_resolution_change": 2,
                                "stride": 8,
                                "up_layer_with_image": True,
                                "upsample_type": "nearest",
                            },
                        },
                    },
                },
            },
        },
        "solver": {
            "find_unused_parameters": False,
            "base_lr": float(args.base_lr),
            "adjust_lr": "none",
            "max_epochs": int(args.pilot_epochs),
            "save_epochs": 1,
            "validation_epochs": 1,
            "sample_iterations": -1,
            "optimizers_and_schedulers": [
                {
                    "name": "transformer",
                    "optimizer": {
                        "target": "torch.optim.AdamW",
                        "params": {
                            "betas": (0.9, 0.999),
                        },
                    },
                }
            ],
        },
        "dataloader": {
            "data_root": str(data_root.resolve()),
            "batch_size": int(args.batch_size),
            "num_workers": int(args.num_workers),
            "train_datasets": [
                {
                    "target": "image_synthesis.data.image_list_dataset.ImageListDataset",
                    "params": {
                        "name": "train",
                        "image_list_file": str(train_manifest.resolve()),
                        "image_end_with": IMAGE_EXTENSIONS,
                        "provided_mask_name": "irregular-mask/testing_mask_dataset",
                        "use_provided_mask": 1.0,
                        "use_provided_mask_ratio": ["0.2", "0.6"],
                        "mask": 1.0,
                        "mask_low_to_high": 0.0,
                        "mask_low_size": [32, 32],
                        "zero_mask": 0.0,
                        "multi_image_mask": False,
                        "return_data_keys": ["image", "mask"],
                        "stroken_mask_params": copy.deepcopy(DEFAULT_TRAIN_STROKE),
                        "im_preprocessor_config": {
                            "target": "image_synthesis.data.utils.image_preprocessor.SimplePreprocessor",
                            "params": {
                                "size": list(parse_hw(args.input_res)),
                                "smallest_max_size": 272,
                                "random_crop": True,
                                "horizon_flip": True,
                            },
                        },
                    },
                }
            ],
            "validation_datasets": [
                {
                    "target": "image_synthesis.data.image_list_dataset.ImageListDataset",
                    "params": {
                        "name": "val250/images",
                        "image_end_with": "png",
                        "provided_mask_name": "val250/masks",
                        "use_provided_mask": 1.0,
                        "use_provided_mask_ratio": [0.0, 1.0],
                        "image_mask_paired": True,
                        "mask": 1.0,
                        "mask_low_to_high": 0.0,
                        "mask_low_size": [32, 32],
                        "zero_mask": 0.0,
                        "multi_image_mask": False,
                        "return_data_keys": ["image", "mask"],
                        "stroken_mask_params": copy.deepcopy(DEFAULT_VAL_STROKE),
                        "im_preprocessor_config": {
                            "target": "image_synthesis.data.utils.image_preprocessor.SimplePreprocessor",
                            "params": {
                                "size": list(parse_hw(args.input_res)),
                                "smallest_max_size": 256,
                                "random_crop": False,
                                "horizon_flip": False,
                            },
                        },
                    },
                }
            ],
        },
    }


def arm_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    return [
        {
            "name": "baseline_ft",
            "label": "Baseline-FT",
            "global_tome_r": 0,
            "boundary_split": False,
            "similarity_scorer": False,
            "similarity_learnable": False,
            "safe_tome_r": 0,
            "safe_tome_score_mode": "similarity",
        },
        {
            "name": "pure_distance_ft_r192",
            "label": "PureDistance-FT-r192",
            "global_tome_r": 0,
            "boundary_split": True,
            "similarity_scorer": False,
            "similarity_learnable": False,
            "safe_tome_r": 192,
            "safe_tome_score_mode": "distance",
        },
        {
            "name": "sbvc_ft_r192",
            "label": "SBVC-FT-r192",
            "global_tome_r": 0,
            "boundary_split": True,
            "similarity_scorer": True,
            "similarity_learnable": True,
            "safe_tome_r": 192,
            "safe_tome_score_mode": "similarity",
        },
    ]


def set_arm_env(spec: dict[str, Any], args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(args.put_root).resolve())
    env["PUT_GLOBAL_TOME_R"] = str(spec["global_tome_r"])
    env["PUT_BOUNDARY_SPLIT"] = "1" if spec["boundary_split"] else "0"
    env["PUT_BOUNDARY_RING_RADIUS"] = str(args.boundary_ring_radius)
    env["PUT_SIMILARITY_SCORER"] = "1" if spec["similarity_scorer"] else "0"
    env["PUT_SIMILARITY_LEARNABLE"] = "1" if spec["similarity_learnable"] else "0"
    env["PUT_SIMILARITY_ALPHA"] = str(args.alpha)
    env["PUT_SIMILARITY_BETA"] = str(args.beta)
    env["PUT_SIMILARITY_KERNEL_SIZE"] = str(args.kernel_size)
    env["PUT_SIMILARITY_TOPK_RATIO"] = str(args.topk_ratio)
    env["PUT_SAFE_TOME_R"] = str(spec["safe_tome_r"])
    env["PUT_SAFE_TOME_SCORE_MODE"] = spec["safe_tome_score_mode"]
    env["PUT_SAFE_TOME_DEBUG"] = "0"
    env["PUT_PHASE4A_TIMING"] = "1"
    return env


def write_yaml_config(config: dict[str, Any], path: Path) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def run_command(command: list[str], cwd: Path, env: dict[str, str]) -> tuple[float, int]:
    start = time.time()
    result = subprocess.run(command, cwd=str(cwd), env=env, check=False)
    return time.time() - start, int(result.returncode)


def parse_training_log(log_path: Path, arm_name: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not log_path.is_file():
        return records
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if ": train:" not in line and ": val:" not in line:
            continue
        timestamp, _, payload = line.partition(": ")
        phase = "train" if ": train:" in payload else "val"
        epoch = None
        epoch_total = None
        iter_index = None
        iter_total = None
        global_iter = None
        global_total = None

        legacy_match = LEGACY_TRAIN_ITER_PATTERN.search(payload)
        if legacy_match is not None:
            epoch = int(legacy_match.group("epoch") or 0)
            iter_index = int(legacy_match.group("iter") or legacy_match.group("global_iter"))
            iter_total = int(legacy_match.group("iter_total") or legacy_match.group("global_total"))
            global_iter = int(legacy_match.group("global_iter"))
            global_total = int(legacy_match.group("global_total"))
        else:
            epoch_match = EPOCH_ITER_PATTERN.search(payload) if phase == "train" else EPOCH_ONLY_PATTERN.search(payload)
            if epoch_match is None:
                continue
            epoch = int(epoch_match.group("epoch"))
            epoch_total = int(epoch_match.group("epoch_total"))
            if phase == "train":
                iter_index = int(epoch_match.group("iter"))
                iter_total = int(epoch_match.group("iter_total"))

        record: dict[str, Any] = {
            "arm": arm_name,
            "timestamp": timestamp,
            "phase": phase,
            "epoch": epoch,
            "epoch_total": epoch_total,
            "iter": iter_index,
            "iter_total": iter_total,
        }
        if global_iter is not None:
            record["global_iter"] = global_iter
        if global_total is not None:
            record["global_total"] = global_total
        for key, value in KV_PATTERN.findall(payload):
            value = value.strip()
            numeric = value.rstrip("s")
            try:
                record[key] = float(numeric)
            except ValueError:
                continue
        records.append(record)
    return records


def summarize_smoke(log_records: list[dict[str, Any]], elapsed_sec: float, max_iterations: int) -> dict[str, Any]:
    train_records = [record for record in log_records if record["phase"] == "train"]
    if len(train_records) == 0:
        return {
            "iterations_target": max_iterations,
            "elapsed_sec": elapsed_sec,
            "mean_logged_iter_time_sec": None,
            "estimated_epoch_hours": None,
        }
    iter_times = [record.get("iter_avg_time") for record in train_records if record.get("iter_avg_time") is not None]
    mean_iter_time = float(np.mean(iter_times)) if len(iter_times) > 0 else float(elapsed_sec / max(max_iterations, 1))
    return {
        "iterations_target": max_iterations,
        "elapsed_sec": float(elapsed_sec),
        "mean_logged_iter_time_sec": mean_iter_time,
        "estimated_epoch_hours": None,
    }


def load_phase5a_reference(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def find_variant(summary: dict[str, Any], name: str) -> dict[str, Any] | None:
    for variant in summary.get("variants", []):
        if variant.get("name") == name:
            return variant
    return None


def compare_with_phase5a(
    phase5a_summary: dict[str, Any],
    phase5c_variants: list[dict[str, Any]],
) -> dict[str, Any]:
    mapping = {
        "baseline_ft": "put_baseline",
        "sbvc_ft_r192": "sbvc_r192",
    }
    out: dict[str, Any] = {}
    for variant in phase5c_variants:
        ref_name = mapping.get(variant["name"])
        if ref_name is None:
            continue
        ref = find_variant(phase5a_summary, ref_name)
        if ref is None:
            continue
        out[variant["name"]] = {}
        for group in REPORT_GROUPS:
            out[variant["name"]][group] = {
                "PSNR_delta": float(
                    variant["quality"]["groups"][group]["psnr"]["mean"] - ref["quality"]["groups"][group]["psnr"]["mean"]
                ),
                "SSIM_delta": float(
                    variant["quality"]["groups"][group]["ssim"]["mean"] - ref["quality"]["groups"][group]["ssim"]["mean"]
                ),
                "LPIPS_output_vs_gt_delta": float(
                    variant["quality"]["groups"][group]["lpips_output_vs_gt"]["mean"]
                    - ref["quality"]["groups"][group]["lpips_output_vs_gt"]["mean"]
                ),
                "latency_delta_sec": float(
                    variant["group_timing_summary"][group]["wall_time_sec"]["mean"]
                    - ref["group_timing_summary"][group]["wall_time_sec"]["mean"]
                ),
            }
    baseline_ref = find_variant(phase5a_summary, "put_baseline")
    sbvc_ref = find_variant(phase5a_summary, "sbvc_r192")
    baseline_ft = next((item for item in phase5c_variants if item["name"] == "baseline_ft"), None)
    sbvc_ft = next((item for item in phase5c_variants if item["name"] == "sbvc_ft_r192"), None)
    if baseline_ref and sbvc_ref and baseline_ft and sbvc_ft:
        out["sbvc_vs_baseline_gap_amplification"] = {}
        for group in REPORT_GROUPS:
            zero_shot_gap = (
                sbvc_ref["quality"]["groups"][group]["psnr"]["mean"]
                - baseline_ref["quality"]["groups"][group]["psnr"]["mean"]
            )
            finetune_gap = (
                sbvc_ft["quality"]["groups"][group]["psnr"]["mean"]
                - baseline_ft["quality"]["groups"][group]["psnr"]["mean"]
            )
            out["sbvc_vs_baseline_gap_amplification"][group] = {
                "zero_shot_psnr_gap": float(zero_shot_gap),
                "finetune_psnr_gap": float(finetune_gap),
                "gap_delta": float(finetune_gap - zero_shot_gap),
            }
    return out


def evaluate_acceptance(variants: list[dict[str, Any]]) -> dict[str, Any]:
    by_name = {variant["name"]: variant for variant in variants}
    baseline = by_name["baseline_ft"]
    pure_distance = by_name["pure_distance_ft_r192"]
    sbvc = by_name["sbvc_ft_r192"]

    def group_quality(variant: dict[str, Any], group: str) -> dict[str, float]:
        return {
            "psnr": float(variant["quality"]["groups"][group]["psnr"]["mean"]),
            "ssim": float(variant["quality"]["groups"][group]["ssim"]["mean"]),
            "lpips": float(variant["quality"]["groups"][group]["lpips_output_vs_gt"]["mean"]),
        }

    quality_vs_baseline = {}
    quality_vs_pure = {}
    for group in REPORT_GROUPS:
        sbvc_q = group_quality(sbvc, group)
        base_q = group_quality(baseline, group)
        pure_q = group_quality(pure_distance, group)
        quality_vs_baseline[group] = {
            "psnr_better": sbvc_q["psnr"] > base_q["psnr"],
            "ssim_better": sbvc_q["ssim"] > base_q["ssim"],
            "lpips_better": sbvc_q["lpips"] < base_q["lpips"],
        }
        quality_vs_pure[group] = {
            "psnr_better": sbvc_q["psnr"] > pure_q["psnr"],
            "ssim_better": sbvc_q["ssim"] > pure_q["ssim"],
            "lpips_better": sbvc_q["lpips"] < pure_q["lpips"],
        }

    bad_restore = 0
    for variant in variants:
        for record in variant["artifact_records"]:
            profile = record.get("safe_tome_profile") or {}
            if profile and not bool(profile.get("restore_alignment_ok", True)):
                bad_restore += 1

    sbvc_overall = quality_vs_baseline["10-60"]
    sbvc_overall_pure = quality_vs_pure["10-60"]
    recommend_full_validation = (
        all(sbvc_overall.values())
        and all(sbvc_overall_pure.values())
        and bad_restore == 0
    )
    return {
        "quality_vs_baseline": quality_vs_baseline,
        "quality_vs_pure_distance": quality_vs_pure,
        "bad_restore": int(bad_restore),
        "shape_or_sampling_crash": False,
        "recommend_full_validation": bool(recommend_full_validation),
    }


def write_metrics_md(
    output_path: Path,
    summary: dict[str, Any],
    phase5a_deltas: dict[str, Any],
    acceptance: dict[str, Any],
) -> None:
    lines = []
    lines.append("# Phase 5C Pilot Metrics Summary")
    lines.append("")
    lines.append("## Core Table (10-60)")
    lines.append("")
    lines.append("| Variant | PSNR | SSIM | LPIPS_output_vs_gt | LPIPS_pairwise_diversity | Latency (s/image) | Peak Mem (MiB) | Core GMACs |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for variant in summary["variants"]:
        group = "10-60"
        div_group = variant["diversity"]["groups"][group] if variant["diversity"] is not None else None
        lines.append(
            "| {label} | {psnr:.4f} | {ssim:.4f} | {lpips:.4f} | {div:.4f} | {lat:.4f} +/- {lat_std:.4f} | {mem:.2f} | {gmacs:.2f} |".format(
                label=variant["label"],
                psnr=variant["quality"]["groups"][group]["psnr"]["mean"],
                ssim=variant["quality"]["groups"][group]["ssim"]["mean"],
                lpips=variant["quality"]["groups"][group]["lpips_output_vs_gt"]["mean"],
                div=0.0 if div_group is None else div_group["lpips_pairwise_diversity"]["mean"],
                lat=variant["group_timing_summary"][group]["wall_time_sec"]["mean"],
                lat_std=variant["group_timing_summary"][group]["wall_time_sec"]["std"],
                mem=variant["timing_summary"]["peak_memory_allocated_mb"]["mean"],
                gmacs=variant["complexity"][group]["core_gmacs"]["mean"],
            )
        )
    lines.append("")
    lines.append("## Acceptance Check")
    lines.append("")
    lines.append(f"- bad_restore: `{acceptance['bad_restore']}`")
    lines.append(f"- shape_or_sampling_crash: `{acceptance['shape_or_sampling_crash']}`")
    lines.append(f"- recommend_full_validation: `{acceptance['recommend_full_validation']}`")
    lines.append("")
    lines.append("## Phase 5A Delta")
    lines.append("")
    if len(phase5a_deltas) == 0:
        lines.append("- no comparable Phase 5A reference was found")
    else:
        for variant_name, group_delta in phase5a_deltas.items():
            if variant_name == "sbvc_vs_baseline_gap_amplification":
                lines.append(f"- {variant_name}: {json.dumps(group_delta, ensure_ascii=False)}")
            else:
                lines.append(f"- {variant_name}: {json.dumps(group_delta, ensure_ascii=False)}")
    lines.append("")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_commands_md(output_path: Path, commands: list[str]) -> None:
    lines = ["# Phase 5C Commands", ""]
    for command in commands:
        lines.append(f"- `{command}`")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_training_curves(output_path: Path, records: list[dict[str, Any]]) -> None:
    if len(records) == 0:
        output_path.write_text("arm,phase,epoch,iter\n", encoding="utf-8")
        return
    fieldnames = sorted({key for record in records for key in record.keys()})
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def checkpoint_manifest_for_run(run_dir: Path) -> dict[str, Any]:
    ckpt_dir = run_dir / "checkpoint"
    checkpoints = sorted(ckpt_dir.glob("*.pth"))
    return {
        "run_dir": str(run_dir.resolve()),
        "checkpoints": [
            {
                "path": str(path.resolve()),
                "size_bytes": int(path.stat().st_size),
                "mtime": float(path.stat().st_mtime),
            }
            for path in checkpoints
        ],
        "last_checkpoint": str((ckpt_dir / "last.pth").resolve()) if (ckpt_dir / "last.pth").is_file() else None,
    }


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    put_root = Path(args.put_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_manifest_path = Path(args.sample_manifest).resolve()
    sample_records = load_sample_records(sample_manifest_path)
    sample_map = build_sample_group_map(sample_records)
    subset_image_dir = Path(sample_records[0]["resized_image_path"]).parent
    subset_mask_dir = Path(sample_records[0]["resized_mask_path"]).parent
    input_res = parse_hw(args.input_res)

    train_overlap_info = build_train_overlap_manifest(
        official_train_list=Path(args.official_train_list).resolve(),
        train_root=Path(args.train_root).resolve(),
        output_path=output_dir / "manifests" / "naturalscene_train_overlap_40k.txt",
    )
    data_root_info = prepare_data_root(
        output_dir=output_dir,
        train_root=Path(args.train_root).resolve(),
        mask_root=Path(args.mask_root).resolve(),
        sample_records=sample_records,
    )
    base_config = build_base_config(
        args=args,
        data_root=Path(data_root_info["data_root"]),
        train_manifest=Path(train_overlap_info["overlap_manifest"]),
    )
    arms = arm_specs(args)
    phase5a_summary = load_phase5a_reference(Path(args.phase5a_summary).resolve())
    commands: list[str] = []
    curve_records: list[dict[str, Any]] = []
    run_summaries: list[dict[str, Any]] = []

    config_manifest = {
        "phase": "Phase 5C selective fine-tune r192 pilot",
        "seed": int(args.seed),
        "put_root": str(put_root),
        "official_transformer_ckpt": str(Path(args.official_transformer_ckpt).resolve()),
        "official_pvqvae_ckpt": str(Path(args.official_pvqvae_ckpt).resolve()),
        "official_train_list": str(Path(args.official_train_list).resolve()),
        "train_root": str(Path(args.train_root).resolve()),
        "mask_root": str(Path(args.mask_root).resolve()),
        "sample_manifest": str(sample_manifest_path),
        "data_root": data_root_info,
        "train_overlap": train_overlap_info,
        "pilot_epochs": int(args.pilot_epochs),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "base_lr": float(args.base_lr),
        "input_res": list(input_res),
        "warmup_runs": int(args.warmup_runs),
        "measure_runs": int(args.measure_runs),
        "num_token_per_iter": int(args.num_token_per_iter),
        "num_token_for_sampling": int(args.num_token_for_sampling),
        "diversity_num_samples": int(args.diversity_num_samples),
        "arms": arms,
    }
    (output_dir / "config_manifest.json").write_text(
        json.dumps(to_serializable(config_manifest), indent=2),
        encoding="utf-8",
    )

    if args.smoke_iterations > 0 and not args.eval_only:
        smoke_spec = next(spec for spec in arms if spec["name"] == args.smoke_arm)
        smoke_dir = output_dir / "smoke" / smoke_spec["name"]
        smoke_config = copy.deepcopy(base_config)
        smoke_config["solver"]["max_epochs"] = -1
        smoke_config["solver"]["max_iterations"] = int(args.smoke_iterations)
        smoke_config["solver"]["save_epochs"] = -1
        smoke_config["solver"]["save_iterations"] = -1
        smoke_config["solver"]["validation_epochs"] = -1
        smoke_config["solver"]["sample_iterations"] = -1
        smoke_config_path = smoke_dir / "config.yaml"
        write_yaml_config(smoke_config, smoke_config_path)
        smoke_env = set_arm_env(smoke_spec, args)
        smoke_command = [
            sys.executable,
            str(put_root / "train_net.py"),
            "--name",
            smoke_spec["name"],
            "--config_file",
            str(smoke_config_path),
            "--output",
            str(smoke_dir),
            "--gpu",
            str(args.gpu),
            "--seed",
            str(args.seed),
            "--log_frequency",
            str(args.log_frequency),
            "--amp",
            "--auto_resume",
        ]
        commands.append(" ".join(smoke_command))
        elapsed_sec, return_code = run_command(smoke_command, cwd=put_root, env=smoke_env)
        if return_code != 0:
            raise RuntimeError(f"smoke training failed for {smoke_spec['name']} with code {return_code}")
        smoke_log = smoke_dir / smoke_spec["name"] / "logs" / "log.txt"
        smoke_records = parse_training_log(smoke_log, smoke_spec["name"])
        smoke_summary = summarize_smoke(smoke_records, elapsed_sec=elapsed_sec, max_iterations=args.smoke_iterations)
        train_iterations_per_epoch = math.ceil(train_overlap_info["local_overlap"] / args.batch_size)
        if smoke_summary["mean_logged_iter_time_sec"] is not None:
            smoke_summary["estimated_epoch_hours"] = float(
                smoke_summary["mean_logged_iter_time_sec"] * train_iterations_per_epoch / 3600.0
            )
        (output_dir / "smoke_estimate.json").write_text(
            json.dumps(to_serializable(smoke_summary), indent=2),
            encoding="utf-8",
        )
        if args.skip_training:
            write_commands_md(output_dir / "COMMANDS.md", commands)
            save_training_curves(output_dir / "training_curves.csv", smoke_records)
            return

    if not args.skip_training and not args.eval_only:
        for spec in arms:
            run_output_root = output_dir / "runs"
            config = copy.deepcopy(base_config)
            config_path = output_dir / "generated_configs" / f"{spec['name']}.yaml"
            write_yaml_config(config, config_path)
            env = set_arm_env(spec, args)
            command = [
                sys.executable,
                str(put_root / "train_net.py"),
                "--name",
                spec["name"],
                "--config_file",
                str(config_path),
                "--output",
                str(run_output_root),
                "--gpu",
                str(args.gpu),
                "--seed",
                str(args.seed),
                "--log_frequency",
                str(args.log_frequency),
                "--amp",
                "--auto_resume",
            ]
            commands.append(" ".join(command))
            elapsed_sec, return_code = run_command(command, cwd=put_root, env=env)
            if return_code != 0:
                raise RuntimeError(f"training failed for {spec['name']} with code {return_code}")
            run_dir = run_output_root / spec["name"]
            log_records = parse_training_log(run_dir / "logs" / "log.txt", spec["name"])
            curve_records.extend(log_records)
            run_summaries.append(
                {
                    "name": spec["name"],
                    "label": spec["label"],
                    "elapsed_sec": float(elapsed_sec),
                    "run_dir": str(run_dir.resolve()),
                    "log_path": str((run_dir / "logs" / "log.txt").resolve()),
                    "checkpoint_manifest": checkpoint_manifest_for_run(run_dir),
                }
            )

    save_training_curves(output_dir / "training_curves.csv", curve_records)
    write_commands_md(output_dir / "COMMANDS.md", commands)

    sys.path.insert(0, str(put_root))
    os.chdir(put_root)
    from scripts.inference import ImagePathDataset
    import lpips

    set_seed(args.seed)
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    dataset = ImagePathDataset(str(subset_image_dir), str(subset_mask_dir), size=input_res)
    lpips_model = lpips.LPIPS(net="vgg", spatial=True).to(device)

    eval_summary = {
        "config_manifest": config_manifest,
        "sample_manifest_csv": str(sample_manifest_path),
        "variants": [],
    }
    checkpoint_manifest = {}
    for spec in arms:
        checkpoint_path = output_dir / "runs" / spec["name"] / "checkpoint" / "last.pth"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"missing checkpoint for {spec['name']}: {checkpoint_path}")
        checkpoint_manifest[spec["name"]] = checkpoint_manifest_for_run(output_dir / "runs" / spec["name"])
        variant_dir = output_dir / spec["name"]
        completed_dir = variant_dir / "completed_single"
        masked_dir = variant_dir / "masked_gt"
        diversity_dir = variant_dir / "diversity"
        completed_dir.mkdir(parents=True, exist_ok=True)
        masked_dir.mkdir(parents=True, exist_ok=True)
        diversity_dir.mkdir(parents=True, exist_ok=True)

        env = set_arm_env(spec, args)
        os.environ.update(env)
        set_seed(args.seed)
        model = load_model(put_root, str(checkpoint_path), device)
        dim = int(model.dim)
        hidden_dim = int(model.blocks[0].mlp.fc1.out_features)
        num_layers = int(len(model.blocks))

        for _ in range(args.warmup_runs):
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
        generate_diversity_outputs(
            model=model,
            dataset=dataset,
            device=device,
            args=args,
            output_dir=diversity_dir,
            variant_index=len(eval_summary["variants"]),
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
            "checkpoint": str(checkpoint_path.resolve()),
            "completed_dir": str(completed_dir.resolve()),
            "masked_dir": str(masked_dir.resolve()),
            "diversity_dir": str(diversity_dir.resolve()),
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
        eval_summary["variants"].append(variant_result)
        del model
        torch.cuda.empty_cache()

    del lpips_model
    torch.cuda.empty_cache()

    boundary_names = [
        next(record["sample_id"] for record in sample_records if record["bucket"] == bucket)
        for bucket in ["20-30", "30-40", "40-50", "50-60"]
    ]
    build_boundary_visuals(
        image_dir=subset_image_dir,
        mask_dir=subset_mask_dir,
        boundary_dir=output_dir / "boundary_visuals",
        final_path=output_dir / "final_phase5c_boundary_comparison.png",
        variant_outputs=[
            {"label": variant["label"], "completed_dir": variant["completed_dir"]}
            for variant in eval_summary["variants"]
        ],
        sample_names=boundary_names,
        cell=args.boundary_zoom_cell,
        pad=args.boundary_zoom_pad,
    )
    eval_summary["boundary_visual_sample_ids"] = boundary_names
    eval_summary["boundary_comparison_path"] = str(
        (output_dir / "final_phase5c_boundary_comparison.png").resolve()
    )

    comparison_rows = build_comparison_rows(eval_summary)
    write_comparison_csv(comparison_rows, output_dir / "comparison_table.csv")
    (output_dir / "checkpoint_manifest.json").write_text(
        json.dumps(to_serializable(checkpoint_manifest), indent=2),
        encoding="utf-8",
    )

    phase5a_deltas = compare_with_phase5a(phase5a_summary, eval_summary["variants"])
    acceptance = evaluate_acceptance(eval_summary["variants"])
    phase5c_summary = {
        "training_runs": run_summaries,
        "evaluation": eval_summary,
        "phase5a_delta": phase5a_deltas,
        "acceptance": acceptance,
    }
    (output_dir / "phase5c_summary.json").write_text(
        json.dumps(to_serializable(phase5c_summary), indent=2),
        encoding="utf-8",
    )
    write_metrics_md(
        output_path=output_dir / "METRICS.md",
        summary=eval_summary,
        phase5a_deltas=phase5a_deltas,
        acceptance=acceptance,
    )


if __name__ == "__main__":
    main()
