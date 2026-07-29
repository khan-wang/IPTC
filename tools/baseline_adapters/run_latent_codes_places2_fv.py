#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from common_places2_fv import load_manifest, select_balanced_subset, subset_from_manifest, write_manifest


REPO_ROOT = Path(__file__).resolve().parents[2]
LATENT_ROOT = REPO_ROOT / "third_party" / "baselines" / "latent-code-inpainting"
DEFAULT_MANIFEST = REPO_ROOT / "exp_data" / "01_main_256_places2_extended36500" / "sample_manifest.csv"
DEFAULT_CONFIG = LATENT_ROOT / "configs" / "places_inpainting.yaml"
EXPECTED_CKPTS = (
    "places256_decoder.ckpt",
    "places256_partialencoder.ckpt",
    "places256_transformer.ckpt",
    "places256_unet.ckpt",
    "places256_vqgan1024_BASE.ckpt",
)
_ORIGINAL_TORCH_LOAD = torch.load


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--subset-manifest", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--seed", type=int, default=20260430)
    parser.add_argument("--per-bucket", type=int, default=0)
    parser.add_argument("--max-len", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--clamp-ratio", type=float, default=0.25)
    parser.add_argument("--sampling-ratio", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--temperature-degradation", type=float, default=0.9)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def select_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    full_rows = load_manifest(Path(args.manifest))
    if args.subset_manifest:
        return subset_from_manifest(full_rows, load_manifest(Path(args.subset_manifest)))
    if args.per_bucket > 0:
        return select_balanced_subset(full_rows, per_bucket=args.per_bucket, seed=args.seed)
    if args.max_len > 0:
        return full_rows[: args.max_len]
    return full_rows


def get_obj_from_str(string: str, reload: bool = False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if "target" not in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def resolve_ckpt_path(name: str) -> Path:
    direct = LATENT_ROOT / "ckpts" / name
    nested = LATENT_ROOT / "ckpts" / "Places365" / name
    if direct.is_file():
        return direct
    if nested.is_file():
        return nested
    raise FileNotFoundError(f"missing required Latent Codes checkpoint: {name}")


def normalize_config_ckpts(config) -> None:
    params = config.model.params
    params.vqmodel_config.params.ckpt_path = str(resolve_ckpt_path("places256_vqgan1024_BASE.ckpt"))
    params.encoder_config.params.ckpt_path = str(resolve_ckpt_path("places256_partialencoder.ckpt"))
    params.decoder_config.params.ckpt_path = str(resolve_ckpt_path("places256_decoder.ckpt"))
    params.unet_config.params.ckpt_path = str(resolve_ckpt_path("places256_unet.ckpt"))
    params.transformer_config.params.ckpt_path = str(resolve_ckpt_path("places256_transformer.ckpt"))


def prepare_imports() -> None:
    os.chdir(LATENT_ROOT)
    root_str = str(LATENT_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def patch_torch_load_for_legacy_ckpts() -> None:
    def patched_torch_load(*args, **kwargs):
        path = args[0] if args else kwargs.get("f")
        if "weights_only" not in kwargs and path is not None:
            try:
                resolved = Path(path).resolve()
            except TypeError:
                resolved = None
            if resolved is not None and (LATENT_ROOT / "ckpts") in resolved.parents:
                kwargs["weights_only"] = False
        return _ORIGINAL_TORCH_LOAD(*args, **kwargs)

    torch.load = patched_torch_load


def force_reference_custom_ops() -> None:
    import core.torch_utils.ops.bias_act as bias_act
    import core.torch_utils.ops.upfirdn2d as upfirdn2d

    bias_act._inited = True
    bias_act._plugin = None
    bias_act._init = lambda: False

    upfirdn2d._inited = True
    upfirdn2d._plugin = None
    upfirdn2d._init = lambda: False


def preprocess_image(path: Path, device: torch.device) -> torch.Tensor:
    image = np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
    tensor = torch.from_numpy(image.transpose(2, 0, 1)).float().to(device) / 127.5 - 1.0
    return tensor.unsqueeze(0)


def preprocess_mask(path: Path, device: torch.device) -> torch.Tensor:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    mask = torch.from_numpy(mask[None, None]).float().to(device)
    return torch.round(mask)


def to_uint8_image(tensor: torch.Tensor) -> Image.Image:
    arr = (tensor.detach().cpu().squeeze(0).permute(1, 2, 0) * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8).numpy()
    return Image.fromarray(arr)


def composite_known_region(pred: Image.Image, gt_path: Path, mask_path: Path) -> Image.Image:
    pred_arr = np.asarray(pred.convert("RGB"), dtype=np.uint8).copy()
    gt_arr = np.asarray(Image.open(gt_path).convert("RGB"), dtype=np.uint8)
    known = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8) >= 128
    pred_arr[known] = gt_arr[known]
    return Image.fromarray(pred_arr)


def load_existing_timing_rows(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row["sample_id"]: row for row in rows if row.get("sample_id")}


def parse_wall_time(value) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def load_existing_rows_by_sample(path: Path, key: str = "sample_id") -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row[key]: row for row in rows if row.get(key)}


def summarize_latent_sbvc_stats(stats_rows: list[dict], head_dim: int) -> dict[str, float | int | str]:
    if not stats_rows:
        return {
            "attention_calls": 0,
            "enabled_calls": 0,
            "original_tokens_mean": 0.0,
            "compact_tokens_mean": 0.0,
            "removed_tokens_sum": 0,
            "removed_tokens_mean": 0.0,
            "safe_tokens_mean": 0.0,
            "protect_tokens_mean": 0.0,
            "score_elements_before": 0,
            "score_elements_after": 0,
            "score_reduction_pct": 0.0,
            "core_gmacs_before": 0.0,
            "core_gmacs_after": 0.0,
            "core_gmacs_reduction_pct": 0.0,
            "actual_compression_ratio": 0.0,
            "route_cache_hits": 0,
            "mode": os.environ.get("LATENT_SBVC_MODE", ""),
            "target_r": int(os.environ.get("LATENT_SBVC_R", "0") or 0),
        }
    before_score = int(sum(int(row["score_elements_before"]) for row in stats_rows))
    after_score = int(sum(int(row["score_elements_after"]) for row in stats_rows))
    original_tokens = int(sum(int(row["original_tokens"]) for row in stats_rows))
    removed_tokens = int(sum(int(row["removed_tokens"]) for row in stats_rows))
    before_gmacs = float(before_score * head_dim / 1.0e9)
    after_gmacs = float(after_score * head_dim / 1.0e9)
    return {
        "attention_calls": len(stats_rows),
        "enabled_calls": int(sum(1 for row in stats_rows if row.get("enabled"))),
        "original_tokens_mean": float(np.mean([int(row["original_tokens"]) for row in stats_rows])),
        "compact_tokens_mean": float(np.mean([int(row["compact_tokens"]) for row in stats_rows])),
        "removed_tokens_sum": removed_tokens,
        "removed_tokens_mean": float(np.mean([int(row["removed_tokens"]) for row in stats_rows])),
        "safe_tokens_mean": float(np.mean([int(row["safe_tokens"]) for row in stats_rows])),
        "protect_tokens_mean": float(np.mean([int(row["protect_tokens"]) for row in stats_rows])),
        "score_elements_before": before_score,
        "score_elements_after": after_score,
        "score_reduction_pct": float((1.0 - after_score / before_score) * 100.0) if before_score > 0 else 0.0,
        "core_gmacs_before": before_gmacs,
        "core_gmacs_after": after_gmacs,
        "core_gmacs_reduction_pct": float((1.0 - after_gmacs / before_gmacs) * 100.0) if before_gmacs > 0 else 0.0,
        "actual_compression_ratio": float(removed_tokens / original_tokens) if original_tokens > 0 else 0.0,
        "route_cache_hits": int(sum(1 for row in stats_rows if row.get("route_cache_hit"))),
        "mode": str(stats_rows[0].get("mode", os.environ.get("LATENT_SBVC_MODE", ""))),
        "target_r": int(stats_rows[0].get("target_r", os.environ.get("LATENT_SBVC_R", "0") or 0)),
    }


def aggregate_sbvc_rows(rows: list[dict[str, str]]) -> dict[str, float | int]:
    numeric_keys = [
        "attention_calls",
        "enabled_calls",
        "removed_tokens_sum",
        "score_elements_before",
        "score_elements_after",
        "core_gmacs_before",
        "core_gmacs_after",
        "route_cache_hits",
    ]
    mean_keys = [
        "original_tokens_mean",
        "compact_tokens_mean",
        "removed_tokens_mean",
        "safe_tokens_mean",
        "protect_tokens_mean",
        "score_reduction_pct",
        "core_gmacs_reduction_pct",
        "actual_compression_ratio",
    ]
    active_rows = [row for row in rows if float(row.get("attention_calls") or 0.0) > 0.0]
    out: dict[str, float | int] = {"num_stat_images": len(rows), "num_active_stat_images": len(active_rows)}
    for key in numeric_keys:
        values = [float(row[key]) for row in rows if row.get(key) not in (None, "")]
        out[f"{key}_sum"] = float(sum(values))
    for key in mean_keys:
        values = [float(row[key]) for row in active_rows if row.get(key) not in (None, "")]
        out[f"{key}_mean"] = float(np.mean(values)) if values else 0.0
    before = float(out.get("core_gmacs_before_sum", 0.0))
    after = float(out.get("core_gmacs_after_sum", 0.0))
    out["core_gmacs_reduction_pct_total"] = float((1.0 - after / before) * 100.0) if before > 0 else 0.0
    original_tokens = float(sum(float(row["attention_calls"]) * float(row["original_tokens_mean"]) for row in rows if row.get("attention_calls")))
    removed_tokens = float(out.get("removed_tokens_sum_sum", 0.0))
    out["actual_compression_ratio_total"] = float(removed_tokens / original_tokens) if original_tokens > 0 else 0.0
    return out


def forward_to_indices(transformer, batch, z_indices, mask, *, sampling_ratio: float, temperature: float, temperature_degradation: float):
    x, c = transformer.get_xc(batch)
    x = x.to(device=transformer.device).float()
    c = c.to(device=transformer.device).float()
    _, c_indices = transformer.encode_to_c(c)
    mask = transformer.preprocess_mask(mask, z_indices)
    r_indices = torch.full_like(z_indices, transformer.mask_token)
    z_start_indices = mask * z_indices + (1 - mask) * r_indices
    index_sample, _, _ = transformer.sample(
        z_start_indices.to(device=transformer.device),
        c_indices.to(device=transformer.device),
        sampling_ratio=sampling_ratio,
        temperature=temperature,
        sample=True,
        temperature_degradation=temperature_degradation,
        top_k=None,
        return_probs=True,
        scheduler="cosine",
    )
    return index_sample


def main() -> None:
    args = parse_args()
    rows = select_rows(args)
    output_dir = Path(args.output_dir).resolve()
    completed_dir = output_dir / "completed_single"
    completed_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(rows, output_dir / "selected_manifest.csv")
    timing_csv_path = output_dir / "per_image_timing.csv"
    existing_timing_rows = load_existing_timing_rows(timing_csv_path)

    prepare_imports()
    patch_torch_load_for_legacy_ckpts()
    force_reference_custom_ops()
    for ckpt_name in EXPECTED_CKPTS:
        resolve_ckpt_path(ckpt_name)

    config = OmegaConf.load(str(Path(args.config).resolve()))
    normalize_config_ckpts(config)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = instantiate_from_config(config.model).to(device)
    model.eval()
    vq_model, encoder, transformer, unet = model.helper_model
    vq_model = vq_model.to(device)
    encoder = encoder.to(device)
    transformer = transformer.to(device)
    if unet is not None:
        unet = unet.to(device)
    latent_mingpt = importlib.import_module("core.modules.transformer.mingpt")
    transformer_core = getattr(transformer, "transformer", None)
    transformer_config = getattr(transformer_core, "config", None)
    head_dim = 88
    if transformer_config is not None:
        head_dim = int(int(transformer_config.n_embd) // int(transformer_config.n_head))
    collect_sbvc_stats = env_flag("LATENT_SBVC_COLLECT_STATS", False) or env_flag("LATENT_SBVC_PROFILE", False)

    fieldnames = [
        "sample_id",
        "wall_time_sec",
        "input_path",
        "mask_path",
        "completed_output_path",
    ]
    timing_rows: list[dict[str, str | float]] = list(existing_timing_rows.values())
    seen_sample_ids = set(existing_timing_rows.keys())
    sbvc_csv_path = output_dir / "per_image_sbvc_stats.csv"
    sbvc_fieldnames = [
        "sample_id",
        "attention_calls",
        "enabled_calls",
        "original_tokens_mean",
        "compact_tokens_mean",
        "removed_tokens_sum",
        "removed_tokens_mean",
        "safe_tokens_mean",
        "protect_tokens_mean",
        "score_elements_before",
        "score_elements_after",
        "score_reduction_pct",
        "core_gmacs_before",
        "core_gmacs_after",
        "core_gmacs_reduction_pct",
        "actual_compression_ratio",
        "route_cache_hits",
        "mode",
        "target_r",
    ]
    existing_sbvc_rows = load_existing_rows_by_sample(sbvc_csv_path)
    sbvc_rows: list[dict[str, str | float | int]] = list(existing_sbvc_rows.values())
    skipped_existing = 0
    processed_new = 0
    timing_mode = "a" if timing_csv_path.is_file() else "w"
    sbvc_mode = "a" if sbvc_csv_path.is_file() else "w"
    with timing_csv_path.open(timing_mode, encoding="utf-8", newline="") as timing_handle, sbvc_csv_path.open(
        sbvc_mode, encoding="utf-8", newline=""
    ) as sbvc_handle:
        writer = csv.DictWriter(timing_handle, fieldnames=fieldnames)
        sbvc_writer = csv.DictWriter(sbvc_handle, fieldnames=sbvc_fieldnames)
        if timing_mode == "w":
            writer.writeheader()
        if sbvc_mode == "w":
            sbvc_writer.writeheader()
        with torch.no_grad():
            for index, row in enumerate(rows):
                sample_id = row["sample_id"]
                image_path = Path(row["resized_image_path"])
                mask_path = Path(row["resized_mask_path"])
                out_path = completed_dir / sample_id

                if args.skip_existing and out_path.is_file():
                    skipped_existing += 1
                    if sample_id not in seen_sample_ids:
                        existing_row = {
                            "sample_id": sample_id,
                            "wall_time_sec": "",
                            "input_path": str(image_path.resolve()),
                            "mask_path": str(mask_path.resolve()),
                            "completed_output_path": str(out_path.resolve()),
                        }
                        timing_rows.append(existing_row)
                        seen_sample_ids.add(sample_id)
                    continue
                x = preprocess_image(image_path, device)
                mask = preprocess_mask(mask_path, device)

                if collect_sbvc_stats:
                    latent_mingpt.reset_latent_sbvc_stats()
                start = time.time()
                quant_z, _, info, mask_out = encoder.encode(x * mask, mask, clamp_ratio=args.clamp_ratio)
                mask_out = mask_out.reshape(x.shape[0], -1)
                z_indices = info[2].reshape(x.shape[0], -1)
                new_batch = {"image": (x * mask).permute(0, 2, 3, 1)}
                z_indices_complete = forward_to_indices(
                    transformer,
                    new_batch,
                    z_indices,
                    mask_out,
                    sampling_ratio=args.sampling_ratio,
                    temperature=args.temperature,
                    temperature_degradation=args.temperature_degradation,
                )
                bsz, channels, height, width = quant_z.shape
                quant_z_complete = vq_model.quantize.get_codebook_entry(
                    z_indices_complete.reshape(-1).int(),
                    shape=(bsz, height, width, channels),
                )
                dec, _, _, _, _ = model.current_model(
                    new_batch,
                    quant=quant_z_complete,
                    mask_in=mask,
                    mask_out=mask_out.reshape(bsz, 1, height, width),
                    return_fstg=False,
                    debug=True,
                )
                rec = x * mask + dec * (1 - mask)
                if unet is not None:
                    rec = unet.refine(rec, mask)
                elapsed = time.time() - start

                pred = to_uint8_image(rec)
                pred = composite_known_region(pred, image_path, mask_path)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                pred.save(out_path)

                timing_row = {
                    "sample_id": sample_id,
                    "wall_time_sec": elapsed,
                    "input_path": str(image_path.resolve()),
                    "mask_path": str(mask_path.resolve()),
                    "completed_output_path": str(out_path.resolve()),
                }
                writer.writerow(timing_row)
                timing_handle.flush()
                timing_rows.append(timing_row)
                seen_sample_ids.add(sample_id)
                if collect_sbvc_stats:
                    sbvc_row = {"sample_id": sample_id, **summarize_latent_sbvc_stats(latent_mingpt.get_latent_sbvc_stats(), head_dim)}
                    sbvc_writer.writerow(sbvc_row)
                    sbvc_handle.flush()
                    sbvc_rows.append(sbvc_row)
                processed_new += 1
                if processed_new == 1 or processed_new % 100 == 0:
                    print(
                        json.dumps(
                            {
                                "progress": "latent_codes_places2_fv",
                                "processed_new": processed_new,
                                "skipped_existing": skipped_existing,
                                "total": len(rows),
                                "last_sample_id": sample_id,
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )

    wall_times = [wall_time for wall_time in (parse_wall_time(row.get("wall_time_sec")) for row in timing_rows) if wall_time is not None]

    summary = {
        "status": "ok",
        "num_images": len(rows),
        "completed_dir": str(completed_dir.resolve()),
        "processed_new": processed_new,
        "skipped_existing": skipped_existing,
        "mean_wall_time_sec": float(np.mean(wall_times)) if wall_times else 0.0,
        "latent_sbvc_collect_stats": bool(collect_sbvc_stats),
        "latent_sbvc_head_dim": int(head_dim),
        "latent_sbvc_summary": aggregate_sbvc_rows([dict(row) for row in sbvc_rows]) if collect_sbvc_stats else {},
    }
    (output_dir / "runtime_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
