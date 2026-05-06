#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
ADAPTER_DIR = REPO_ROOT / "tools" / "baseline_adapters"
DEFAULT_PROBE_MANIFEST = REPO_ROOT / "outputs" / "places2_fv_reproduction" / "smoke20" / "LatentCodes_probe" / "selected_manifest.csv"
DEFAULT_FALLBACK_MANIFEST = REPO_ROOT / "exp_data" / "01_main_256_places2_extended36500" / "sample_manifest.csv"

if str(ADAPTER_DIR) not in sys.path:
    sys.path.insert(0, str(ADAPTER_DIR))

import run_latent_codes_places2_fv as latent_adapter  # noqa: E402


class SectionTimer:
    def __init__(self, device: torch.device):
        self.device = device
        self.is_cuda = device.type == "cuda"
        self.records: list[tuple[str, Any, Any]] = []

    def timeit(self, name: str):
        return _SectionTimerContext(self, name)

    def _start(self, name: str) -> None:
        if self.is_cuda:
            start = torch.cuda.Event(enable_timing=True)
            start.record()
            self.records.append((name, start, None))
        else:
            self.records.append((name, time.perf_counter(), None))

    def _stop(self) -> None:
        name, start, _ = self.records[-1]
        if self.is_cuda:
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            self.records[-1] = (name, start, end)
        else:
            self.records[-1] = (name, start, time.perf_counter())

    def summary_ms(self) -> dict[str, float]:
        if self.is_cuda:
            torch.cuda.synchronize(self.device)
        totals: dict[str, float] = defaultdict(float)
        for name, start, end in self.records:
            if self.is_cuda:
                totals[name] += float(start.elapsed_time(end))
            else:
                totals[name] += float((end - start) * 1000.0)
        return dict(totals)


class _SectionTimerContext:
    def __init__(self, timer: SectionTimer, name: str):
        self.timer = timer
        self.name = name

    def __enter__(self):
        self.timer._start(self.name)
        return self

    def __exit__(self, exc_type, exc, tb):
        self.timer._stop()
        return False


class AttentionProbe:
    def __init__(self, device: torch.device):
        self.device = device
        self.is_cuda = device.type == "cuda"
        self.handles: list[Any] = []
        self.calls: list[dict[str, Any]] = []
        self._pending: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def register(self, root: torch.nn.Module, prefix: str = "") -> None:
        for name, module in root.named_modules():
            class_name = module.__class__.__name__
            if class_name not in {"CausalSelfAttention", "AttnBlock"}:
                continue
            full_name = f"{prefix}.{name}" if prefix and name else (prefix or name)
            category = "gpt_causal_self_attention" if class_name == "CausalSelfAttention" else "spatial_attn_block"
            self.handles.append(module.register_forward_pre_hook(self._make_pre_hook(full_name, class_name, category)))
            self.handles.append(module.register_forward_hook(self._make_post_hook(full_name)))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _make_pre_hook(self, name: str, class_name: str, category: str):
        def hook(module, inputs):
            x = inputs[0] if inputs else None
            shape = list(x.shape) if torch.is_tensor(x) else []
            score_elements = self._estimate_score_elements(module, class_name, x)
            entry = {
                "module": name,
                "class": class_name,
                "category": category,
                "input_shape": shape,
                "output_shape": [],
                "score_elements": score_elements,
            }
            if self.is_cuda:
                start = torch.cuda.Event(enable_timing=True)
                start.record()
                entry["start_event"] = start
            else:
                entry["start_time"] = time.perf_counter()
            self._pending[name].append(entry)

        return hook

    def _make_post_hook(self, name: str):
        def hook(module, inputs, output):
            entry = self._pending[name].pop()
            y = output[0] if isinstance(output, tuple) and output else output
            if torch.is_tensor(y):
                entry["output_shape"] = list(y.shape)
            if self.is_cuda:
                end = torch.cuda.Event(enable_timing=True)
                end.record()
                entry["end_event"] = end
            else:
                entry["elapsed_ms"] = (time.perf_counter() - entry["start_time"]) * 1000.0
            self.calls.append(entry)

        return hook

    @staticmethod
    def _estimate_score_elements(module, class_name: str, x: Any) -> int:
        if not torch.is_tensor(x):
            return 0
        if class_name == "CausalSelfAttention" and x.ndim == 3:
            batch, tokens, _channels = x.shape
            heads = int(getattr(module, "n_head", 1))
            return int(batch * heads * tokens * tokens)
        if class_name == "AttnBlock" and x.ndim == 4:
            batch, _channels, height, width = x.shape
            tokens = int(height * width)
            return int(batch * tokens * tokens)
        return 0

    def finalize(self) -> None:
        if self.is_cuda:
            torch.cuda.synchronize(self.device)
            for entry in self.calls:
                entry["elapsed_ms"] = float(entry["start_event"].elapsed_time(entry["end_event"]))
                del entry["start_event"]
                del entry["end_event"]
        else:
            for entry in self.calls:
                entry.pop("start_time", None)

    def module_summary(self) -> list[dict[str, Any]]:
        grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for call in self.calls:
            grouped[(call["category"], call["class"], call["module"])].append(call)

        rows: list[dict[str, Any]] = []
        for (category, class_name, module), calls in sorted(grouped.items()):
            elapsed = [float(call["elapsed_ms"]) for call in calls]
            input_shapes = sorted({str(call["input_shape"]) for call in calls})
            output_shapes = sorted({str(call["output_shape"]) for call in calls})
            score_elements = int(sum(int(call["score_elements"]) for call in calls))
            token_lengths = [self._token_length(call["input_shape"]) for call in calls]
            rows.append(
                {
                    "category": category,
                    "class": class_name,
                    "module": module,
                    "calls": len(calls),
                    "total_ms": float(sum(elapsed)),
                    "mean_ms": float(np.mean(elapsed)) if elapsed else 0.0,
                    "min_ms": float(np.min(elapsed)) if elapsed else 0.0,
                    "max_ms": float(np.max(elapsed)) if elapsed else 0.0,
                    "score_elements": score_elements,
                    "token_length_min": int(min(token_lengths)) if token_lengths else 0,
                    "token_length_max": int(max(token_lengths)) if token_lengths else 0,
                    "input_shapes": "; ".join(input_shapes),
                    "output_shapes": "; ".join(output_shapes),
                }
            )
        return rows

    @staticmethod
    def _token_length(shape: list[int]) -> int:
        if len(shape) == 3:
            return int(shape[1])
        if len(shape) == 4:
            return int(shape[2] * shape[3])
        return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only probe for Latent Codes attention topology and timing.")
    parser.add_argument("--manifest", default=str(DEFAULT_PROBE_MANIFEST if DEFAULT_PROBE_MANIFEST.is_file() else DEFAULT_FALLBACK_MANIFEST))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=str(latent_adapter.DEFAULT_CONFIG))
    parser.add_argument("--seed", type=int, default=20260501)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--clamp-ratio", type=float, default=0.25)
    parser.add_argument("--sampling-ratio", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--temperature-degradation", type=float, default=0.9)
    parser.add_argument("--sample", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def load_single_row(manifest: Path, row_index: int) -> dict[str, str]:
    rows = latent_adapter.load_manifest(manifest)
    if not rows:
        raise ValueError(f"empty manifest: {manifest}")
    if row_index < 0 or row_index >= len(rows):
        raise IndexError(f"row-index {row_index} out of range for {len(rows)} rows")
    return rows[row_index]


def load_latent_codes_model(args: argparse.Namespace, device: torch.device):
    latent_adapter.prepare_imports()
    latent_adapter.patch_torch_load_for_legacy_ckpts()
    latent_adapter.force_reference_custom_ops()
    for ckpt_name in latent_adapter.EXPECTED_CKPTS:
        latent_adapter.resolve_ckpt_path(ckpt_name)
    config = OmegaConf.load(str(Path(args.config).resolve()))
    latent_adapter.normalize_config_ckpts(config)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = latent_adapter.instantiate_from_config(config.model).to(device)
    model.eval()
    vq_model, encoder, transformer, unet = model.helper_model
    # The Latent Codes repo overwrites helper.train with a plain function, so
    # calling helper.eval() can return False instead of the module itself.
    vq_model = vq_model.to(device)
    encoder = encoder.to(device)
    transformer = transformer.to(device)
    if unet is not None:
        unet = unet.to(device)
    return model, vq_model, encoder, transformer, unet, config


@torch.no_grad()
def run_pipeline(args: argparse.Namespace, row: dict[str, str], model_parts, device: torch.device, timer: SectionTimer | None = None):
    model, vq_model, encoder, transformer, unet, _config = model_parts
    image_path = Path(row["resized_image_path"])
    mask_path = Path(row["resized_mask_path"])
    x = latent_adapter.preprocess_image(image_path, device)
    mask = latent_adapter.preprocess_mask(mask_path, device)

    local_timer = timer if timer is not None else SectionTimer(device)
    with local_timer.timeit("encoder_ms"):
        quant_z, _, info, mask_out = encoder.encode(x * mask, mask, clamp_ratio=args.clamp_ratio)
        mask_out_flat = mask_out.reshape(x.shape[0], -1)
        z_indices = info[2].reshape(x.shape[0], -1)

    new_batch = {"image": (x * mask).permute(0, 2, 3, 1)}

    with local_timer.timeit("condition_encode_ms"):
        _x_in, c = transformer.get_xc(new_batch)
        c = c.to(device=transformer.device).float()
        _, c_indices = transformer.encode_to_c(c)

    with local_timer.timeit("sample_loop_ms"):
        mask_latent = transformer.preprocess_mask(mask_out_flat, z_indices)
        r_indices = torch.full_like(z_indices, transformer.mask_token)
        z_start_indices = mask_latent * z_indices + (1 - mask_latent) * r_indices
        z_indices_complete, prob_results, gathered_idx_info = transformer.sample(
            z_start_indices.to(device=transformer.device),
            c_indices.to(device=transformer.device),
            sampling_ratio=args.sampling_ratio,
            temperature=args.temperature,
            sample=args.sample,
            temperature_degradation=args.temperature_degradation,
            top_k=None,
            return_probs=True,
            scheduler="cosine",
        )

    with local_timer.timeit("codebook_decode_ms"):
        bsz, channels, height, width = quant_z.shape
        quant_z_complete = vq_model.quantize.get_codebook_entry(
            z_indices_complete.reshape(-1).int(),
            shape=(bsz, height, width, channels),
        )

    with local_timer.timeit("partial_decoder_ms"):
        dec, _, _, _, _ = model.current_model(
            new_batch,
            quant=quant_z_complete,
            mask_in=mask,
            mask_out=mask_out_flat.reshape(bsz, 1, height, width),
            return_fstg=False,
            debug=True,
        )
        rec_pre_unet = x * mask + dec * (1 - mask)

    with local_timer.timeit("unet_refine_ms"):
        rec = unet.refine(rec_pre_unet, mask) if unet is not None else rec_pre_unet

    token_info = {
        "z_token_length": int(z_indices.shape[1]),
        "condition_token_length": int(c_indices.shape[1]),
        "transformer_token_length": int(z_indices.shape[1] + c_indices.shape[1]),
        "latent_grid_hw": [int(height), int(width)],
        "latent_channels": int(channels),
        "known_latent_tokens": int(mask_latent.sum().item()),
        "unknown_latent_tokens": int((1 - mask_latent).sum().item()),
        "sample_iterations": int(len(gathered_idx_info)),
        "selected_tokens_per_iteration": [int(np.asarray(item).shape[1]) for item in gathered_idx_info],
    }
    outputs = {
        "x": x,
        "mask": mask,
        "rec_pre_unet": rec_pre_unet,
        "rec": rec,
    }
    return token_info, outputs


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, summary: dict[str, Any]) -> None:
    token = summary["token_info"]
    timing = summary["timing_ms"]
    attn = summary["attention_summary"]
    qkv = summary["qkv_compression_assessment"]
    lines = [
        "# Latent Codes Read-Only Probe",
        "",
        "## Topology",
        f"- Sample: `{summary['sample_id']}`",
        f"- Latent z tokens: `{token['z_token_length']}` on grid `{token['latent_grid_hw'][0]}x{token['latent_grid_hw'][1]}`",
        f"- Condition tokens: `{token['condition_token_length']}`",
        f"- Transformer sequence length: `{token['transformer_token_length']}`",
        f"- GPT layers / heads / embedding: `{summary['model_config']['n_layer']} / {summary['model_config']['n_head']} / {summary['model_config']['n_embd']}`",
        f"- Known / unknown latent tokens in this mask: `{token['known_latent_tokens']} / {token['unknown_latent_tokens']}`",
        f"- Sampling iterations: `{token['sample_iterations']}` with selected tokens `{token['selected_tokens_per_iteration']}`",
        "",
        "## Timing",
        f"- Model wall time, measured sections sum: `{timing['model_sections_total_ms']:.3f} ms`",
        f"- Sample loop: `{timing['sample_loop_ms']:.3f} ms`",
        f"- GPT attention total: `{attn['gpt_attention_total_ms']:.3f} ms`",
        f"- Spatial AttnBlock total: `{attn['spatial_attn_total_ms']:.3f} ms`",
        f"- GPT attention share of sample loop: `{attn['gpt_attention_share_of_sample_loop']:.2%}`",
        f"- GPT attention share of model sections: `{attn['gpt_attention_share_of_model_sections']:.2%}`",
        f"- All hooked attention share of model sections: `{attn['all_attention_share_of_model_sections']:.2%}`",
        f"- GPT attention score elements: `{attn['gpt_score_elements_total']}`",
        "",
        "## Q/K/V Compression Assessment",
        f"- Verdict: `{qkv['verdict']}`",
        f"- Reason: {qkv['reason']}",
        f"- Recommended first SBVC target: `{qkv['recommended_first_target']}`",
        "",
        "## Evidence Files",
        f"- JSON: `{summary['paths']['summary_json']}`",
        f"- Module CSV: `{summary['paths']['module_csv']}`",
        f"- Call CSV: `{summary['paths']['call_csv']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_qkv_assessment() -> dict[str, str]:
    return {
        "verdict": "conditionally_possible_attention_local_only",
        "reason": (
            "CausalSelfAttention projects Q/K/V from the same [B,T,C] latent-token sequence, "
            "so a shared token assignment can be applied inside one attention block. It must be "
            "restored to T=257 before the residual/MLP/head because the sampler needs per-position "
            "logits/confidence for all 256 latent codes. Physical cross-block token deletion is not safe."
        ),
        "recommended_first_target": (
            "do not implement yet; if attention share is high enough, prototype either K/V-only or "
            "Q/K/V same-route merge inside CausalSelfAttention with immediate restore, protecting the SOS token "
            "and all currently unknown/masked latent tokens"
        ),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = Path(args.manifest).resolve()
    row = load_single_row(manifest, args.row_index)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model_parts = load_latent_codes_model(args, device)
    model, vq_model, encoder, transformer, unet, _config = model_parts

    with torch.no_grad():
        for _ in range(max(args.warmup, 0)):
            run_pipeline(args, row, model_parts, device, timer=None)

    probe = AttentionProbe(device)
    probe.register(vq_model, "vq_model")
    probe.register(encoder, "encoder")
    probe.register(transformer, "transformer")
    probe.register(model.current_model, "decoder")
    if unet is not None:
        probe.register(unet, "unet")
    timer = SectionTimer(device)
    with torch.no_grad():
        token_info, outputs = run_pipeline(args, row, model_parts, device, timer=timer)
    probe.finalize()
    probe.close()
    timing_ms = timer.summary_ms()

    pred_path = output_dir / "probe_output.png"
    raw_path = output_dir / "probe_pre_unet_output.png"
    latent_adapter.to_uint8_image(outputs["rec"]).save(pred_path)
    latent_adapter.to_uint8_image(outputs["rec_pre_unet"]).save(raw_path)

    module_rows = probe.module_summary()
    call_rows = []
    for idx, call in enumerate(probe.calls):
        clean = dict(call)
        clean["call_index"] = idx
        clean["input_shape"] = str(clean["input_shape"])
        clean["output_shape"] = str(clean["output_shape"])
        call_rows.append(clean)

    gpt_rows = [row_ for row_ in module_rows if row_["category"] == "gpt_causal_self_attention"]
    spatial_rows = [row_ for row_ in module_rows if row_["category"] == "spatial_attn_block"]
    gpt_ms = float(sum(row_["total_ms"] for row_ in gpt_rows))
    spatial_ms = float(sum(row_["total_ms"] for row_ in spatial_rows))
    model_sections_total_ms = float(sum(timing_ms.values()))
    sample_loop_ms = float(timing_ms.get("sample_loop_ms", 0.0))
    attention_summary = {
        "gpt_attention_layers": len(gpt_rows),
        "gpt_attention_calls": int(sum(row_["calls"] for row_ in gpt_rows)),
        "gpt_attention_total_ms": gpt_ms,
        "gpt_attention_mean_ms_per_call": gpt_ms / max(1, int(sum(row_["calls"] for row_ in gpt_rows))),
        "gpt_score_elements_total": int(sum(row_["score_elements"] for row_ in gpt_rows)),
        "spatial_attn_modules": len(spatial_rows),
        "spatial_attn_calls": int(sum(row_["calls"] for row_ in spatial_rows)),
        "spatial_attn_total_ms": spatial_ms,
        "gpt_attention_share_of_sample_loop": gpt_ms / sample_loop_ms if sample_loop_ms > 0 else math.nan,
        "gpt_attention_share_of_model_sections": gpt_ms / model_sections_total_ms if model_sections_total_ms > 0 else math.nan,
        "all_attention_share_of_model_sections": (gpt_ms + spatial_ms) / model_sections_total_ms if model_sections_total_ms > 0 else math.nan,
    }
    timing_ms["model_sections_total_ms"] = model_sections_total_ms

    paths = {
        "summary_json": str(output_dir / "summary.json"),
        "module_csv": str(output_dir / "attention_modules.csv"),
        "call_csv": str(output_dir / "attention_calls.csv"),
        "report_md": str(output_dir / "REPORT.md"),
        "probe_output": str(pred_path),
        "probe_pre_unet_output": str(raw_path),
    }
    summary = {
        "status": "ok",
        "sample_id": row["sample_id"],
        "manifest": str(manifest),
        "image_path": str(Path(row["resized_image_path"]).resolve()),
        "mask_path": str(Path(row["resized_mask_path"]).resolve()),
        "device": str(device),
        "seed": args.seed,
        "sampling": {
            "sampling_ratio": args.sampling_ratio,
            "temperature": args.temperature,
            "temperature_degradation": args.temperature_degradation,
            "sample": args.sample,
        },
        "model_config": {
            "block_size": int(transformer.transformer.config.block_size),
            "vocab_size": int(transformer.transformer.config.vocab_size),
            "n_layer": int(transformer.transformer.config.n_layer),
            "n_head": int(transformer.transformer.config.n_head),
            "n_embd": int(transformer.transformer.config.n_embd),
        },
        "token_info": token_info,
        "timing_ms": timing_ms,
        "attention_summary": attention_summary,
        "qkv_compression_assessment": build_qkv_assessment(),
        "paths": paths,
    }

    write_csv(output_dir / "attention_modules.csv", module_rows)
    write_csv(output_dir / "attention_calls.csv", call_rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_report(output_dir / "REPORT.md", summary)

    print(json.dumps(summary["attention_summary"], indent=2))
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()
