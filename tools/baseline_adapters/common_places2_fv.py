from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Iterable

import numpy as np


BUCKET_ORDER = ("01-10", "10-20", "20-30", "30-40", "40-50", "50-60")
BUCKET_RANK = {bucket: idx for idx, bucket in enumerate(BUCKET_ORDER)}


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_manifest(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot write empty manifest")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def index_by_sample_id(rows: Iterable[dict[str, str]]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        sample_id = row["sample_id"]
        if sample_id in out:
            raise ValueError(f"duplicate sample_id in manifest: {sample_id}")
        out[sample_id] = row
    return out


def subset_from_manifest(
    full_rows: list[dict[str, str]],
    subset_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    by_id = index_by_sample_id(full_rows)
    selected = []
    for row in subset_rows:
        sample_id = row["sample_id"]
        if sample_id not in by_id:
            raise KeyError(f"subset sample_id not found in full manifest: {sample_id}")
        selected.append(by_id[sample_id])
    return selected


def select_balanced_subset(
    rows: list[dict[str, str]],
    *,
    per_bucket: int,
    seed: int,
) -> list[dict[str, str]]:
    rng = np.random.default_rng(seed)
    selected: list[dict[str, str]] = []
    buckets_present = sorted({row["bucket"] for row in rows}, key=lambda bucket: BUCKET_RANK.get(bucket, 999))
    for bucket in buckets_present:
        bucket_rows = [row for row in rows if row["bucket"] == bucket]
        if len(bucket_rows) < per_bucket:
            raise ValueError(f"bucket {bucket} has only {len(bucket_rows)} rows, need {per_bucket}")
        order = rng.permutation(len(bucket_rows))[:per_bucket]
        picked = [bucket_rows[int(idx)] for idx in order]
        picked.sort(key=lambda row: row["sample_id"])
        selected.extend(picked)
    return selected


def symlink_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src, dst)


def materialize_manifest_links(
    rows: list[dict[str, str]],
    *,
    image_dir: Path,
    mask_dir: Path,
    image_key: str = "resized_image_path",
    mask_key: str = "resized_mask_path",
) -> None:
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        sample_id = row["sample_id"]
        symlink_file(Path(row[image_key]), image_dir / sample_id)
        symlink_file(Path(row[mask_key]), mask_dir / sample_id)


def write_sample_id_list(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(f"{row['sample_id']}\n")
