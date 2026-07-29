#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from common_places2_fv import load_manifest, select_balanced_subset, write_manifest, write_sample_id_list


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--per-bucket", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260430)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    output_manifest = Path(args.output_manifest)

    rows = load_manifest(manifest_path)
    selected = select_balanced_subset(rows, per_bucket=args.per_bucket, seed=args.seed)
    write_manifest(selected, output_manifest)
    write_sample_id_list(selected, output_manifest.with_suffix(".sample_ids.txt"))

    print(
        {
            "status": "ok",
            "manifest": str(output_manifest),
            "num_rows": len(selected),
            "per_bucket": args.per_bucket,
            "seed": args.seed,
        }
    )


if __name__ == "__main__":
    main()
