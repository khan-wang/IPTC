#!/usr/bin/env python3
"""Public entry point for image-quality evaluation."""

import sys
from pathlib import Path


ADAPTER_ROOT = Path(__file__).resolve().parent / "baseline_adapters"
sys.path.insert(0, str(ADAPTER_ROOT))

from eval_places2_fv_outputs import main  # noqa: E402


if __name__ == "__main__":
    main()
