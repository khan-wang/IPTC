#!/usr/bin/env python3
"""Public entry point for Latent Codes SBVC inference."""

import sys
from pathlib import Path


ADAPTER_ROOT = Path(__file__).resolve().parent / "baseline_adapters"
sys.path.insert(0, str(ADAPTER_ROOT))

from run_latent_codes_places2_fv import main  # noqa: E402


if __name__ == "__main__":
    main()
