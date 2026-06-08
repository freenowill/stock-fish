#!/usr/bin/env python3
"""Walk-forward stage2 runner for AlphaExtra factors.

Thin wrapper around run_stage2_walk_forward.py that overrides MODEL_SPECS
to use AlphaExtra templates (cn_extra_data provider, AlphaExtra handler).

All walk-forward logic (fold generation, training, backtest) is inherited
from the original script — only the model specs differ.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Import the original module
sys.path.insert(0, str(SCRIPTS_DIR := ROOT / "scripts" / "practice"))
import run_stage2_walk_forward as _orig

# Override MODEL_SPECS to use AlphaExtra templates
_orig.MODEL_SPECS = [
    {
        "name": "lightgbm",
        "template": ROOT / "examples" / "benchmarks" / "LightGBM" / "workflow_config_lightgbm_AlphaExtra.yaml",
        "model_mode": "robust",
    },
]

if __name__ == "__main__":
    _orig.main()
