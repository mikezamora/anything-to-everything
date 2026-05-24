"""A1-A8 ablation harness (spec §14.4)."""

from .ablation_runner import (
    ABLATION_CONFIGS,
    AblationConfig,
    run_ablation_matrix,
)

__all__ = ["ABLATION_CONFIGS", "AblationConfig", "run_ablation_matrix"]
