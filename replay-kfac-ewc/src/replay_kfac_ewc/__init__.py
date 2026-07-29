"""Generative replay and K-FAC EWC for causal language models."""

from .ewc import KFACEWC, LoRADelta
from .factors import (
    CompressedFactor,
    FactorBundle,
    FactorEstimationConfig,
    FactorPair,
    compress_psd,
    dense_accumulator_bytes,
    estimate_factors,
    resolve_linear_modules,
)
from .replay import (
    ReplayConfig,
    ReplayRecord,
    generate_replay,
    load_replay,
    merge_replay,
)
from .validation import CalibrationReport, fit_calibration, mean_forward_kl

__all__ = [
    "CalibrationReport",
    "CompressedFactor",
    "FactorBundle",
    "FactorEstimationConfig",
    "FactorPair",
    "KFACEWC",
    "LoRADelta",
    "ReplayConfig",
    "ReplayRecord",
    "compress_psd",
    "dense_accumulator_bytes",
    "estimate_factors",
    "fit_calibration",
    "generate_replay",
    "load_replay",
    "mean_forward_kl",
    "merge_replay",
    "resolve_linear_modules",
]

__version__ = "0.1.0"
