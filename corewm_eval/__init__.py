"""Read-only CoRe-WM evaluation infrastructure."""

from .config import (
    ALPHA_GRID, CLIP_LENGTH, HORIZONS, LONG_HORIZONS, MANUSCRIPT_START,
    MIN_CLIP_LENGTH, NUM_PREFIXES, PERFORMANCE_THRESHOLDS, ROLLOUT_LENGTH)
from .metrics import linear_cka, multivariate_r2, reconstruction_metrics
from .slicing import split_prefixes_blocks

__all__ = [
    'ALPHA_GRID', 'CLIP_LENGTH', 'HORIZONS', 'LONG_HORIZONS',
    'MANUSCRIPT_START', 'MIN_CLIP_LENGTH', 'NUM_PREFIXES',
    'PERFORMANCE_THRESHOLDS', 'ROLLOUT_LENGTH',
    'linear_cka', 'multivariate_r2', 'reconstruction_metrics',
    'split_prefixes_blocks',
]
