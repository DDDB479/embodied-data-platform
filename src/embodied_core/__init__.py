"""embodied_core — a format-aware toolkit for embodied-AI robotics datasets.

The package is deliberately dependency-light: ``numpy`` plus the standard
library are enough for detection and structural validation. Heavier readers
(``pyarrow`` for parquet, ``h5py`` for HDF5, ``mcap`` for MCAP) are optional
and are detected at import time, so the same code runs inside a DSH plugin, a
bare MCP server, or a CI container with a trimmed dependency set.
"""

from .formats.base import DatasetObject, Detection, FramesUnavailable
from .model import DatasetSummary, EpisodeRecord, FeatureSpec, Issue, Severity, summarize_issues, to_jsonable
from .registry import UnsupportedFormat, describe_formats, detect_dataset, open_dataset
from .validate import CHECKS, DEFAULT_CHECKS, validate

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "CHECKS",
    "DEFAULT_CHECKS",
    "DatasetObject",
    "DatasetSummary",
    "Detection",
    "EpisodeRecord",
    "FeatureSpec",
    "FramesUnavailable",
    "Issue",
    "Severity",
    "UnsupportedFormat",
    "describe_formats",
    "detect_dataset",
    "open_dataset",
    "summarize_issues",
    "to_jsonable",
    "validate",
]
