"""embodied_core — a format-aware toolkit for embodied-AI robotics datasets.

The package is deliberately dependency-light: ``numpy`` plus the standard
library are enough for detection and structural validation. Heavier readers
(``pyarrow`` for parquet, ``h5py`` for HDF5, ``mcap`` for MCAP) are optional
and are detected at import time, so the same code runs inside a DSH plugin, a
bare MCP server, or a CI container with a trimmed dependency set.
"""

from .cache import QualityCache, dataset_fingerprint, default_quality_cache_path
from .formats.base import (
    DatasetObject,
    Detection,
    FrameChunk,
    FrameLimitExceeded,
    FramesUnavailable,
    collect_frame_chunks,
)
from .gate import DEFAULT_RULES, evaluate_quality_gate, load_quality_rules
from .jobs import JobCancelled, JobManager, JobNotFound
from .model import DatasetSummary, EpisodeRecord, FeatureSpec, Issue, Severity, summarize_issues, to_jsonable
from .registry import UnsupportedFormat, describe_formats, detect_dataset, open_dataset
from .reports import quality_report
from .validate import CHECKS, DEFAULT_CHECKS, validate

__version__ = "0.3.0"

__all__ = [
    "__version__",
    "CHECKS",
    "DEFAULT_CHECKS",
    "DEFAULT_RULES",
    "DatasetObject",
    "DatasetSummary",
    "Detection",
    "EpisodeRecord",
    "FeatureSpec",
    "FrameChunk",
    "FrameLimitExceeded",
    "FramesUnavailable",
    "Issue",
    "JobCancelled",
    "JobManager",
    "JobNotFound",
    "QualityCache",
    "Severity",
    "UnsupportedFormat",
    "describe_formats",
    "detect_dataset",
    "dataset_fingerprint",
    "default_quality_cache_path",
    "collect_frame_chunks",
    "evaluate_quality_gate",
    "load_quality_rules",
    "open_dataset",
    "quality_report",
    "summarize_issues",
    "to_jsonable",
    "validate",
]
