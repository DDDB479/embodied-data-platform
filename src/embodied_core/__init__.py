"""embodied_core — a format-aware toolkit for embodied-AI robotics datasets.

The package is deliberately dependency-light: ``numpy`` plus the standard
library are enough for detection and structural validation. Heavier readers
(``pyarrow`` for parquet, ``h5py`` for HDF5, ``mcap`` for MCAP) are optional
and are detected at import time, so the same code runs inside a DSH plugin, a
bare MCP server, or a CI container with a trimmed dependency set.
"""

__version__ = "0.5.0"

from .annotations import apply_annotation_feedback, review_annotations
from .cache import QualityCache, dataset_fingerprint, default_quality_cache_path
from .convert import compute_normalization_stats, convert, to_lerobot_v3
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
from .media import probe_dataset_media, probe_video
from .platform import capability_report
from .model import DatasetSummary, EpisodeRecord, FeatureSpec, Issue, Severity, summarize_issues, to_jsonable
from .registry import UnsupportedFormat, describe_formats, detect_dataset, open_dataset
from .reports import quality_report
from .ros2_codec import decode_message, supported_message_types
from .storage import parse_storage_uri, pull
from .tfrecord import decode_rlds_record, iter_tfrecords
from .validate import CHECKS, DEFAULT_CHECKS, validate

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
    "apply_annotation_feedback",
    "compute_normalization_stats",
    "capability_report",
    "convert",
    "describe_formats",
    "detect_dataset",
    "dataset_fingerprint",
    "default_quality_cache_path",
    "decode_message",
    "decode_rlds_record",
    "collect_frame_chunks",
    "evaluate_quality_gate",
    "load_quality_rules",
    "open_dataset",
    "parse_storage_uri",
    "probe_dataset_media",
    "probe_video",
    "pull",
    "quality_report",
    "review_annotations",
    "supported_message_types",
    "iter_tfrecords",
    "to_lerobot_v3",
    "summarize_issues",
    "to_jsonable",
    "validate",
]
