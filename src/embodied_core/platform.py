"""Platform capability and dependency diagnostics."""

from __future__ import annotations

import importlib.util
import os
import platform as py_platform
import sys
from typing import Any

from . import __version__
from .media import find_ffprobe
from .registry import describe_formats
from .ros2_codec import max_message_bytes, max_point_count, supported_message_types
from .tfrecord import max_record_bytes


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def capability_report() -> dict[str, Any]:
    """Return JSON-ready capabilities, optional dependencies and limits."""

    optional = {
        name: _has_module(name)
        for name in ("pyarrow", "h5py", "mcap", "boto3", "tensorflow", "PIL")
    }
    return {
        "name": "embodied-core",
        "version": __version__,
        "python": sys.version.split()[0],
        "platform": py_platform.platform(),
        "formats": describe_formats(),
        "ros2_cdr": {
            "available": True,
            "message_types": supported_message_types(),
            "max_message_bytes": max_message_bytes(),
            "max_points": max_point_count(),
        },
        "rlds": {
            "light_decoder": True,
            "max_record_bytes": max_record_bytes(),
            "tensorflow_optional": optional["tensorflow"],
        },
        "media": {
            "ffprobe": find_ffprobe(),
            "video_metadata": find_ffprobe() is not None,
        },
        "storage": {
            "schemes": ["file", "http", "https", "s3"],
            "s3_available": optional["boto3"],
            "max_download_bytes": _env_int("EMBODIED_MAX_DOWNLOAD_BYTES", 10 * 1024 * 1024 * 1024),
        },
        "http_api": {
            "available": True,
            "version": "v1",
            "auth": ["Bearer", "X-API-Key"],
            "etag": True,
            "pagination": True,
        },
        "optional_dependencies": optional,
    }
