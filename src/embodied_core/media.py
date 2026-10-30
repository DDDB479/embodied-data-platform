"""Optional media probing and modality inventory.

ffprobe is never required. Video metadata is probed only when the binary is
available; depth/point-cloud/language inventory comes from the adapter summary
and therefore works in minimal installs.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable

from .model import DatasetSummary

VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".mpg", ".mpeg"}


def find_ffprobe() -> str | None:
    configured = os.environ.get("FFPROBE")
    if configured and Path(configured).is_file():
        return str(Path(configured))
    return shutil.which("ffprobe")


def probe_video(
    path: str | Path,
    *,
    ffprobe: str | None = None,
    timeout_s: float = 15.0,
    runner: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Return normalized video metadata, or a structured unavailable result."""

    target = Path(path).expanduser()
    binary = ffprobe or find_ffprobe()
    body: dict[str, Any] = {"path": str(target), "available": binary is not None}
    if binary is None:
        body.update(
            {
                "reason": "ffprobe not found",
                "install": "install FFmpeg and ensure ffprobe is on PATH, or set FFPROBE",
            }
        )
        return body
    if not target.is_file():
        body.update({"available": False, "reason": "file not found"})
        return body
    command = [
        binary,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(target),
    ]
    run = runner or subprocess.run
    try:
        completed = run(command, capture_output=True, text=True, timeout=timeout_s, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        body.update({"available": False, "reason": f"{type(exc).__name__}: {exc}"})
        return body
    if int(getattr(completed, "returncode", 1)) != 0:
        detail = (getattr(completed, "stderr", "") or "")[-500:]
        body.update({"available": False, "reason": f"ffprobe failed: {detail.strip()}"})
        return body
    try:
        payload = json.loads(getattr(completed, "stdout", "") or "{}")
    except json.JSONDecodeError as exc:
        body.update({"available": False, "reason": f"invalid ffprobe JSON: {exc}"})
        return body
    streams = payload.get("streams") or []
    video = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    audio = next((stream for stream in streams if stream.get("codec_type") == "audio"), None)
    fmt = payload.get("format") or {}
    body.update(
        {
            "available": True,
            "has_video": video is not None,
            "has_audio": audio is not None,
            "duration_s": _float(fmt.get("duration")),
            "size_bytes": _int(fmt.get("size")),
            "format_name": fmt.get("format_name"),
            "bit_rate": _int(fmt.get("bit_rate")),
            "codec_name": video.get("codec_name") if video else None,
            "width": _int(video.get("width")) if video else None,
            "height": _int(video.get("height")) if video else None,
            "pix_fmt": video.get("pix_fmt") if video else None,
            "avg_frame_rate": video.get("avg_frame_rate") if video else None,
            "r_frame_rate": video.get("r_frame_rate") if video else None,
            "nb_frames": _int(video.get("nb_frames")) if video else None,
            "streams": len(streams),
        }
    )
    return body


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def modality_inventory(summary: DatasetSummary) -> dict[str, Any]:
    """Summarize image/video/depth/point-cloud/language modalities."""

    counts: dict[str, int] = {}
    features: dict[str, list[dict[str, Any]]] = {
        "image": [],
        "video": [],
        "depth": [],
        "point_cloud": [],
        "language": [],
    }
    for feature in summary.features:
        kind = feature.kind
        counts[kind] = counts.get(kind, 0) + 1
        if kind in features:
            features[kind].append(feature.to_dict())
    return {
        "counts": counts,
        "image_features": features["image"],
        "video_features": features["video"],
        "depth_features": features["depth"],
        "point_cloud_features": features["point_cloud"],
        "language_features": features["language"],
        "tasks": list(summary.tasks),
    }


def probe_dataset_media(
    dataset: Any,
    *,
    max_videos: int = 4,
    probe: bool = True,
) -> dict[str, Any]:
    """Inventory modalities and optionally probe a bounded set of videos."""

    summary: DatasetSummary = dataset.summary
    inventory = modality_inventory(summary)
    root = Path(dataset.root)
    video_files = sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    )[: max(0, int(max_videos))]
    binary = find_ffprobe()
    probes = [probe_video(path) for path in video_files] if probe and binary else []
    return {
        **inventory,
        "video_probe": {
            "available": binary is not None,
            "binary": binary,
            "install": None if binary else "install FFmpeg and ensure ffprobe is on PATH, or set FFPROBE",
            "files_found": len(video_files),
            "probes": probes,
        },
    }
