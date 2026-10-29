"""Reusable report builders shared by the CLI, MCP server and job runner.

Keeping this logic outside ``mcp_server`` means synchronous calls, background
jobs and the command line all produce byte-identical payloads and all use the
same incremental cache.
"""

from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable, Sequence

from .cache import QualityCache, dataset_fingerprint, quality_cache_key
from .formats.base import resolve_chunk_frames, resolve_max_frames
from .quality import (
    compute_dataset_stats,
    detect_anomalies,
    merge_feature_stats,
    segment_by_motion,
    shift_anomaly_windows,
    shift_segments,
)
from .registry import open_dataset

ProgressCallback = Callable[[float, str], None]


def _report(progress: ProgressCallback | None, value: float, message: str) -> None:
    if progress is not None:
        progress(value, message)


def _cache_size() -> int:
    try:
        return max(1, int(os.environ.get("EMBODIED_QUALITY_CACHE_MAX_ENTRIES", "1000")))
    except ValueError:
        return 1000


def quality_report(
    path: str | Path,
    *,
    episodes: Sequence[int] | None = None,
    features: Sequence[str] | None = None,
    z: float = 8.0,
    include_segments: bool = False,
    skip_anomalies: bool = False,
    use_cache: bool = True,
    cache_path: str | Path | None = None,
    refresh_cache: bool = False,
    chunk_frames: int | None = None,
    max_frames: int | None = None,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Compute a quality report, reusing a fingerprint-keyed cached result."""

    resolved_chunk = resolve_chunk_frames(chunk_frames)
    resolved_max = resolve_max_frames(max_frames)
    params = {
        "episodes": list(episodes) if episodes else None,
        "features": list(features) if features else None,
        "z": float(z),
        "include_segments": bool(include_segments),
        "skip_anomalies": bool(skip_anomalies),
        "chunk_frames": resolved_chunk,
        "max_frames": resolved_max,
    }
    started = time.perf_counter()
    cache_info: dict[str, Any] = {"enabled": bool(use_cache), "hit": False}
    cache: QualityCache | None = None
    key: str | None = None
    fingerprint: dict[str, Any] | None = None

    if use_cache:
        _report(progress, 2.0, "fingerprinting dataset")
        try:
            cache = QualityCache(cache_path, max_entries=_cache_size())
            fingerprint = dataset_fingerprint(path, exclude=[cache.path])
            key = quality_cache_key(path, str(fingerprint["sha256"]), params)
            cache_info.update(
                {
                    "path": str(cache.path),
                    "key": key,
                    "fingerprint": fingerprint["sha256"],
                    "files": fingerprint["files"],
                    "bytes": fingerprint["bytes"],
                }
            )
            cached = None if refresh_cache else cache.get(key)
        except (OSError, sqlite3.Error, ValueError) as exc:
            # Caching is an optimization. A read-only home directory or a
            # transient lock must not make the actual report fail.
            cached = None
            cache_info["error"] = f"{type(exc).__name__}: {exc}"
        if cached is not None:
            payload = cached["payload"]
            cache_info.update(
                {
                    "hit": True,
                    "created_at": cached["created_at"],
                    "age_s": round(time.time() - float(cached["created_at"]), 3),
                    "hits": cached["hits"],
                    "compute_ms": cached["compute_ms"],
                }
            )
            payload["cache"] = cache_info
            _report(progress, 100.0, "cache hit")
            return payload

    _report(progress, 12.0, "opening dataset")
    dataset = open_dataset(path)
    try:
        _report(progress, 20.0, "reading frames")
        total_hint = dataset.summary.total_frames or 0
        stats_parts: list[dict[str, Any]] = []
        windows: list[dict[str, Any]] = []
        segments: dict[int, list[dict[str, Any]]] = {}
        processed = 0
        chunks = 0
        for chunk in dataset.iter_frames(
            episodes or None,
            features or None,
            chunk_frames=resolved_chunk,
            max_frames=resolved_max,
        ):
            chunks += 1
            processed += chunk.frames
            stats_parts.append(compute_dataset_stats(chunk.columns))
            if not skip_anomalies:
                windows.extend(
                    shift_anomaly_windows(
                        detect_anomalies(
                            chunk.ids,
                            chunk.columns,
                            fps=dataset.summary.fps,
                            z=float(z),
                            features=features or None,
                        ),
                        chunk.offsets,
                        fps=dataset.summary.fps,
                    )
                )
            if include_segments:
                moved = shift_segments(
                    segment_by_motion(
                        chunk.ids,
                        chunk.columns,
                        feature=(features or [None])[0],
                        fps=dataset.summary.fps,
                    ),
                    chunk.offsets,
                    fps=dataset.summary.fps,
                )
                for episode, rows in moved.items():
                    segments.setdefault(episode, []).extend(rows)
            if total_hint:
                _report(progress, 20.0 + min(processed / total_hint, 1.0) * 18.0, "reading frames")
        _report(progress, 42.0, "computing statistics")
        body: dict[str, Any] = {
            "dataset": dataset.summary.to_dict(include_episodes=False),
            "stats": merge_feature_stats(stats_parts),
            "scan": {
                "streamed": True,
                "chunks": chunks,
                "frames_scanned": processed,
                "chunk_frames": resolved_chunk,
                "max_frames": resolved_max,
                "truncated": False,
            },
        }
        if not skip_anomalies:
            _report(progress, 62.0, "detecting anomalies")
            body["anomalies"] = windows
            body["anomaly_counts"] = {
                key: sum(1 for item in windows if item["severity"] == key) for key in ("error", "warn", "info")
            }
        if include_segments:
            _report(progress, 84.0, "segmenting motion")
            body["segments"] = segments
    finally:
        dataset.close()

    compute_ms = round((time.perf_counter() - started) * 1000.0, 3)
    cache_info["compute_ms"] = compute_ms
    if use_cache and cache is not None and key is not None and fingerprint is not None:
        _report(progress, 94.0, "writing cache")
        try:
            cache.put(
                key,
                path=path,
                fingerprint=str(fingerprint["sha256"]),
                params=params,
                payload=body,
                compute_ms=compute_ms,
            )
        except (OSError, sqlite3.Error, ValueError) as exc:
            cache_info["error"] = f"{type(exc).__name__}: {exc}"
    body["cache"] = cache_info
    _report(progress, 100.0, "completed")
    return body
