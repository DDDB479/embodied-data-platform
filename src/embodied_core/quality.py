"""Data quality: statistics, anomaly windows and segmentation.

Everything here runs on the numeric columns returned by an adapter's
``frames()``, so it works for LeRobot parquet, HDF5 demos and canonical JSONL
alike. The checks are deliberately robust (median / MAD instead of mean / std)
because robot logs are full of legitimate outliers.

Typical use::

    dataset = open_dataset("./my_dataset")
    ids, columns = dataset.frames()
    stats = compute_feature_stats(columns)
    windows = detect_anomalies(ids, columns, fps=30.0)
    manifest = build_labeling_tasks(dataset, windows)
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .model import Issue, Severity

EPS = 1e-9


def _episode_slices(ids: Sequence[int]) -> list[tuple[int, np.ndarray]]:
    """Contiguous runs of the same episode id, in order."""
    order = np.asarray(list(ids), dtype=np.int64)
    if order.size == 0:
        return []
    breaks = np.where(np.diff(order) != 0)[0] + 1
    starts = np.concatenate(([0], breaks))
    ends = np.concatenate((breaks, [order.size]))
    return [(int(order[start]), np.arange(start, end)) for start, end in zip(starts, ends)]


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------
def _feature_column_stats(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    flat = arr.reshape(-1)
    finite = flat[np.isfinite(flat)]
    nan_count = int(np.isnan(flat).sum())
    inf_count = int(np.isinf(flat).sum())
    body: dict[str, Any] = {
        "dims": int(arr.shape[1]) if arr.ndim > 1 else 1,
        "frames": int(arr.shape[0]),
        "nan_count": nan_count,
        "inf_count": inf_count,
        "nan_ratio": round(nan_count / max(flat.size, 1), 4),
    }
    if finite.size:
        norm = np.linalg.norm(arr, axis=1) if arr.ndim > 1 else np.abs(arr)
        body.update(
            {
                "min": float(finite.min()),
                "max": float(finite.max()),
                "mean": float(finite.mean()),
                "std": float(finite.std()),
                "median": float(np.median(finite)),
                "p05": float(np.percentile(finite, 5)),
                "p95": float(np.percentile(finite, 95)),
                "norm_mean": float(np.nanmean(norm)),
                "norm_max": float(np.nanmax(norm)),
                "near_zero_ratio": round(float(np.mean(np.abs(norm) < 1e-6)), 4),
            }
        )
    else:
        body.update({k: None for k in ("min", "max", "mean", "std", "median", "p05", "p95")})
    return body


def compute_feature_stats(columns: dict[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    return {key: _feature_column_stats(values) for key, values in columns.items()}


def compute_dataset_stats(
    columns: dict[str, np.ndarray],
    features: Sequence[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Statistics for the columns a reviewer actually cares about.

    Bookkeeping columns (episode_index, frame_index, ...) are skipped so a
    report describes robot signals rather than the encoder.
    """
    return compute_feature_stats({key: columns[key] for key in signal_features(columns, features)})


def basic_issue_scan(columns: dict[str, np.ndarray], *, nan_ratio_warn: float = 0.01) -> list[Issue]:
    return issues_from_stats(compute_feature_stats(columns), nan_ratio_warn=nan_ratio_warn)


def issues_from_stats(stats_by_key: Mapping[str, Mapping[str, Any]], *, nan_ratio_warn: float = 0.01) -> list[Issue]:
    """Turn already-merged feature statistics into the stable issue vocabulary."""

    issues: list[Issue] = []
    for key, stats in stats_by_key.items():
        if stats["nan_count"] or stats["inf_count"]:
            ratio = float(stats["nan_ratio"])
            severity = Severity.ERROR if ratio > nan_ratio_warn else Severity.WARN
            issues.append(
                Issue(
                    code="feature.non_finite",
                    severity=severity,
                    message=f"{key}: {stats['nan_count']} NaN and {stats['inf_count']} inf values ({ratio:.1%})",
                    scope="feature",
                    feature=key,
                    hint="interpolate short gaps and drop frames where the robot state is undefined",
                    detail={"nan": stats["nan_count"], "inf": stats["inf_count"]},
                )
            )
        if stats.get("frames", 0) > 1 and stats.get("std") == 0.0:
            issues.append(
                Issue(
                    code="feature.constant",
                    severity=Severity.INFO,
                    message=f"{key} never changes across {stats['frames']} frames",
                    scope="feature",
                    feature=key,
                    hint="a frozen joint or unused column adds no learning signal",
                )
            )
        if stats.get("max") is not None and abs(float(stats["max"])) > 1e6:
            issues.append(
                Issue(
                    code="feature.extreme_range",
                    severity=Severity.WARN,
                    message=f"{key} reaches {stats['max']:.3g}; check units (rad vs deg, m vs mm)",
                    scope="feature",
                    feature=key,
                )
            )
    return issues


def merge_feature_stats(parts: Sequence[Mapping[str, Mapping[str, Any]]]) -> dict[str, dict[str, Any]]:
    """Merge chunk statistics without retaining their frame arrays.

    Min/max/mean/std/non-finite counts are exact. Median and percentiles are a
    finite-count-weighted blend of chunk estimates and are marked with
    ``percentiles_estimated=true`` when more than one chunk contributes.
    """

    keys = sorted({key for part in parts for key in part})
    merged: dict[str, dict[str, Any]] = {}
    for key in keys:
        items = [part[key] for part in parts if key in part]
        if not items:
            continue
        frames = sum(int(item.get("frames", 0)) for item in items)
        dims = max(int(item.get("dims", 1)) for item in items)
        nan_count = sum(int(item.get("nan_count", 0)) for item in items)
        inf_count = sum(int(item.get("inf_count", 0)) for item in items)
        values = sum(int(item.get("frames", 0)) * int(item.get("dims", 1)) for item in items)
        finite = max(0, values - nan_count - inf_count)
        body: dict[str, Any] = {
            "dims": dims,
            "frames": frames,
            "nan_count": nan_count,
            "inf_count": inf_count,
            "nan_ratio": round(nan_count / max(values, 1), 4),
        }
        numeric_keys = ("min", "max", "mean", "std", "median", "p05", "p95", "norm_mean", "norm_max", "near_zero_ratio")
        valid = [item for item in items if item.get("mean") is not None]
        if not valid or finite == 0:
            body.update({name: None for name in numeric_keys})
            merged[key] = body
            continue
        def weight(item: Mapping[str, Any]) -> int:
            return max(
                0,
                int(item.get("frames", 0)) * int(item.get("dims", 1))
                - int(item.get("nan_count", 0))
                - int(item.get("inf_count", 0)),
            )

        mean = sum(float(item["mean"]) * weight(item) for item in valid) / finite
        variance = sum(
            weight(item) * (float(item.get("std") or 0.0) ** 2 + (float(item["mean"]) - mean) ** 2)
            for item in valid
        ) / finite
        body.update(
            {
                "min": min(float(item["min"]) for item in valid if item.get("min") is not None),
                "max": max(float(item["max"]) for item in valid if item.get("max") is not None),
                "mean": mean,
                "std": math.sqrt(max(variance, 0.0)),
                "median": sum(float(item.get("median") or 0.0) * max(1, int(item.get("frames", 0))) for item in valid)
                / sum(max(1, int(item.get("frames", 0))) for item in valid),
                "p05": sum(float(item.get("p05") or 0.0) * max(1, int(item.get("frames", 0))) for item in valid)
                / sum(max(1, int(item.get("frames", 0))) for item in valid),
                "p95": sum(float(item.get("p95") or 0.0) * max(1, int(item.get("frames", 0))) for item in valid)
                / sum(max(1, int(item.get("frames", 0))) for item in valid),
                "norm_mean": sum(float(item.get("norm_mean") or 0.0) * max(1, int(item.get("frames", 0))) for item in valid)
                / sum(max(1, int(item.get("frames", 0))) for item in valid),
                "norm_max": max(float(item.get("norm_max") or 0.0) for item in valid),
                "near_zero_ratio": sum(
                    float(item.get("near_zero_ratio") or 0.0) * max(1, int(item.get("frames", 0))) for item in valid
                )
                / sum(max(1, int(item.get("frames", 0))) for item in valid),
            }
        )
        if len(valid) > 1:
            body["percentiles_estimated"] = True
        merged[key] = body
    return merged


# --------------------------------------------------------------------------
# anomaly detection
# --------------------------------------------------------------------------
def _robust_threshold(values: np.ndarray, z: float) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return math.inf
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    return median + z * 1.4826 * mad if mad > EPS else median + z * float(np.std(finite))


def _jump_threshold(diff: np.ndarray, values: np.ndarray, z: float) -> float:
    """Robust threshold for frame-to-frame jumps.

    Plain MAD breaks on two very different inputs: perfectly uniform motion
    (MAD == 0, so the threshold collapses onto the median) and a teleport
    (two huge diffs inflate the std enough to mask themselves). The fallback
    below scales with the robust span of the signal, which keeps both cases
    sane: the teleport stays far above the threshold while ordinary ramps
    stay below it.
    """
    finite = diff[np.isfinite(diff)]
    if finite.size == 0:
        return math.inf
    median = float(np.median(finite))
    mad = 1.4826 * float(np.median(np.abs(finite - median)))
    if mad <= EPS:
        flat = values[np.isfinite(values)]
        span = float(np.percentile(flat, 95) - np.percentile(flat, 5)) if flat.size else 0.0
        mad = max(0.01 * span, EPS)
    return median + z * mad


def _runs(mask: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    start = None
    for position, flag in enumerate(mask):
        if flag and start is None:
            start = position
        elif not flag and start is not None:
            if position - start >= min_len:
                out.append((start, position - 1))
            start = None
    if start is not None and len(mask) - start >= min_len:
        out.append((start, len(mask) - 1))
    return out


METADATA_FEATURES = frozenset({"episode_index", "frame_index", "index", "task_index", "timestamp"})


def signal_features(columns: Mapping[str, Any], features: Sequence[str] | None = None) -> list[str]:
    """Keys worth scanning for physical anomalies.

    Bookkeeping columns (episode_index, frame_index, ...) are constant or ramping
    by construction, so scanning them only produces noise such as a "stall" on
    every episode boundary. An explicit ``features`` list always wins.
    """
    if features is not None:
        return [key for key in features if key in columns]
    return [key for key in columns if key not in METADATA_FEATURES]


def detect_anomalies(
    ids: Sequence[int],
    columns: dict[str, np.ndarray],
    *,
    fps: float | None = None,
    z: float = 8.0,
    min_run: int = 2,
    features: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Find suspicious windows (jumps, stalls, non-finite spans) per episode."""
    selected = {key: columns[key] for key in signal_features(columns, features)}
    if not selected:
        return []
    slices = _episode_slices(ids)
    findings: list[dict[str, Any]] = []
    for episode, index in slices:
        for key, values in selected.items():
            arr = np.asarray(values, dtype=np.float64)[index]
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            if arr.shape[0] < 3:
                continue
            finite = np.isfinite(arr).all(axis=1)
            if not finite.all():
                for start, end in _runs(~finite, 1):
                    findings.append(
                        _window("non_finite", "error", episode, start, end, fps, key, {"frames": end - start + 1})
                    )
            diff = np.linalg.norm(np.diff(arr, axis=0), axis=1)
            diff = np.where(np.isfinite(diff), diff, np.nan)
            if np.isfinite(diff).sum() >= 3:
                threshold = _jump_threshold(diff, arr, z)
                for start, end in _runs(diff > threshold, min_run):
                    findings.append(
                        _window(
                            "jump",
                            "warn",
                            episode,
                            start,
                            end + 1,
                            fps,
                            key,
                            {"peak": float(np.nanmax(diff[start : end + 1])), "threshold": round(float(threshold), 4)},
                        )
                    )
            speed = np.linalg.norm(np.diff(arr, axis=0), axis=1)
            moving = speed[speed > 0]
            reference = float(np.median(moving)) if moving.size else 0.0
            threshold = max(1e-9, 0.02 * reference)
            # A frozen span of frames [i..j] shows up as zero velocity on i..j-1,
            # so the reported window is shifted by one frame.
            for start, end in _runs(speed <= threshold, min_run):
                findings.append(
                    _window(
                        "stall",
                        "warn",
                        episode,
                        start + 1,
                        end + 1,
                        fps,
                        key,
                        {"frames": end - start + 1, "speed_threshold": round(threshold, 9)},
                    )
                )
    findings.sort(key=lambda item: (item["episode"], item["start_frame"]))
    return findings


def _window(
    code: str,
    severity: str,
    episode: int,
    start: int,
    end: int,
    fps: float | None,
    feature: str,
    detail: dict[str, Any],
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "code": code,
        "severity": severity,
        "episode": int(episode),
        "feature": feature,
        "start_frame": int(start),
        "end_frame": int(end),
        "frames": int(end - start + 1),
        "detail": detail,
    }
    if fps:
        body["start_s"] = round(start / fps, 3)
        body["end_s"] = round(end / fps, 3)
    return body


def shift_anomaly_windows(
    windows: Sequence[dict[str, Any]],
    offsets: Mapping[int, int],
    *,
    fps: float | None = None,
) -> list[dict[str, Any]]:
    """Move chunk-local anomaly windows back to episode-local coordinates."""

    shifted: list[dict[str, Any]] = []
    for window in windows:
        episode = int(window["episode"])
        offset = int(offsets.get(episode, 0))
        item = dict(window)
        item["start_frame"] = int(item["start_frame"]) + offset
        item["end_frame"] = int(item["end_frame"]) + offset
        if fps:
            item["start_s"] = round(item["start_frame"] / fps, 3)
            item["end_s"] = round(item["end_frame"] / fps, 3)
        shifted.append(item)
    return shifted


def shift_segments(
    segments: Mapping[int, Sequence[dict[str, Any]]],
    offsets: Mapping[int, int],
    *,
    fps: float | None = None,
) -> dict[int, list[dict[str, Any]]]:
    """Move chunk-local motion segments back to episode-local coordinates."""

    out: dict[int, list[dict[str, Any]]] = {}
    for episode, rows in segments.items():
        offset = int(offsets.get(int(episode), 0))
        moved: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["start_frame"] = int(item["start_frame"]) + offset
            item["end_frame"] = int(item["end_frame"]) + offset
            if fps:
                item["start_s"] = round(item["start_frame"] / fps, 3)
                item["end_s"] = round(item["end_frame"] / fps, 3)
            moved.append(item)
        out[int(episode)] = moved
    return out


# --------------------------------------------------------------------------
# segmentation
# --------------------------------------------------------------------------
def segment_by_motion(
    ids: Sequence[int],
    columns: dict[str, np.ndarray],
    *,
    feature: str | None = None,
    fps: float | None = None,
    min_segment: int = 5,
    relative_threshold: float = 0.15,
) -> dict[int, list[dict[str, Any]]]:
    """Split each episode into moving/idle segments using a feature norm.

    The threshold is relative to the feature's own 90th percentile speed, so no
    hand-tuned absolute value is needed across robots with different units.
    """
    key = feature or next((k for k in ("action", "observation.state") if k in columns), None)
    if key is None:
        key = next(iter(columns), None)
    if key is None:
        return {}
    values = np.asarray(columns[key], dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    out: dict[int, list[dict[str, Any]]] = {}
    for episode, index in _episode_slices(ids):
        arr = values[index]
        speed = np.linalg.norm(np.diff(arr, axis=0), axis=1)
        speed = np.nan_to_num(speed, nan=0.0)
        if speed.size == 0:
            continue
        p90 = float(np.percentile(speed, 90))
        threshold = max(p90 * relative_threshold, 1e-6)
        moving = speed > threshold
        segments: list[dict[str, Any]] = []
        start = 0
        for position in range(1, len(moving) + 1):
            if position == len(moving) or moving[position] != moving[start]:
                end = position
                if end - start >= min_segment:
                    segments.append(
                        {
                            "start_frame": int(start),
                            "end_frame": int(end),
                            "frames": int(end - start),
                            "label": "moving" if moving[start] else "idle",
                            "mean_speed": round(float(speed[start:end].mean()), 4),
                        }
                    )
                start = position
        if fps:
            for segment in segments:
                segment["start_s"] = round(segment["start_frame"] / fps, 3)
                segment["end_s"] = round(segment["end_frame"] / fps, 3)
        out[episode] = segments
    return out


# --------------------------------------------------------------------------
# labeling tasks
# --------------------------------------------------------------------------
SUGGESTED_LABELS = {
    "jump": "check discontinuity (sensor dropout / teleport)",
    "stall": "check idle span (dead time / slow operator)",
    "non_finite": "repair or drop frames",
}


def _window_bounds(window: dict[str, Any]) -> tuple[int, int, int]:
    """Accept both the anomaly-window and the slicing-plan key spellings."""
    episode = int(window.get("episode", window.get("episode_index", 0)))
    start = int(window.get("start_frame", window.get("start", 0)))
    end = int(window.get("end_frame", window.get("end", start)))
    return episode, start, end


def build_labeling_tasks(
    summary,
    windows: Sequence[dict[str, Any]],
    *,
    max_tasks: int = 100,
    include_segments: bool = False,
    segments: dict[int, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Turn anomaly windows into an annotation manifest for humans or VLMs."""
    tasks: list[dict[str, Any]] = []
    for window in windows[:max_tasks]:
        episode, start_frame, end_frame = _window_bounds(window)
        tasks.append(
            {
                "kind": "verify_window",
                "episode": episode,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "cues": [cue for cue in (window.get("code"), window.get("feature")) if cue],
                "question": (
                    f"Episode {episode}, frames {start_frame}-{end_frame}: "
                    f"{SUGGESTED_LABELS.get(window.get('code'), 'review this span')}. "
                    "Is the recorded behaviour valid? Answer valid / invalid / needs_fix with a short reason."
                ),
                "suggested_label": SUGGESTED_LABELS.get(window.get("code"), "review"),
            }
        )
    if include_segments and segments:
        for episode, rows in segments.items():
            for segment in rows:
                if len(tasks) >= max_tasks:
                    break
                tasks.append(
                    {
                        "kind": "describe_phase",
                        "episode": episode,
                        "start_frame": segment["start_frame"],
                        "end_frame": segment["end_frame"],
                        "cues": [segment["label"]],
                        "question": (
                            f"Episode {episode}, frames {segment['start_frame']}-{segment['end_frame']} "
                            f"({segment['label']}): briefly describe the sub-task being performed."
                        ),
                        "suggested_label": f"describe the {segment['label']} phase",
                    }
                )
    manifest_source = {
        "path": summary.path,
        "format": summary.format,
        "episodes": summary.episode_count,
        "task_bounds": [
            [task["kind"], task["episode"], task["start_frame"], task["end_frame"]] for task in tasks
        ],
    }
    manifest_id = "lm_" + hashlib.sha256(
        json.dumps(manifest_source, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:12]
    for position, task in enumerate(tasks):
        task["task_id"] = "task_" + hashlib.sha256(f"{manifest_id}:{position}".encode("utf-8")).hexdigest()[:12]
        task["status"] = "pending"
    return {
        "schema_version": 1,
        "manifest_id": manifest_id,
        "dataset": {
            "path": summary.path,
            "format": summary.format,
            "episodes": summary.episode_count,
            "fps": summary.fps,
            "features": [f.key for f in summary.features],
        },
        "task_count": len(tasks),
        "tasks": tasks,
        "instructions": (
            "For each task, inspect the given frame range and answer the question. "
            "Return JSON lines: {\"task_id\": <task_id>, \"answer\": valid|invalid|needs_fix, \"reason\": ...}. "
            "Legacy {\"id\": <task index>, \"answer\": ...} is also accepted."
        ),
    }


# --------------------------------------------------------------------------
# training windows
# --------------------------------------------------------------------------
def make_windows(lengths: Sequence[int], window: int, stride: int | None = None) -> list[dict[str, int]]:
    """Fixed-length slicing plan, used by export/dataloader scaffolding.

    Windows use the same `episode`/`start_frame`/`end_frame` vocabulary as the
    anomaly report, so the two can be fed to the same consumers.
    """
    stride = stride or window
    out: list[dict[str, int]] = []
    for episode, length in enumerate(lengths):
        if length < window:
            continue
        for start in range(0, length - window + 1, stride):
            out.append({"episode": episode, "start_frame": start, "end_frame": start + window, "frames": window})
    return out
