"""Canonical data model shared by every adapter, check and tool."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable


class Severity(str, Enum):
    ERROR = "error"
    WARN = "warn"
    INFO = "info"

    @property
    def weight(self) -> int:
        return {"error": 30, "warn": 8, "info": 1}[self.value]


def _clean(value: Any) -> Any:
    """Make a value JSON-serialisable (numpy scalars, NaN, tuples, paths)."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return round(value, 6)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (list, tuple, set)):
        return [_clean(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    item = getattr(value, "item", None)
    if callable(item):  # numpy scalar
        try:
            return _clean(item())
        except Exception:  # pragma: no cover - defensive
            pass
    return str(value)


@dataclass
class FeatureSpec:
    """One column/stream of an episode (state, action, camera, language...)."""

    key: str
    dtype: str = "unknown"
    shape: tuple[int, ...] = ()
    kind: str = "other"
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def dim(self) -> int:
        out = 1
        for size in self.shape:
            out *= int(size)
        return out

    def to_dict(self) -> dict[str, Any]:
        body = {"key": self.key, "dtype": self.dtype, "shape": list(self.shape), "kind": self.kind}
        if self.extra:
            body["extra"] = _clean(self.extra)
        return body


@dataclass
class EpisodeRecord:
    """Episode-level metadata. ``data`` stays in the adapter, never here."""

    index: int
    length: int | None = None
    task: str | None = None
    duration_s: float | None = None
    fps: float | None = None
    path: str | None = None
    start_time_s: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        body = {
            "index": self.index,
            "length": self.length,
            "task": self.task,
            "duration_s": self.duration_s,
            "fps": self.fps,
        }
        if self.path:
            body["path"] = self.path
        if self.start_time_s is not None:
            body["start_time_s"] = self.start_time_s
        if self.meta:
            body["meta"] = _clean(self.meta)
        return body


@dataclass
class Issue:
    """A finding produced by a check. ``code`` is stable and machine-readable."""

    code: str
    severity: Severity
    message: str
    scope: str = "dataset"  # dataset | episode | feature | stream
    episode: int | None = None
    feature: str | None = None
    hint: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        body = {
            "code": self.code,
            "severity": self.severity.value if isinstance(self.severity, Severity) else str(self.severity),
            "scope": self.scope,
            "message": self.message,
        }
        if self.episode is not None:
            body["episode"] = self.episode
        if self.feature:
            body["feature"] = self.feature
        if self.hint:
            body["hint"] = self.hint
        if self.detail:
            body["detail"] = _clean(self.detail)
        return body


@dataclass
class DatasetSummary:
    """Everything a tool needs to reason about a dataset without loading it."""

    path: str
    format: str
    version: str | None = None
    fps: float | None = None
    robot_type: str | None = None
    features: list[FeatureSpec] = field(default_factory=list)
    episodes: list[EpisodeRecord] = field(default_factory=list)
    tasks: list[str] = field(default_factory=list)
    split: str | None = None
    notes: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)

    # ---- derived -------------------------------------------------------
    @property
    def episode_count(self) -> int:
        return len(self.episodes)

    @property
    def total_frames(self) -> int | None:
        lengths = [e.length for e in self.episodes if e.length is not None]
        return sum(int(v) for v in lengths) if lengths else None

    @property
    def total_duration_s(self) -> float | None:
        durations = [e.duration_s for e in self.episodes if e.duration_s is not None]
        if durations:
            return float(sum(durations))
        if self.total_frames and self.fps:
            return float(self.total_frames) / float(self.fps)
        return None

    def feature_keys(self) -> list[str]:
        return [f.key for f in self.features]

    def modality_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for feature in self.features:
            counts[feature.kind] = counts.get(feature.kind, 0) + 1
        return counts

    def to_dict(self, *, include_episodes: bool = True) -> dict[str, Any]:
        body: dict[str, Any] = {
            "path": self.path,
            "format": self.format,
            "version": self.version,
            "episodes": self.episode_count,
            "total_frames": self.total_frames,
            "total_duration_s": self.total_duration_s,
            "fps": self.fps,
            "robot_type": self.robot_type,
            "split": self.split,
            "tasks": _clean(self.tasks),
            "features": [f.to_dict() for f in self.features],
            "modalities": self.modality_counts(),
            "notes": _clean(self.notes),
            "sources": self.sources,
        }
        if include_episodes:
            body["episode_list"] = [e.to_dict() for e in self.episodes]
        if self.meta:
            body["meta"] = _clean(self.meta)
        return body


def summarize_issues(issues: Iterable[Issue]) -> dict[str, Any]:
    """Aggregate issues into counts plus a 0-100 health score."""
    items = list(issues)
    counts = {s.value: 0 for s in Severity}
    for issue in items:
        sev = issue.severity.value if isinstance(issue.severity, Severity) else str(issue.severity)
        counts[sev] = counts.get(sev, 0) + 1
    penalty = sum(
        (issue.severity.weight if isinstance(issue.severity, Severity) else Severity.WARN.weight)
        for issue in items
    )
    return {
        "errors": counts.get("error", 0),
        "warnings": counts.get("warn", 0),
        "infos": counts.get("info", 0),
        "score": max(0, 100 - penalty),
    }


def to_jsonable(obj: Any) -> Any:
    """Recursively convert dataclasses/containers into JSON-safe structures."""
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    return _clean(obj)
