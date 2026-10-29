"""Adapter protocol shared by every supported on-disk layout.

An adapter answers three questions:

* ``detect(root)`` — does this path look like my format, and how sure am I?
* ``load(root)``   — give me a :class:`DatasetObject` with a summary attached.
* ``frames(...)``  — give me numeric columns for one or more episodes.

Adapters that only understand structure (MCAP, rosbag, RLDS) simply raise
:class:`FramesUnavailable` from ``frames`` — the validator then runs the
structural checks that do not need tabular data.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np

from ..model import DatasetSummary, EpisodeRecord, FeatureSpec

STATE_PREFIXES = ("observation.state", "observation/joint", "state", "qpos", "proprio")
ACTION_PREFIXES = ("action", "actions", "ctrl", "command")
IMAGE_HINTS = ("image", "rgb", "depth", "camera", "video")
LANGUAGE_HINTS = ("language", "instruction", "task", "prompt", "caption")

DEFAULT_CHUNK_FRAMES = 50_000
DEFAULT_MAX_FRAMES = 2_000_000


class FramesUnavailable(RuntimeError):
    """Raised when tabular frame access needs an optional dependency."""

    def __init__(self, reason: str, *, install: str | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.install = install


class FrameLimitExceeded(RuntimeError):
    """Raised when a caller asks for more frames than its memory budget allows."""

    def __init__(self, limit: int, frames: int, *, hint: str | None = None) -> None:
        self.limit = int(limit)
        self.frames = int(frames)
        self.hint = hint
        super().__init__(
            f"frame limit exceeded: {self.frames} > {self.limit}; "
            "use iter_frames(..., chunk_frames=...) or raise max_frames/EMBODIED_MAX_FRAMES"
        )


def _env_positive_int(name: str, default: int, *, allow_zero: bool = False) -> int:
    try:
        value = int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default
    if value < 0 or (value == 0 and not allow_zero):
        return default
    return value


def resolve_chunk_frames(value: int | None) -> int:
    """Resolve a per-chunk frame budget."""

    if value is not None:
        resolved = int(value)
        if resolved < 1:
            raise ValueError("chunk_frames must be at least 1")
        return resolved
    return _env_positive_int("EMBODIED_CHUNK_FRAMES", DEFAULT_CHUNK_FRAMES)


def resolve_max_frames(value: int | None) -> int:
    """Resolve the total materialization budget. Zero means unlimited."""

    if value is not None:
        resolved = int(value)
        if resolved < 0:
            raise ValueError("max_frames must be >= 0 (0 means unlimited)")
        return resolved
    return _env_positive_int("EMBODIED_MAX_FRAMES", DEFAULT_MAX_FRAMES, allow_zero=True)


@dataclass
class FrameChunk:
    """One bounded slice of numeric frames.

    ``offsets`` maps an episode id to the local frame number of that episode's
    first occurrence in this chunk. Consumers can shift chunk-local anomaly
    windows back to episode-local coordinates with it.
    """

    ids: list[int]
    columns: dict[str, np.ndarray]
    offsets: dict[int, int] = field(default_factory=dict)

    @property
    def frames(self) -> int:
        return len(self.ids)


@dataclass
class Detection:
    format: str
    version: str | None = None
    confidence: float = 0.5
    evidence: list[str] = field(default_factory=list)
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "format": self.format,
            "version": self.version,
            "confidence": round(self.confidence, 2),
            "evidence": self.evidence,
        }
        if self.detail:
            body["detail"] = self.detail
        return body


def guess_kind(key: str, shape: Sequence[int] | None = None, dtype: str = "") -> str:
    """Classify a feature key into the canonical kinds used by checks and QC."""
    low = key.lower()
    if any(h in low for h in IMAGE_HINTS) and not low.endswith("_shape"):
        return "image" if "image" in low or "rgb" in low or "depth" in low else "video"
    if any(h in low for h in LANGUAGE_HINTS):
        return "language"
    if any(low.startswith(p) or f"/{p}" in low or f".{p}" in low for p in ACTION_PREFIXES):
        return "action"
    if any(low.startswith(p) or f"/{p}" in low or f".{p}" in low for p in STATE_PREFIXES):
        return "state"
    if dtype in ("string", "str", "object"):
        return "language"
    return "scalar"


def to_numeric(value: Any) -> np.ndarray | None:
    """Best-effort conversion of a column of values into a 2-D float array."""
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        arr = value
    else:
        try:
            arr = np.asarray(value)
        except Exception:
            return None
    if arr.dtype == object:
        # Lists (e.g. parquet list<float>) arrive as object arrays of arrays.
        try:
            arr = np.stack([np.asarray(row, dtype=np.float64).ravel() for row in arr])
        except Exception:
            return None
    if arr.dtype.kind not in "fiub":
        return None
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr.astype(np.float64, copy=False)


class DatasetObject:
    """A loaded dataset: summary plus optional frame access."""

    def __init__(self, root: Path, summary: DatasetSummary, *, capabilities: Iterable[str] = ()) -> None:
        self.root = Path(root)
        self.summary = summary
        self.capabilities = set(capabilities)

    # -- lifecycle -------------------------------------------------------
    def close(self) -> None:  # pragma: no cover - adapters keep no handles by default
        pass

    def __enter__(self) -> "DatasetObject":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- data ------------------------------------------------------------
    def frames(
        self,
        episodes: Sequence[int] | None = None,
        features: Sequence[str] | None = None,
        *,
        max_frames: int | None = None,
    ) -> tuple[list[int], dict[str, np.ndarray]]:
        """Return ``(episode_ids, columns)`` for tabular features.

        ``columns`` is keyed by feature name and always 2-D (frames x dim).
        ``max_frames`` is a materialization guard. Prefer :meth:`iter_frames`
        for datasets that do not fit in memory.
        """
        raise FramesUnavailable(f"{self.summary.format} does not expose tabular frames through this adapter")

    def iter_frames(
        self,
        episodes: Sequence[int] | None = None,
        features: Sequence[str] | None = None,
        *,
        chunk_frames: int | None = None,
        max_frames: int | None = None,
    ) -> Iterator[FrameChunk]:
        """Yield bounded frame chunks.

        The default implementation slices ``frames()``; adapters with a native
        iterator override it so large parquet/HDF5 files never materialize at
        once. ``max_frames=0`` means unlimited.
        """

        limit = resolve_max_frames(max_frames)
        size = resolve_chunk_frames(chunk_frames)
        ids, columns = self.frames(episodes, features, max_frames=limit)
        seen: dict[int, int] = {}
        for start in range(0, len(ids), size):
            stop = min(len(ids), start + size)
            chunk_ids = list(ids[start:stop])
            offsets = {episode: seen.get(episode, 0) for episode in dict.fromkeys(chunk_ids)}
            for episode in chunk_ids:
                seen[episode] = seen.get(episode, 0) + 1
            yield FrameChunk(chunk_ids, {key: values[start:stop] for key, values in columns.items()}, offsets)

    def help(self) -> dict[str, Any]:
        return {
            "format": self.summary.format,
            "capabilities": sorted(self.capabilities),
            "optional_dependencies": self.optional_dependencies(),
        }

    @staticmethod
    def optional_dependencies() -> dict[str, str]:
        return {}

    # -- helpers ---------------------------------------------------------
    def numeric_features(self) -> list[FeatureSpec]:
        return [f for f in self.summary.features if f.kind in ("state", "action", "scalar")]

    def episode_ids(self) -> list[int]:
        return [e.index for e in self.summary.episodes]

    def feature_spec(self, key: str) -> FeatureSpec | None:
        for feature in self.summary.features:
            if feature.key == key:
                return feature
        return None


def episode_records_from_lengths(
    lengths: dict[int, int],
    *,
    fps: float | None,
    tasks: dict[int, str] | None = None,
    paths: dict[int, str] | None = None,
) -> list[EpisodeRecord]:
    out: list[EpisodeRecord] = []
    for index in sorted(lengths):
        length = int(lengths[index])
        out.append(
            EpisodeRecord(
                index=index,
                length=length,
                task=(tasks or {}).get(index),
                fps=fps,
                duration_s=(length / fps) if fps else None,
                path=(paths or {}).get(index),
            )
        )
    return out


def collect_frame_chunks(chunks: Iterable[FrameChunk], *, max_frames: int | None = None) -> tuple[list[int], dict[str, np.ndarray]]:
    """Collect chunks for the legacy ``frames()`` API with a hard frame cap."""

    limit = resolve_max_frames(max_frames)
    ids: list[int] = []
    parts: dict[str, list[np.ndarray]] = {}
    common: set[str] | None = None
    for chunk in chunks:
        seen = len(ids) + len(chunk.ids)
        if limit and seen > limit:
            raise FrameLimitExceeded(limit, seen)
        ids.extend(chunk.ids)
        keys = set(chunk.columns)
        common = keys if common is None else common & keys
        for key, values in chunk.columns.items():
            parts.setdefault(key, []).append(values)
    columns: dict[str, np.ndarray] = {}
    for key in common or set():
        try:
            columns[key] = np.concatenate(parts[key], axis=0)
        except ValueError:
            continue
    return ids, columns
