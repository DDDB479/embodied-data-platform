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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np

from ..model import DatasetSummary, EpisodeRecord, FeatureSpec

STATE_PREFIXES = ("observation.state", "observation/joint", "state", "qpos", "proprio")
ACTION_PREFIXES = ("action", "actions", "ctrl", "command")
IMAGE_HINTS = ("image", "rgb", "depth", "camera", "video")
LANGUAGE_HINTS = ("language", "instruction", "task", "prompt", "caption")


class FramesUnavailable(RuntimeError):
    """Raised when tabular frame access needs an optional dependency."""

    def __init__(self, reason: str, *, install: str | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.install = install


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
    ) -> tuple[list[int], dict[str, np.ndarray]]:
        """Return ``(episode_ids, columns)`` for tabular features.

        ``columns`` is keyed by feature name and always 2-D (frames x dim).
        """
        raise FramesUnavailable(f"{self.summary.format} does not expose tabular frames through this adapter")

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
