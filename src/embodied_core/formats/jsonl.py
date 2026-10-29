"""Canonical JSONL interchange format (``embodied-jsonl``).

This is the studio's own lossless text format, meant for small datasets, tests,
fixtures and for moving a slice of an episode between tools. Two shapes are
accepted:

Layout A — directory:
    dataset.json        {"format":"embodied-jsonl","fps":30,"robot_type":"x",
                         "features":{"observation.state":{"dtype":"float32","shape":[7]}}}
    episodes.jsonl      one episode per line

Layout B — a single ``.jsonl`` file, where the first line may be a header:

    {"__header__": true, "fps": 30, "features": {...}}
    {"episode_index": 0, "task": "pick", "frames": [{"action": [...]}, ...]}
    {"episode_index": 1, "task": "pick", "columns": {"action": [[...]], "reward": [...]}}

Each episode line carries either ``frames`` (a list of per-frame objects) or
``columns`` (a mapping feature -> list of values). Episodes default to the line
order when ``episode_index`` is absent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, Sequence

from ..model import DatasetSummary, EpisodeRecord, FeatureSpec
from .base import (
    DatasetObject,
    Detection,
    FrameChunk,
    FrameLimitExceeded,
    FramesUnavailable,
    collect_frame_chunks,
    guess_kind,
    resolve_chunk_frames,
    resolve_max_frames,
    to_numeric,
)

HEADER_KEY = "__header__"
FORMAT = "embodied-jsonl"


def _jsonl_files(root: Path) -> list[Path]:
    if root.is_file() and root.suffix.lower() in {".jsonl", ".ndjson"}:
        return [root]
    if root.is_dir():
        candidate = root / "episodes.jsonl"
        if candidate.is_file():
            return [candidate]
        return sorted(p for p in root.glob("*.jsonl") if p.is_file())
    return []


def _read_lines(path: Path) -> Iterator[Any]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            text = line.strip()
            if not text:
                continue
            try:
                yield json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name}:{line_no}: {exc}") from exc


def detect(root: Path) -> Detection | None:
    files = _jsonl_files(root)
    if not files:
        return None
    if root.is_dir():
        info = root / "dataset.json"
        if info.is_file():
            try:
                payload = json.loads(info.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = {}
            if str(payload.get("format", "")).lower() == FORMAT:
                return Detection(FORMAT, version=str(payload.get("version", "1")), confidence=0.98, evidence=["dataset.json: format=embodied-jsonl"])
            return Detection(FORMAT, version=None, confidence=0.7, evidence=["dataset.json + episodes.jsonl"])
    try:
        first = next(_read_lines(files[0]), None)
    except ValueError:
        return None
    if isinstance(first, dict) and (HEADER_KEY in first or "episode_index" in first or "frames" in first):
        return Detection(FORMAT, version=None, confidence=0.8, evidence=[f"{files[0].name}: JSONL episodes"])
    return None


class JSONLDataset(DatasetObject):
    def __init__(self, root: Path, summary: DatasetSummary) -> None:
        super().__init__(root, summary, capabilities={"frames", "episodes"})

    def rows(self, episodes=None, features=None) -> tuple[list[int], list[dict[str, Any]]]:
        """Raw per-frame objects, preserving non-numeric values such as strings."""
        wanted = set(episodes) if episodes is not None else None
        feature_filter = set(features) if features else None
        ids: list[int] = []
        rows: list[dict[str, Any]] = []
        ordinal = 0
        for path in _jsonl_files(self.root):
            for line_no, payload in enumerate(_read_lines(path), 1):
                if isinstance(payload, dict) and payload.get(HEADER_KEY):
                    continue
                episode = _episode_rows(payload, line_no)
                index = int(episode.get("episode_index", ordinal))
                ordinal += 1
                if wanted is not None and index not in wanted:
                    continue
                for row in _frame_rows(episode):
                    ids.append(index)
                    rows.append(_filter_row(row, feature_filter))
        if not rows:
            raise FramesUnavailable("no frames matched the request")
        return ids, rows

    def iter_frames(
        self,
        episodes: Sequence[int] | None = None,
        features: Sequence[str] | None = None,
        *,
        chunk_frames: int | None = None,
        max_frames: int | None = None,
    ) -> Iterator[FrameChunk]:
        wanted = set(episodes) if episodes is not None else None
        feature_filter = set(features) if features else None
        size = resolve_chunk_frames(chunk_frames)
        limit = resolve_max_frames(max_frames)
        total = 0
        seen: dict[int, int] = {}
        chunk_ids: list[int] = []
        chunk_rows: list[dict[str, Any]] = []
        emitted = False
        ordinal = 0

        def flush() -> FrameChunk | None:
            nonlocal chunk_ids, chunk_rows, emitted
            if not chunk_ids:
                return None
            columns = _numeric_columns(chunk_rows, features)
            offsets = {episode: seen.get(episode, 0) for episode in dict.fromkeys(chunk_ids)}
            for episode in chunk_ids:
                seen[episode] = seen.get(episode, 0) + 1
            chunk = FrameChunk(chunk_ids, columns, offsets) if columns else None
            chunk_ids = []
            chunk_rows = []
            if chunk is not None:
                emitted = True
            return chunk

        for path in _jsonl_files(self.root):
            for line_no, payload in enumerate(_read_lines(path), 1):
                if isinstance(payload, dict) and payload.get(HEADER_KEY):
                    continue
                episode = _episode_rows(payload, line_no)
                index = int(episode.get("episode_index", ordinal))
                ordinal += 1
                if wanted is not None and index not in wanted:
                    continue
                for row in _frame_rows(episode):
                    total += 1
                    if limit and total > limit:
                        raise FrameLimitExceeded(limit, total)
                    chunk_ids.append(index)
                    chunk_rows.append(_filter_row(row, feature_filter))
                    if len(chunk_ids) >= size:
                        chunk = flush()
                        if chunk is not None:
                            yield chunk
        chunk = flush()
        if chunk is not None:
            yield chunk
        if not emitted:
            raise FramesUnavailable("no numeric frames matched the request")

    def frames(
        self,
        episodes: Sequence[int] | None = None,
        features: Sequence[str] | None = None,
        *,
        max_frames: int | None = None,
    ) -> tuple[list[int], dict[str, Any]]:
        return collect_frame_chunks(
            self.iter_frames(episodes, features, max_frames=max_frames),
            max_frames=max_frames,
        )


def _frame_rows(episode: dict[str, Any]) -> Iterator[dict[str, Any]]:
    columns = episode.get("columns")
    if isinstance(columns, dict):
        length = max((len(v) for v in columns.values() if isinstance(v, list)), default=0)
        for position in range(length):
            yield {
                key: values[position]
                for key, values in columns.items()
                if isinstance(values, list) and position < len(values)
            }
        return
    for row in episode.get("frames") or []:
        if isinstance(row, dict):
            yield row


def _numeric_columns(rows: list[dict[str, Any]], features: Sequence[str] | None) -> dict[str, Any]:
    keep = set(features) if features else None
    columns: dict[str, Any] = {}
    keys = {key for row in rows for key in row}
    for key in sorted(keys):
        if keep is not None and key not in keep:
            continue
        numeric = to_numeric([row.get(key) for row in rows])
        if numeric is not None:
            columns[key] = numeric
    return columns


def _filter_row(row: dict[str, Any], keep: set[str] | None) -> dict[str, Any]:
    if keep is None:
        return dict(row)
    return {k: v for k, v in row.items() if k in keep}


def _infer_features(episodes: list[dict[str, Any]], declared: dict[str, Any]) -> list[FeatureSpec]:
    specs: dict[str, FeatureSpec] = {}
    for key, meta in (declared or {}).items():
        if isinstance(meta, dict):
            specs[key] = FeatureSpec(
                key=key,
                dtype=str(meta.get("dtype", "unknown")),
                shape=tuple(int(v) for v in (meta.get("shape") or ())),
                kind=str(meta.get("kind") or guess_kind(key)),
            )
        else:
            specs[key] = FeatureSpec(key=key, dtype="unknown", shape=(), kind=guess_kind(key))

    def remember(key: str, value: Any) -> None:
        if key in specs:
            return
        dtype, shape = _describe_value(value)
        specs[key] = FeatureSpec(key=key, dtype=dtype, shape=shape, kind=guess_kind(key))

    for episode in episodes[:8]:
        columns = episode.get("columns")
        if isinstance(columns, dict):
            for key, values in columns.items():
                sample = next((v for v in values if v is not None), None) if isinstance(values, list) else None
                remember(key, sample)
        for row in (episode.get("frames") or [])[:4]:
            if isinstance(row, dict):
                for key, value in row.items():
                    remember(key, value)
    return list(specs.values())


def _describe_value(value: Any) -> tuple[str, tuple[int, ...]]:
    if value is None:
        return "unknown", ()
    if isinstance(value, bool):
        return "bool", ()
    if isinstance(value, (int, float)):
        return ("int64" if isinstance(value, int) else "float64"), ()
    if isinstance(value, str):
        return "string", ()
    if isinstance(value, list):
        if value and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value):
            dtype = "float64" if any(isinstance(v, float) for v in value) else "int64"
            return dtype, (len(value),)
        return "list", (len(value),)
    if isinstance(value, dict):
        return "object", ()
    return "unknown", ()


def _episode_rows(payload: Any, line_no: int) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"line {line_no}: episode must be an object")
    return payload


def load(root: Path, detection: Detection | None = None) -> JSONLDataset:
    files = _jsonl_files(root)
    declared: dict[str, Any] = {}
    meta: dict[str, Any] = {"files": [p.name for p in files]}
    notes: list[str] = []
    if root.is_dir() and (root / "dataset.json").is_file():
        try:
            info = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
            declared = info.get("features") or {}
            meta.update({k: info.get(k) for k in ("fps", "robot_type", "version", "tasks") if info.get(k) is not None})
        except (OSError, json.JSONDecodeError) as exc:
            notes.append(f"dataset.json unreadable: {exc}")

    samples: list[dict[str, Any]] = []
    records: list[EpisodeRecord] = []
    ordinal = 0
    for path in files:
        for line_no, payload in enumerate(_read_lines(path), 1):
            if isinstance(payload, dict) and payload.get(HEADER_KEY):
                meta.setdefault("fps", payload.get("fps"))
                meta.setdefault("robot_type", payload.get("robot_type"))
                declared = declared or (payload.get("features") or {})
                continue
            row = _episode_rows(payload, line_no)
            index = int(row.get("episode_index", ordinal))
            ordinal += 1
            frames = row.get("frames")
            columns = row.get("columns")
            length = len(frames) if isinstance(frames, list) else (
                max((len(v) for v in columns.values() if isinstance(v, list)), default=0)
                if isinstance(columns, dict)
                else None
            )
            fps = row.get("fps") or meta.get("fps")
            duration = (length / float(fps)) if (length and fps) else None
            records.append(
                EpisodeRecord(
                    index=index,
                    length=length,
                    task=row.get("task"),
                    duration_s=duration,
                    fps=float(fps) if fps else None,
                    meta={k: v for k, v in row.items() if k not in {"frames", "columns", "episode_index", "task"}},
                )
            )
            if len(samples) < 8:
                sample = {k: v for k, v in row.items() if k not in {"frames", "columns"}}
                if isinstance(frames, list):
                    sample["frames"] = frames[:4]
                if isinstance(columns, dict):
                    sample["columns"] = {
                        key: value[:8] if isinstance(value, list) else value for key, value in columns.items()
                    }
                samples.append(sample)

    features = _infer_features(samples, declared)
    if not records:
        notes.append("no episode lines found")

    tasks = meta.get("tasks") or []
    if not tasks:
        tasks = sorted({r.task for r in records if r.task})
    summary = DatasetSummary(
        path=str(root),
        format=FORMAT,
        version=str(meta.get("version") or "1"),
        fps=float(meta["fps"]) if meta.get("fps") else None,
        robot_type=meta.get("robot_type"),
        features=features,
        episodes=records,
        tasks=list(tasks),
        notes=notes,
        meta=meta,
        sources=[p.name for p in files],
    )
    dataset = JSONLDataset(root, summary)
    return dataset
