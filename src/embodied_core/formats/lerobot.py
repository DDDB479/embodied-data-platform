"""LeRobot dataset adapter (v2.0 / v2.1 / v3.0 layouts).

v2.x keeps one parquet per episode plus JSONL metadata::

    meta/info.json, meta/episodes.jsonl, meta/tasks.jsonl
    data/chunk-000/episode_000000.parquet

v3.0 groups episodes into chunked files and moves metadata to parquet::

    meta/info.json, meta/tasks.parquet, meta/episodes/chunk-000/file-000.parquet
    data/chunk-000/file-000.parquet

This adapter reads the layout defensively: metadata first, then a bounded scan
of ``data/**.parquet`` when metadata is missing or partial.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from ..model import DatasetSummary, EpisodeRecord, FeatureSpec
from .base import (
    DatasetObject,
    Detection,
    FramesUnavailable,
    episode_records_from_lengths,
    guess_kind,
    to_numeric,
)

_EPISODE_FILE = re.compile(r"episode_(\d+)\.parquet$", re.IGNORECASE)
_INDEX_COLUMNS = ("episode_index", "episode_id", "episode")


def _pyarrow() -> Any:
    try:
        import pyarrow.parquet as pq  # type: ignore

        return pq
    except Exception as exc:  # pragma: no cover - depends on environment
        raise FramesUnavailable(
            f"reading LeRobot parquet data requires pyarrow ({exc})",
            install="pip install 'embodied-core[parquet]'",
        ) from exc


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_or_empty(path: Path) -> dict[str, Any]:
    """Tolerate a missing/invalid metadata file so validation can report it."""
    try:
        return _read_json(path)
    except (OSError, json.JSONDecodeError):
        return {}


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _parquet_paths(root: Path) -> list[Path]:
    return sorted((root / "data").rglob("*.parquet"))


def _meta_episode_parquet(root: Path) -> list[Path]:
    return sorted((root / "meta" / "episodes").rglob("*.parquet"))


def detect(root: Path) -> Detection | None:
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        return None
    evidence = ["meta/info.json"]
    version = None
    try:
        info = _read_json(info_path)
        version = str(info.get("codebase_version") or info.get("version") or "") or None
    except Exception:
        info = {}
    if (root / "meta" / "episodes.jsonl").is_file():
        evidence.append("meta/episodes.jsonl")
    if _meta_episode_parquet(root):
        evidence.append("meta/episodes/**/*.parquet")
    if _parquet_paths(root):
        evidence.append("data/**/*.parquet")
    confidence = 0.98 if version else 0.9
    detail = None
    if not _parquet_paths(root):
        detail = "metadata only: no data/**/*.parquet found yet"
        confidence = min(confidence, 0.7)
    return Detection("lerobot", version=version, confidence=confidence, evidence=evidence, detail=detail)


def _features(info: dict[str, Any]) -> list[FeatureSpec]:
    raw = info.get("features") or {}
    out: list[FeatureSpec] = []
    for key, spec in raw.items():
        if not isinstance(spec, dict):
            continue
        shape = spec.get("shape") or []
        if not isinstance(shape, (list, tuple)):
            shape = [shape]
        dtype = str(spec.get("dtype") or "unknown")
        out.append(
            FeatureSpec(
                key=str(key),
                dtype=dtype,
                shape=tuple(int(s) for s in shape if isinstance(s, (int, float))),
                kind=guess_kind(str(key), shape, dtype),
                extra={k: v for k, v in spec.items() if k not in ("dtype", "shape")},
            )
        )
    return out


def _tasks_from_meta(root: Path) -> list[str]:
    jsonl = root / "meta" / "tasks.jsonl"
    if jsonl.is_file():
        tasks = []
        for row in _iter_jsonl(jsonl):
            task = row.get("task")
            if task:
                tasks.append(str(task))
        return tasks
    parquet = root / "meta" / "tasks.parquet"
    if parquet.is_file():
        try:
            pq = _pyarrow()
            table = pq.read_table(parquet)
            column = table.column("task") if "task" in table.column_names else None
            if column is not None:
                return [str(v) for v in column.to_pylist()]
        except Exception:
            return []
    return []


def _episodes_from_meta(root: Path) -> tuple[dict[int, int], dict[int, str], list[str]]:
    """Return (lengths, tasks_by_episode, notes)."""
    notes: list[str] = []
    lengths: dict[int, int] = {}
    tasks: dict[int, str] = {}
    jsonl = root / "meta" / "episodes.jsonl"
    if jsonl.is_file():
        for row in _iter_jsonl(jsonl):
            index = row.get("episode_index", row.get("episode_id"))
            if index is None:
                continue
            index = int(index)
            if row.get("length") is not None:
                lengths[index] = int(row["length"])
            task_list = row.get("tasks")
            if isinstance(task_list, list) and task_list:
                tasks[index] = str(task_list[0])
            elif row.get("task"):
                tasks[index] = str(row["task"])
        if lengths:
            return lengths, tasks, notes
    meta_files = _meta_episode_parquet(root)
    if meta_files:
        try:
            pq = _pyarrow()
            for path in meta_files:
                table = pq.read_table(path)
                names = set(table.column_names)
                index_col = next((c for c in _INDEX_COLUMNS if c in names), None)
                if index_col is None:
                    notes.append(f"{path.name}: no episode index column")
                    continue
                rows = table.to_pydict()
                for i, index in enumerate(rows[index_col]):
                    if index is None:
                        continue
                    index = int(index)
                    if "length" in rows and rows["length"][i] is not None:
                        lengths[index] = int(rows["length"][i])
                    if "tasks" in rows and rows["tasks"][i]:
                        value = rows["tasks"][i]
                        if isinstance(value, (list, tuple)):
                            if value:
                                tasks[index] = str(value[0])
                        else:
                            tasks[index] = str(value)
        except FramesUnavailable as exc:
            notes.append(f"episode metadata unreadable: {exc.reason}")
        except Exception as exc:  # pragma: no cover - malformed third-party files
            notes.append(f"episode metadata unreadable: {exc}")
    return lengths, tasks, notes


def _episodes_from_data(root: Path, notes: list[str]) -> dict[int, int]:
    """Last-resort episode lengths derived from the data files themselves."""
    lengths: dict[int, int] = {}
    for path in _parquet_paths(root):
        match = _EPISODE_FILE.search(path.name)
        if match and path.parent.name.startswith("chunk-"):
            try:
                pq = _pyarrow()
                meta = pq.read_metadata(path)
                lengths[int(match.group(1))] = int(meta.num_rows)
                continue
            except Exception as exc:
                notes.append(f"{path.name}: {exc}")
                continue
        try:
            pq = _pyarrow()
            table = pq.read_table(path, columns=None)
            if "episode_index" in table.column_names:
                values = table.column("episode_index").to_pylist()
                counts: dict[int, int] = defaultdict(int)
                for value in values:
                    if value is not None:
                        counts[int(value)] += 1
                for index, count in counts.items():
                    lengths[index] = lengths.get(index, 0) + count
            elif "file_index" not in table.column_names:
                notes.append(f"{path.name}: no episode_index column; counted as one episode")
                lengths[len(lengths)] = int(table.num_rows)
        except FramesUnavailable as exc:
            notes.append(exc.reason)
            break
        except Exception as exc:
            notes.append(f"{path.name}: {exc}")
    return lengths


class LeRobotDataset(DatasetObject):
    """LeRobot v2/v3 dataset with optional tabular frame access."""

    def __init__(self, root: Path, summary: DatasetSummary) -> None:
        super().__init__(root, summary, capabilities={"frames", "episodes", "features"})

    @staticmethod
    def optional_dependencies() -> dict[str, str]:
        return {"pyarrow": "parquet data access"}

    def frames(
        self,
        episodes: Sequence[int] | None = None,
        features: Sequence[str] | None = None,
    ) -> tuple[list[int], dict[str, np.ndarray]]:
        pq = _pyarrow()
        wanted = set(episodes) if episodes is not None else None
        chunks: list[Any] = []
        for path in _parquet_paths(self.root):
            match = _EPISODE_FILE.search(path.name)
            file_episode = int(match.group(1)) if match else None
            if file_episode is not None and wanted is not None and file_episode not in wanted:
                continue
            table = pq.read_table(path)
            names = [n for n in table.column_names if n not in ("episode_index",) or True]
            arrays: dict[str, Any] = {}
            for name in names:
                column = table.column(name)
                if str(column.type) in ("string", "large_string", "bool"):
                    continue
                try:
                    data = column.to_pylist()
                except Exception:
                    continue
                numeric = to_numeric(data)
                if numeric is not None:
                    arrays[name] = numeric
            if not arrays:
                continue
            if "episode_index" in table.column_names:
                ep_values = np.asarray(table.column("episode_index").to_pylist(), dtype=np.int64)
            else:
                ep_values = np.full(len(next(iter(arrays.values()))), -1 if file_episode is None else file_episode, dtype=np.int64)
            chunks.append((ep_values, arrays))
        if not chunks:
            raise FramesUnavailable("no readable parquet data found under data/")

        merged: dict[str, list[np.ndarray]] = defaultdict(list)
        ep_parts: list[np.ndarray] = []
        for ep_values, arrays in chunks:
            ep_parts.append(ep_values)
            for key, value in arrays.items():
                merged[key].append(value)
        ep_all = np.concatenate(ep_parts)
        columns: dict[str, np.ndarray] = {}
        for key, parts in merged.items():
            try:
                columns[key] = np.concatenate(parts, axis=0)
            except ValueError:
                continue
        if "episode_index" in columns:
            ep_all = columns.pop("episode_index").astype(np.int64).ravel()
        if features:
            keep = set(features)
            columns = {k: v for k, v in columns.items() if k in keep or k in ("timestamp", "frame_index")}
        order = np.argsort(ep_all, kind="stable")
        ep_all = ep_all[order]
        columns = {k: v[order] for k, v in columns.items()}
        if wanted is not None:
            mask = np.isin(ep_all, list(wanted))
            ep_all = ep_all[mask]
            columns = {k: v[mask] for k, v in columns.items()}
        # `ids` is per frame (same length as the columns), matching every other
        # adapter: consumers use it to slice contiguous same-episode runs.
        return ep_all.tolist(), columns


def load(root: Path, detection: Detection | None = None) -> LeRobotDataset:
    notes: list[str] = []
    if not (root / "meta" / "info.json").is_file():
        notes.append("meta/info.json is missing; format version and fps are unknown")
    info = _read_json_or_empty(root / "meta" / "info.json")
    version = str(info.get("codebase_version") or info.get("version") or "") or None
    fps = info.get("fps")
    fps = float(fps) if isinstance(fps, (int, float)) else None
    lengths, tasks, extra_notes = _episodes_from_meta(root)
    notes.extend(extra_notes)
    if not lengths:
        lengths = _episodes_from_data(root, notes)
        if lengths:
            notes.append("episode lengths derived from data files (metadata incomplete)")
    records = episode_records_from_lengths(lengths, fps=fps, tasks=tasks)
    all_tasks = _tasks_from_meta(root)
    if not all_tasks:
        seen: list[str] = []
        for record in records:
            if record.task and record.task not in seen:
                seen.append(record.task)
        all_tasks = seen
    summary = DatasetSummary(
        path=str(root),
        format="lerobot",
        version=version,
        fps=fps,
        robot_type=info.get("robot_type"),
        features=_features(info),
        episodes=records,
        tasks=all_tasks,
        notes=notes,
        meta={
            k: v
            for k, v in info.items()
            if k in ("total_episodes", "total_frames", "total_tasks", "chunks_size", "data_path", "video_path", "splits")
        },
        sources=[str(p.relative_to(root)) for p in _parquet_paths(root)[:50]],
    )
    return LeRobotDataset(root, summary)
