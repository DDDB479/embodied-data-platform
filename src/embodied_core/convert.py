"""Conversions between on-disk layouts.

Two targets are implemented, both numeric-first:

* ``jsonl``    — the studio's canonical interchange format (no dependencies),
                 good for fixtures, diffs and moving slices between tools.
* ``lerobot``  — a LeRobot v2.1 dataset (``data/chunk-000/episode_*.parquet``
                 plus ``meta/info.json`` / ``meta/episodes.jsonl`` /
                 ``meta/tasks.jsonl``). Needs pyarrow. Camera streams are not
                 re-encoded: video features are dropped with an explicit note,
                 because transcoding mp4 files is not a metadata operation.

Structural-only sources (MCAP, rosbag, RLDS) raise :class:`ConversionError`
explaining which reader they need, rather than writing an empty dataset.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .formats.base import FramesUnavailable
from .model import to_jsonable
from .registry import detect_dataset, open_dataset
from .scaffold import write_json, write_jsonl


class ConversionError(RuntimeError):
    pass


def _episode_columns(dataset, episodes: Sequence[int] | None) -> tuple[list[int], dict[str, np.ndarray], list[int]]:
    ids, columns = dataset.frames(episodes)
    if not columns:
        raise ConversionError("the source exposes no numeric columns to convert")
    return list(ids), columns, sorted(set(ids))


def _rows_for_episode(ids: list[int], columns: dict[str, np.ndarray], episode: int) -> list[dict[str, Any]]:
    positions = [i for i, value in enumerate(ids) if value == episode]
    rows: list[dict[str, Any]] = []
    for position in positions:
        row: dict[str, Any] = {}
        for key, values in columns.items():
            row[key] = values[position].tolist() if values.ndim > 1 else float(values[position])
        rows.append(row)
    return rows


def to_jsonl(dataset, destination: Path, *, episodes: Sequence[int] | None = None, overwrite: bool = False) -> dict[str, Any]:
    ids, columns, episode_ids = _episode_columns(dataset, episodes)
    destination = destination.expanduser()
    if destination.exists() and not overwrite:
        raise ConversionError(f"{destination} exists; pass overwrite=True")
    destination.mkdir(parents=True, exist_ok=True)
    summary = dataset.summary
    tasks_by_episode = {e.index: e.task for e in summary.episodes}
    header = {
        "format": "embodied-jsonl",
        "version": "1",
        "fps": summary.fps,
        "robot_type": summary.robot_type,
        "tasks": summary.tasks,
        "features": {f.key: {"dtype": f.dtype, "shape": list(f.shape), "kind": f.kind} for f in summary.features if f.key in columns},
        "derived_from": {"path": summary.path, "format": summary.format},
    }
    write_json(destination / "dataset.json", header)
    lines = []
    for episode in episode_ids:
        rows = _rows_for_episode(ids, columns, episode)
        lines.append(
            {
                "episode_index": episode,
                "task": tasks_by_episode.get(episode),
                "fps": summary.fps,
                "frames": rows,
            }
        )
    write_jsonl(destination / "episodes.jsonl", lines)
    return {
        "target": "embodied-jsonl",
        "destination": str(destination.resolve()),
        "episodes": len(episode_ids),
        "frames": len(ids),
        "features": list(columns),
        "dropped_features": [f.key for f in summary.features if f.key not in columns],
    }


def to_lerobot(dataset, destination: Path, *, overwrite: bool = False, repo_id: str | None = None) -> dict[str, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on the environment
        raise ConversionError("LeRobot export needs pyarrow: pip install 'embodied-core[parquet]'") from exc

    ids, columns, episode_ids = _episode_columns(dataset, None)
    destination = destination.expanduser()
    if destination.exists() and not overwrite:
        raise ConversionError(f"{destination} exists; pass overwrite=True")
    data_dir = destination / "data" / "chunk-000"
    meta_dir = destination / "meta"
    data_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    summary = dataset.summary
    fps = float(summary.fps or 30.0)
    tasks: list[str] = list(summary.tasks)
    tasks_by_episode = {e.index: e.task for e in summary.episodes}
    for episode in episode_ids:
        task = tasks_by_episode.get(episode)
        if task and task not in tasks:
            tasks.append(task)

    episode_rows: list[dict[str, Any]] = []
    frame_cursor = 0
    total_written = 0
    for episode in episode_ids:
        rows = _rows_for_episode(ids, columns, episode)
        if not rows:
            continue
        task = tasks_by_episode.get(episode)
        task_index = tasks.index(task) if task in tasks else 0
        arrays: dict[str, Any] = {}
        for key, values in columns.items():
            block = values[[i for i, value in enumerate(ids) if value == episode]]
            if block.ndim == 1:
                block = block.reshape(-1, 1)
            elif block.ndim > 2:
                block = block.reshape(block.shape[0], -1)
            arrays[key] = pa.array([row.tolist() for row in block], type=pa.list_(pa.float32()))
        length = len(rows)
        arrays["timestamp"] = pa.array((np.arange(length) / fps).astype(np.float32), type=pa.float32())
        arrays["frame_index"] = pa.array(np.arange(length, dtype=np.int64), type=pa.int64())
        arrays["episode_index"] = pa.array(np.full(length, episode, dtype=np.int64), type=pa.int64())
        arrays["index"] = pa.array(np.arange(frame_cursor, frame_cursor + length, dtype=np.int64), type=pa.int64())
        arrays["task_index"] = pa.array(np.full(length, task_index, dtype=np.int64), type=pa.int64())
        table = pa.table(arrays)
        target = data_dir / f"episode_{episode:06d}.parquet"
        pq.write_table(table, target, compression="snappy")
        episode_rows.append({"episode_index": episode, "tasks": [task] if task else [], "length": length})
        frame_cursor += length
        total_written += length

    features: dict[str, Any] = {
        key: {"dtype": "float32", "shape": [int(values.shape[1]) if values.ndim > 1 else 1], "names": None}
        for key, values in columns.items()
    }
    if not features:
        raise ConversionError("nothing to write")
    info = {
        "codebase_version": "v2.1",
        "robot_type": summary.robot_type or "unknown",
        "total_episodes": len(episode_rows),
        "total_frames": total_written,
        "total_tasks": len(tasks),
        "total_videos": 0,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": f"0:{len(episode_rows)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": None,
        "features": features,
    }
    if repo_id:
        info["repo_id"] = repo_id
    write_json(meta_dir / "info.json", info)
    write_jsonl(
        meta_dir / "episodes.jsonl",
        [{"episode_index": row["episode_index"], "tasks": row["tasks"], "length": row["length"]} for row in episode_rows],
    )
    write_jsonl(meta_dir / "tasks.jsonl", [{"task_index": i, "task": task} for i, task in enumerate(tasks)])
    return {
        "target": "lerobot(v2.1)",
        "destination": str(destination.resolve()),
        "episodes": len(episode_rows),
        "frames": total_written,
        "features": list(features),
        "dropped_features": [f.key for f in summary.features if f.key not in columns],
        "note": "numeric features only; camera streams were not re-encoded",
    }


def convert(
    source: str | Path,
    destination: str | Path,
    *,
    target: str = "jsonl",
    overwrite: bool = False,
    episodes: Sequence[int] | None = None,
    format: str | None = None,
) -> dict[str, Any]:
    """Convert ``source`` into ``destination`` in the requested ``target`` layout."""
    dataset = open_dataset(source, format)
    try:
        if target in ("jsonl", "embodied-jsonl", "canonical"):
            return to_jsonl(dataset, Path(destination), episodes=episodes, overwrite=overwrite)
        if target in ("lerobot", "lerobot-v2", "lerobot-v2.1"):
            return to_lerobot(dataset, Path(destination), overwrite=overwrite)
        raise ConversionError(f"unknown target {target!r}; supported: jsonl, lerobot")
    except FramesUnavailable as exc:
        detection = None
        try:
            detection = detect_dataset(source)
        except Exception:
            pass
        hint = f" (detected {detection.format})" if detection else ""
        raise ConversionError(f"{exc.reason}{hint}") from exc
    finally:
        dataset.close()


def copy_dataset(source: str | Path, destination: str | Path, *, overwrite: bool = False) -> dict[str, Any]:
    """Filesystem copy with a manifest, for archiving a dataset unchanged."""
    src = Path(source).expanduser()
    dst = Path(destination).expanduser()
    if dst.exists() and not overwrite:
        raise ConversionError(f"{dst} exists; pass overwrite=True")
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=overwrite)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    dataset = open_dataset(dst)
    try:
        manifest = {
            "source": str(src.resolve()),
            "destination": str(dst.resolve()),
            "summary": to_jsonable(dataset.summary.to_dict(include_episodes=False)),
        }
    finally:
        dataset.close()
    write_json(dst.parent / f"{dst.name}.manifest.json" if dst.is_file() else dst / "manifest.json", manifest)
    return manifest
