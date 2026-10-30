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
import os
import shutil
import tempfile
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


def compute_normalization_stats(columns: dict[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    """Per-feature normalization/observation statistics for LeRobot v3."""

    stats: dict[str, dict[str, Any]] = {}
    for key, values in columns.items():
        array = np.asarray(values, dtype=np.float64)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        if array.ndim != 2:
            array = array.reshape(array.shape[0], -1)
        finite = np.where(np.isfinite(array), array, np.nan)
        with np.errstate(all="ignore"):
            mean = np.nanmean(finite, axis=0)
            std = np.nanstd(finite, axis=0)
            minimum = np.nanmin(finite, axis=0)
            maximum = np.nanmax(finite, axis=0)
            q01 = np.nanpercentile(finite, 1, axis=0)
            q99 = np.nanpercentile(finite, 99, axis=0)
        stats[key] = {
            "count": int(np.isfinite(array).any(axis=1).sum()),
            "mean": _clean_numeric(mean),
            "std": _clean_numeric(std),
            "min": _clean_numeric(minimum),
            "max": _clean_numeric(maximum),
            "q01": _clean_numeric(q01),
            "q99": _clean_numeric(q99),
        }
    return stats


def _clean_numeric(values: np.ndarray) -> list[float | None]:
    out: list[float | None] = []
    for value in np.asarray(values).reshape(-1):
        item = float(value)
        out.append(None if not np.isfinite(item) else round(item, 8))
    return out


def _split_ranges(
    episodes: Sequence[int],
    *,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
) -> tuple[dict[str, str], dict[int, str]]:
    count = len(episodes)
    if count == 0:
        return {}, {}
    ratios = [max(0.0, float(train_ratio)), max(0.0, float(validation_ratio)), max(0.0, float(test_ratio))]
    if sum(ratios) <= 0:
        ratios[0] = 1.0
    total = sum(ratios)
    ratios = [value / total for value in ratios]
    train = int(round(count * ratios[0]))
    validation = int(round(count * ratios[1]))
    if ratios[0] > 0:
        train = max(1, train)
    if ratios[1] > 0 and count - train > 0:
        validation = max(1, min(count - train, validation))
    if train + validation > count:
        validation = max(0, count - train)
    test = count - train - validation
    if ratios[2] > 0 and test <= 0 and count > 1:
        test = 1
        if train > validation and train > 1:
            train -= 1
        elif validation > 1:
            validation -= 1
    cursor = 0
    ranges: dict[str, str] = {}
    labels: dict[int, str] = {}
    for name, size in (("train", train), ("validation", validation), ("test", test)):
        if size <= 0:
            continue
        ranges[name] = f"{cursor}:{cursor + size}"
        for episode in episodes[cursor : cursor + size]:
            labels[int(episode)] = name
        cursor += size
    return ranges, labels


def to_lerobot_v3(
    dataset,
    destination: Path,
    *,
    overwrite: bool = False,
    repo_id: str | None = None,
    episodes: Sequence[int] | None = None,
    train_ratio: float = 0.9,
    validation_ratio: float = 0.0,
    test_ratio: float = 0.0,
) -> dict[str, Any]:
    """Write a LeRobot v3.0 training dataset with stats and splits."""

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on the environment
        raise ConversionError("LeRobot v3 export needs pyarrow: pip install 'embodied-core[parquet]'") from exc

    ids, columns, episode_ids = _episode_columns(dataset, episodes)
    if not columns:
        raise ConversionError("nothing to write")
    destination = destination.expanduser()
    if destination.exists() and not overwrite:
        raise ConversionError(f"{destination} exists; pass overwrite=True")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.v3-", dir=str(destination.parent)))
    summary = dataset.summary
    fps = float(summary.fps or 30.0)
    tasks: list[str] = list(summary.tasks)
    tasks_by_episode = {e.index: e.task for e in summary.episodes}
    for episode in episode_ids:
        task = tasks_by_episode.get(episode)
        if task and task not in tasks:
            tasks.append(task)
    splits, split_labels = _split_ranges(
        episode_ids,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
    )
    try:
        data_dir = staging / "data" / "chunk-000"
        meta_dir = staging / "meta"
        episode_dir = meta_dir / "episodes" / "chunk-000"
        data_dir.mkdir(parents=True, exist_ok=True)
        episode_dir.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, Any] = {key: [] for key in columns}
        timestamps: list[float] = []
        frame_indices: list[int] = []
        episode_indices: list[int] = []
        global_indices: list[int] = []
        task_indices: list[int] = []
        episode_rows: list[dict[str, Any]] = []
        cursor = 0
        for episode in episode_ids:
            positions = [position for position, value in enumerate(ids) if value == episode]
            if not positions:
                continue
            length = len(positions)
            task = tasks_by_episode.get(episode)
            task_index = tasks.index(task) if task in tasks else 0
            for key, values in columns.items():
                block = values[positions]
                if block.ndim == 1:
                    block = block.reshape(-1, 1)
                elif block.ndim > 2:
                    block = block.reshape(block.shape[0], -1)
                arrays[key].extend(np.asarray(block, dtype=np.float32).tolist())
            timestamps.extend((np.arange(length) / fps).astype(np.float32).tolist())
            frame_indices.extend(range(length))
            episode_indices.extend([int(episode)] * length)
            global_indices.extend(range(cursor, cursor + length))
            task_indices.extend([int(task_index)] * length)
            episode_rows.append(
                {
                    "episode_index": int(episode),
                    "tasks": [task] if task else [],
                    "length": length,
                    "split": split_labels.get(int(episode)),
                    "data/chunk_index": 0,
                    "data/file_index": 0,
                    "dataset_from_index": cursor,
                    "dataset_to_index": cursor + length,
                }
            )
            cursor += length
        if not episode_rows:
            raise ConversionError("no episodes with frames to write")
        feature_arrays: dict[str, Any] = {}
        for key, values in arrays.items():
            dimensions = len(values[0]) if values and isinstance(values[0], list) else 1
            feature_arrays[key] = pa.array(values, type=pa.list_(pa.float32(), dimensions))
        feature_arrays["timestamp"] = pa.array(timestamps, type=pa.float32())
        feature_arrays["frame_index"] = pa.array(frame_indices, type=pa.int64())
        feature_arrays["episode_index"] = pa.array(episode_indices, type=pa.int64())
        feature_arrays["index"] = pa.array(global_indices, type=pa.int64())
        feature_arrays["task_index"] = pa.array(task_indices, type=pa.int64())
        pq.write_table(pa.table(feature_arrays), data_dir / "file-000.parquet", compression="snappy")
        pq.write_table(pa.Table.from_pylist(episode_rows), episode_dir / "file-000.parquet", compression="snappy")
        pq.write_table(
            pa.table({"task_index": list(range(len(tasks))), "task": tasks}),
            meta_dir / "tasks.parquet",
            compression="snappy",
        )
        stats = compute_normalization_stats({key: values for key, values in columns.items()})
        write_json(meta_dir / "stats.json", stats)
        features: dict[str, Any] = {
            key: {
                "dtype": "float32",
                "shape": [int(values.shape[1]) if values.ndim > 1 else 1],
                "names": None,
            }
            for key, values in columns.items()
        }
        info = {
            "codebase_version": "v3.0",
            "robot_type": summary.robot_type or "unknown",
            "total_episodes": len(episode_rows),
            "total_frames": cursor,
            "total_tasks": len(tasks),
            "total_videos": 0,
            "total_chunks": 1,
            "chunks_size": 1000,
            "fps": fps,
            "splits": splits,
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            "features": features,
        }
        if repo_id:
            info["repo_id"] = repo_id
        write_json(meta_dir / "info.json", info)
        if destination.exists():
            if destination.is_dir():
                shutil.rmtree(destination)
            else:
                destination.unlink()
        os.replace(staging, destination)
        try:
            from .validate import validate

            validation = validate(destination, checks=["structure", "lerobot.spec"])
        except Exception as exc:
            validation = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "target": "lerobot(v3.0)",
        "destination": str(destination.resolve()),
        "episodes": len(episode_rows),
        "frames": cursor,
        "features": list(features),
        "splits": splits,
        "stats": str((destination / "meta" / "stats.json").resolve()),
        "validation": validation,
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
    train_ratio: float = 0.9,
    validation_ratio: float = 0.0,
    test_ratio: float = 0.0,
) -> dict[str, Any]:
    """Convert ``source`` into ``destination`` in the requested ``target`` layout."""
    dataset = open_dataset(source, format)
    try:
        if target in ("jsonl", "embodied-jsonl", "canonical"):
            return to_jsonl(dataset, Path(destination), episodes=episodes, overwrite=overwrite)
        if target in ("lerobot", "lerobot-v2", "lerobot-v2.1"):
            return to_lerobot(dataset, Path(destination), overwrite=overwrite)
        if target in ("lerobot-v3", "lerobot-v3.0", "v3"):
            return to_lerobot_v3(
                dataset,
                Path(destination),
                overwrite=overwrite,
                episodes=episodes,
                train_ratio=train_ratio,
                validation_ratio=validation_ratio,
                test_ratio=test_ratio,
            )
        raise ConversionError(f"unknown target {target!r}; supported: jsonl, lerobot, lerobot-v3")
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
