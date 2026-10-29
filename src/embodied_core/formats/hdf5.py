"""HDF5 adapter — Robomimic / robosuite demos, RoboTurk and generic HDF5 stores.

Two layouts are recognised:

* **robomimic**: ``data/demo_N/{obs/*, actions, rewards, dones}`` with the
  environment description in the root ``env_args`` attribute.
* **flat/generic**: any group tree whose leaves are ``(T, D)`` numeric datasets;
  each top-level group that owns a time axis becomes one episode.

Image datasets are reported as features but excluded from tabular frames.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

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

HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"
IMAGE_KEYS = ("image", "rgb", "depth", "camera")


def _h5py() -> Any:
    try:
        import h5py  # type: ignore

        return h5py
    except Exception as exc:  # pragma: no cover - depends on environment
        raise FramesUnavailable(
            f"reading HDF5 requires h5py ({exc})",
            install="pip install 'embodied-core[hdf5]'",
        ) from exc


def _hdf5_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root] if root.suffix.lower() in (".h5", ".hdf5", ".hdf") else []
    return sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in (".h5", ".hdf5", ".hdf")
    )


def detect(root: Path) -> Detection | None:
    files = _hdf5_files(root)
    if not files:
        return None
    evidence: list[str] = []
    try:
        with files[0].open("rb") as handle:
            head = handle.read(8)
    except OSError:
        head = b""
    if head != HDF5_MAGIC:
        return None
    evidence.append(f"{files[0].name}: HDF5 signature")
    if len(files) > 1:
        evidence.append(f"{len(files)} hdf5 files")
    detail = None
    layout = "generic"
    try:
        h5py = _h5py()
        with h5py.File(files[0], "r") as handle:
            if "data" in handle and any(k.startswith("demo") for k in handle["data"]):
                layout = "robomimic"
                evidence.append("data/demo_* groups")
            elif "env_args" in handle.attrs:
                evidence.append("env_args attribute")
            detail = f"layout={layout}"
    except FramesUnavailable:
        detail = "h5py unavailable; structural detection only"
    except Exception as exc:  # pragma: no cover - corrupt file
        detail = f"unreadable: {exc}"
    return Detection("hdf5", version=None, confidence=0.9, evidence=evidence, detail=detail)


def _decode_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError:
            return None
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


def _collect_numeric(group: Any, prefix: str, out: dict[str, np.ndarray]) -> None:
    for key, item in group.items():
        name = f"{prefix}/{key}" if prefix else key
        if hasattr(item, "shape"):
            if any(hint in name.lower() for hint in IMAGE_KEYS) and item.ndim >= 3:
                continue
            if item.ndim == 0:
                continue
            try:
                data = item[()]
            except Exception:
                continue
            numeric = to_numeric(data)
            if numeric is not None and numeric.ndim == 2:
                out[name] = numeric
        else:
            _collect_numeric(item, name, out)


class HDF5Dataset(DatasetObject):
    def __init__(self, root: Path, summary: DatasetSummary, files: list[Path]) -> None:
        super().__init__(root, summary, capabilities={"frames", "episodes", "features", "groups"})
        self.files = files

    @staticmethod
    def optional_dependencies() -> dict[str, str]:
        return {"h5py": "HDF5 data access"}

    def _episode_groups(self) -> dict[int, tuple[Path, str]]:
        mapping: dict[int, tuple[Path, str]] = {}
        for index, record in enumerate(self.summary.episodes):
            if record.path:
                file_part, _, group_path = record.path.partition("::")
                mapping[index] = (self.root / file_part, group_path)
        return mapping

    def frames(
        self,
        episodes: Sequence[int] | None = None,
        features: Sequence[str] | None = None,
        *,
        max_frames: int | None = None,
    ) -> tuple[list[int], dict[str, np.ndarray]]:
        return collect_frame_chunks(
            self.iter_frames(episodes, features, max_frames=max_frames),
            max_frames=max_frames,
        )

    def iter_frames(
        self,
        episodes: Sequence[int] | None = None,
        features: Sequence[str] | None = None,
        *,
        chunk_frames: int | None = None,
        max_frames: int | None = None,
    ) -> Iterator[FrameChunk]:
        h5py = _h5py()
        groups = self._episode_groups()
        wanted = set(episodes) if episodes is not None else None
        keep = set(features) if features else None
        size = resolve_chunk_frames(chunk_frames)
        limit = resolve_max_frames(max_frames)
        total = 0
        seen: dict[int, int] = {}
        emitted = False
        for index, (path, group_path) in groups.items():
            if wanted is not None and index not in wanted:
                continue
            with h5py.File(path, "r") as handle:
                node = handle[group_path] if group_path else handle
                refs: dict[str, Any] = {}
                _numeric_datasets(node, "", refs, h5py)
                if keep is not None:
                    refs = {name: ref for name, ref in refs.items() if name in keep}
                if not refs:
                    continue
                length = len(next(iter(refs.values())))
                refs = {name: ref for name, ref in refs.items() if len(ref) == length}
                for start in range(0, length, size):
                    stop = min(length, start + size)
                    arrays: dict[str, np.ndarray] = {}
                    for name, ref in refs.items():
                        values = np.asarray(ref[start:stop])
                        if values.ndim == 1:
                            values = values.reshape(-1, 1)
                        arrays[name] = values
                    if not arrays:
                        continue
                    frames = stop - start
                    total += frames
                    if limit and total > limit:
                        raise FrameLimitExceeded(limit, total)
                    offsets = {index: seen.get(index, 0)}
                    seen[index] = seen.get(index, 0) + frames
                    emitted = True
                    yield FrameChunk([index] * frames, arrays, offsets)
        if not emitted:
            raise FramesUnavailable("no numeric (T, D) datasets found in the HDF5 groups")


def _numeric_datasets(group: Any, prefix: str, out: dict[str, Any], h5py: Any) -> None:
    for name, item in group.items():
        path = f"{prefix}/{name}".lstrip("/")
        if isinstance(item, h5py.Dataset):
            if item.ndim in (1, 2) and item.dtype.kind in "fiub":
                out[path] = item
        elif isinstance(item, h5py.Group):
            _numeric_datasets(item, path, out, h5py)


def load(root: Path, detection: Detection | None = None) -> HDF5Dataset:
    files = _hdf5_files(root)
    h5py = _h5py()
    records: list[EpisodeRecord] = []
    features: dict[str, FeatureSpec] = {}
    notes: list[str] = []
    robot_type: str | None = None
    fps = None
    meta: dict[str, Any] = {"files": [str(f.name) for f in files]}

    for file_path in files:
        rel = str(file_path.relative_to(root)) if root.is_dir() else file_path.name
        with h5py.File(file_path, "r") as handle:
            env_args = _decode_attr(handle.attrs.get("env_args"))
            if isinstance(env_args, dict):
                robot_type = robot_type or env_args.get("env_name")
                meta.setdefault("env_args", env_args)
                candidates = [env_args.get("control_freq")]
                kwargs = env_args.get("env_kwargs")
                if isinstance(kwargs, dict):
                    candidates.append(kwargs.get("control_freq"))
                for candidate in candidates:
                    if isinstance(candidate, (int, float)) and float(candidate) > 0:
                        fps = float(candidate)
                        break
            groups: list[tuple[str, Any]] = []
            if "data" in handle and any(k.startswith("demo") for k in handle["data"]):
                for name in sorted(handle["data"], key=lambda s: (len(s), s)):
                    groups.append((f"data/{name}", handle["data"][name]))
            else:
                for name in sorted(handle):
                    item = handle[name]
                    if hasattr(item, "keys") and len(item.keys()) > 0:
                        groups.append((name, item))
                if not groups:
                    groups.append(("", handle))
            for group_path, group in groups:
                columns: dict[str, np.ndarray] = {}
                _collect_numeric(group, "", columns)
                length = max((len(v) for v in columns.values()), default=None)
                task = None
                try:
                    if "task" in group.attrs:
                        task = str(_decode_attr(group.attrs.get("task")))
                except Exception:
                    task = None
                records.append(
                    EpisodeRecord(
                        index=len(records),
                        length=length,
                        task=task,
                        fps=fps,
                        duration_s=(length / fps) if (length and fps) else None,
                        path=f"{rel}::{group_path}",
                    )
                )
                for name, data in columns.items():
                    # Feature keys are schema-level (``obs/object``), not
                    # per-demo (``data/demo_0/obs/object``): every demo shares
                    # the same schema, and duplicated keys would triple the
                    # feature list of a 100-demo file.
                    if name not in features:
                        features[name] = FeatureSpec(
                            key=name,
                            dtype=str(data.dtype),
                            shape=tuple(data.shape[1:]),
                            kind=guess_kind(name),
                        )
    if not records:
        notes.append("no episode groups found")
    summary = DatasetSummary(
        path=str(root),
        format="hdf5",
        version=None,
        fps=fps,
        robot_type=robot_type,
        features=sorted(features.values(), key=lambda f: f.key),
        episodes=records,
        tasks=sorted({r.task for r in records if r.task}),
        notes=notes,
        meta=meta,
        sources=[str(f.relative_to(root)) if root.is_dir() else f.name for f in files[:50]],
    )
    return HDF5Dataset(root, summary, files)
