"""Structural validation.

``validate`` answers "is this dataset well-formed for its declared format?"
without loading frame data where possible. Checks are small functions taking a
:class:`DatasetObject` and yielding :class:`Issue` objects, so callers can run
a subset (``checks=["lerobot.spec"]``) or plug in their own.

Severities follow the model: ``error`` breaks most loaders, ``warn`` degrades
training or silently changes semantics, ``info`` is a convention hint.
"""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from .formats.base import DatasetObject, FramesUnavailable
from .model import DatasetSummary, Issue, Severity, summarize_issues
from .registry import UnsupportedFormat, detect_dataset, open_dataset

CheckFn = Callable[[DatasetObject], Iterable[Issue]]


# --------------------------------------------------------------------------
# dataset-level checks
# --------------------------------------------------------------------------
def check_non_empty(dataset: DatasetObject) -> Iterable[Issue]:
    summary = dataset.summary
    if not summary.episodes:
        yield Issue(
            code="dataset.empty",
            severity=Severity.ERROR,
            message="no episodes were found",
            hint="check that data/ or meta/episodes/ contains parquet files, or that the path points at the dataset root",
        )
    elif summary.total_frames == 0:
        yield Issue(
            code="dataset.no_frames",
            severity=Severity.ERROR,
            message="episodes exist but no frame counts could be determined",
            hint="metadata may be incomplete; try the format's native CLI to inspect the files",
        )
    return


def check_metadata(dataset: DatasetObject) -> Iterable[Issue]:
    summary = dataset.summary
    if not summary.fps:
        yield Issue(
            code="meta.missing_fps",
            severity=Severity.WARN,
            message="fps is not declared, so durations cannot be derived",
            hint="add fps to meta/info.json (LeRobot) or env_args.control_freq (robomimic)",
        )
    if not summary.robot_type:
        yield Issue(
            code="meta.missing_robot_type",
            severity=Severity.WARN,
            message="robot_type is not declared",
            hint="set robot_type in meta/info.json; downstream training configs usually key off it",
        )
    if not summary.features:
        yield Issue(
            code="meta.no_features",
            severity=Severity.ERROR,
            message="no feature descriptions were found",
            hint="declare features in meta/info.json or equivalent metadata",
        )
    return


def check_feature_vocabulary(dataset: DatasetObject) -> Iterable[Issue]:
    summary = dataset.summary
    kinds = {f.kind for f in summary.features}
    if summary.features and "state" not in kinds:
        yield Issue(
            code="feature.no_state",
            severity=Severity.WARN,
            message="no proprioceptive state feature was identified",
            hint="rename the column to observation.state (LeRobot) or state/qpos so policies can find it",
        )
    if summary.features and "action" not in kinds:
        yield Issue(
            code="feature.no_action",
            severity=Severity.WARN,
            message="no action feature was identified",
            hint="rename the command column to action (LeRobot) or actions (robomimic)",
        )
    seen: dict[str, int] = {}
    for feature in summary.features:
        if feature.dtype in ("", "unknown"):
            yield Issue(
                code="feature.unknown_dtype",
                severity=Severity.INFO,
                message=f"dtype of {feature.key!r} is unknown",
                scope="feature",
                feature=feature.key,
            )
        seen[feature.kind] = seen.get(feature.kind, 0) + 1
    if seen.get("image", 0) and not seen.get("state"):
        yield Issue(
            code="feature.vision_only",
            severity=Severity.INFO,
            message="dataset has camera streams but no state vector",
            hint="vision-only datasets cannot train state-conditioned policies",
        )
    return


def check_episodes(dataset: DatasetObject) -> Iterable[Issue]:
    episodes = dataset.summary.episodes
    indices = [e.index for e in episodes]
    if len(set(indices)) != len(indices):
        duplicates = sorted({i for i in indices if indices.count(i) > 1})
        yield Issue(
            code="episode.duplicate_index",
            severity=Severity.ERROR,
            message=f"duplicate episode indices: {duplicates[:10]}",
            detail={"count": len(duplicates)},
        )
    zero = [e.index for e in episodes if e.length == 0]
    if zero:
        yield Issue(
            code="episode.zero_length",
            severity=Severity.WARN,
            message=f"{len(zero)} episode(s) have zero frames",
            episode=zero[0],
            detail={"episodes": zero[:50]},
            hint="empty demos usually come from aborted recordings and should be dropped",
        )
    lengths = [e.length for e in episodes if e.length]
    if len(lengths) >= 5:
        median = statistics.median(lengths)
        short = [e.index for e in episodes if e.length and median and e.length < median * 0.1]
        if short:
            yield Issue(
                code="episode.length_outlier",
                severity=Severity.INFO,
                message=f"{len(short)} episode(s) are under 10% of the median length ({int(median)} frames)",
                episode=short[0],
                detail={"episodes": short[:50], "median_frames": int(median)},
            )
    missing_task = [e.index for e in episodes if not e.task]
    if episodes and len(missing_task) == len(episodes):
        yield Issue(
            code="episode.no_tasks",
            severity=Severity.INFO,
            message="no episode declares a task/instruction",
            hint="language-conditioned training needs a task string per episode",
        )
    elif missing_task:
        yield Issue(
            code="episode.missing_task",
            severity=Severity.INFO,
            message=f"{len(missing_task)} of {len(episodes)} episodes have no task string",
            episode=missing_task[0],
            detail={"episodes": missing_task[:50]},
        )
    return


def check_fps_consistency(dataset: DatasetObject) -> Iterable[Issue]:
    summary = dataset.summary
    if not summary.fps:
        return
    mismatched: list[int] = []
    for episode in summary.episodes:
        if not episode.length or not episode.duration_s:
            continue
        implied = episode.length / episode.duration_s
        if implied and abs(implied - summary.fps) / summary.fps > 0.05:
            mismatched.append(episode.index)
    if mismatched:
        yield Issue(
            code="episode.fps_mismatch",
            severity=Severity.WARN,
            message=f"{len(mismatched)} episode(s) report a duration inconsistent with fps={summary.fps}",
            episode=mismatched[0],
            detail={"episodes": mismatched[:50]},
            hint="either the declared fps or the episode duration is wrong; motion will play at the wrong speed",
        )
    return


# --------------------------------------------------------------------------
# format-specific checks
# --------------------------------------------------------------------------
def _lerobot_meta(dataset: DatasetObject) -> dict[str, Any]:
    path = dataset.root / "meta" / "info.json"
    if not path.is_file():
        return {}
    import json

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def check_lerobot_spec(dataset: DatasetObject) -> Iterable[Issue]:
    if dataset.summary.format != "lerobot":
        return
    root = dataset.root
    info = _lerobot_meta(dataset)
    if not info:
        yield Issue(
            code="lerobot.meta_missing",
            severity=Severity.ERROR,
            message="meta/info.json is required by the LeRobot format",
            hint="datasets must ship meta/info.json with codebase_version, fps, robot_type and features",
        )
        return
    version = str(info.get("codebase_version") or "")
    if version and not (version.startswith("v2") or version.startswith("v3")):
        yield Issue(
            code="lerobot.version_unknown",
            severity=Severity.INFO,
            message=f"codebase_version {version!r} is outside the v2.x/v3.x layouts this validator knows",
        )
    if not info.get("fps"):
        yield Issue(code="lerobot.fps_missing", severity=Severity.WARN, message="meta/info.json has no fps")
    total_episodes = info.get("total_episodes")
    if isinstance(total_episodes, int) and total_episodes != len(dataset.summary.episodes):
        yield Issue(
            code="lerobot.episode_count_mismatch",
            severity=Severity.WARN,
            message=f"meta/info.json declares {total_episodes} episodes but {len(dataset.summary.episodes)} were found",
            hint="re-run the LeRobot conversion or drop the stale meta/info.json counters",
        )
    total_frames = info.get("total_frames")
    found_frames = dataset.summary.total_frames
    if isinstance(total_frames, int) and found_frames and total_frames != found_frames:
        yield Issue(
            code="lerobot.frame_count_mismatch",
            severity=Severity.WARN,
            message=f"meta/info.json declares {total_frames} frames but {found_frames} were found in metadata",
        )
    tasks = root / "meta" / "tasks.jsonl"
    tasks_parquet = root / "meta" / "tasks.parquet"
    if not tasks.is_file() and not tasks_parquet.is_file() and not dataset.summary.tasks:
        yield Issue(
            code="lerobot.tasks_missing",
            severity=Severity.WARN,
            message="no meta/tasks.jsonl or meta/tasks.parquet",
            hint="v3 stores tasks in meta/tasks.parquet; v2 expects meta/tasks.jsonl",
        )
    if version.startswith("v3"):
        yield from _check_lerobot_v3_layout(dataset)
    data_dir = root / "data"
    if not data_dir.is_dir():
        yield Issue(code="lerobot.data_missing", severity=Severity.ERROR, message="data/ directory is missing")
    return


def _check_lerobot_v3_layout(dataset: DatasetObject) -> Iterable[Issue]:
    import re

    root = dataset.root
    meta_eps = root / "meta" / "episodes"
    if not meta_eps.is_dir():
        yield Issue(
            code="lerobot.v3_meta_episodes_missing",
            severity=Severity.ERROR,
            message="v3 layout requires meta/episodes/**, found none",
            hint="run the v2->v3 migration from lerobot (meta/episodes/chunk-000/file-000.parquet)",
        )
        return
    shards = sorted(p for p in meta_eps.rglob("*.parquet") if p.is_file())
    if not shards:
        yield Issue(
            code="lerobot.v3_meta_episodes_empty",
            severity=Severity.ERROR,
            message="meta/episodes/ contains no parquet shards",
        )
        return
    pattern = re.compile(r"chunk-(\d+)/file-(\d+)")
    seen: list[tuple[int, int]] = []
    for shard in shards:
        match = pattern.search(shard.as_posix())
        if match:
            seen.append((int(match.group(1)), int(match.group(2))))
    if seen:
        chunks = sorted({c for c, _ in seen})
        if chunks != list(range(chunks[0], chunks[0] + len(chunks))):
            yield Issue(
                code="lerobot.v3_chunk_gap",
                severity=Severity.WARN,
                message=f"meta/episodes chunk indices are not contiguous: {chunks}",
                detail={"chunks": chunks},
            )
        best: dict[int, list[int]] = {}
        for chunk, file_index in seen:
            best.setdefault(chunk, []).append(file_index)
        for chunk, files in best.items():
            ordered = sorted(files)
            if ordered != list(range(ordered[0], ordered[0] + len(ordered))):
                yield Issue(
                    code="lerobot.v3_file_gap",
                    severity=Severity.WARN,
                    message=f"meta/episodes/chunk-{chunk:03d} file indices are not contiguous: {ordered}",
                )
    videos = root / "videos"
    image_features = [f for f in dataset.summary.features if f.kind in ("image", "video")]
    if image_features and not videos.is_dir():
        yield Issue(
            code="lerobot.v3_videos_missing",
            severity=Severity.WARN,
            message=f"{len(image_features)} camera feature(s) declared but no videos/ directory",
            hint="v3 stores videos under videos/<camera>/chunk-XXX/file-XXX.mp4",
        )
    return


def check_parquet_readable(dataset: DatasetObject) -> Iterable[Issue]:
    """Open the first data parquet of each chunk to catch truncated files."""
    if dataset.summary.format != "lerobot":
        return
    try:
        import pyarrow.parquet as pq  # type: ignore
    except Exception:
        yield Issue(
            code="parquet.reader_missing",
            severity=Severity.INFO,
            message="pyarrow is not installed, so parquet footers were not verified",
            hint="install embodied-core[parquet]",
        )
        return
    data_dir = dataset.root / "data"
    if not data_dir.is_dir():
        return
    for directory in sorted(p for p in data_dir.iterdir() if p.is_dir())[:8]:
        for parquet in sorted(directory.glob("*.parquet"))[:1]:
            try:
                meta = pq.read_metadata(parquet)
            except Exception as exc:
                yield Issue(
                    code="parquet.unreadable",
                    severity=Severity.ERROR,
                    message=f"{parquet.relative_to(dataset.root)}: {exc}",
                    hint="the file is truncated or was written by an incompatible writer",
                )
                continue
            if meta.num_rows == 0:
                yield Issue(
                    code="parquet.empty",
                    severity=Severity.WARN,
                    message=f"{parquet.relative_to(dataset.root)} has no rows",
                )
    return


def check_hdf5_demos(dataset: DatasetObject) -> Iterable[Issue]:
    if dataset.summary.format != "hdf5":
        return
    try:
        import h5py  # type: ignore
    except Exception:
        return
    try:
        with h5py.File(dataset.root, "r") as handle:
            base = handle.get("data")
            if base is None:
                return
            names = [name for name in base]
            if not names:
                yield Issue(code="hdf5.no_demos", severity=Severity.ERROR, message="data/ contains no demo groups")
            empty = []
            for name in names:
                group = base[name]
                if not group.keys():
                    empty.append(name)
            if empty:
                yield Issue(
                    code="hdf5.empty_demos",
                    severity=Severity.WARN,
                    message=f"{len(empty)} demo group(s) are empty: {empty[:10]}",
                    detail={"demos": empty[:50]},
                )
    except OSError as exc:
        yield Issue(code="hdf5.unreadable", severity=Severity.ERROR, message=str(exc))
    return


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------
CHECKS: dict[str, CheckFn] = {
    "structure": check_non_empty,
    "metadata": check_metadata,
    "features": check_feature_vocabulary,
    "episodes": check_episodes,
    "fps": check_fps_consistency,
    "lerobot.spec": check_lerobot_spec,
    "parquet": check_parquet_readable,
    "hdf5.demos": check_hdf5_demos,
}

DEFAULT_CHECKS = [name for name in CHECKS if name not in {"parquet"}]


def run_checks(dataset: DatasetObject, checks: Sequence[str] | None = None) -> list[Issue]:
    names = list(checks or DEFAULT_CHECKS)
    unknown = [name for name in names if name not in CHECKS]
    if unknown:
        raise KeyError(f"unknown checks: {unknown}; available: {sorted(CHECKS)}")
    issues: list[Issue] = []
    for name in names:
        try:
            issues.extend(CHECKS[name](dataset) or [])
        except Exception as exc:  # a failing check is reported, not fatal
            issues.append(
                Issue(
                    code=f"check.failed:{name}",
                    severity=Severity.WARN,
                    message=f"check {name!r} raised {type(exc).__name__}: {exc}",
                )
            )
    return issues


def validate(
    path: str | Path,
    *,
    format: str | None = None,
    checks: Sequence[str] | None = None,
    include_frames: bool = False,
) -> dict[str, Any]:
    """Validate a dataset and return a JSON-ready report."""
    try:
        dataset = open_dataset(path, format)
    except UnsupportedFormat as exc:
        return {
            "path": str(path),
            "ok": False,
            "format": None,
            "issues": [Issue(code="format.unsupported", severity=Severity.ERROR, message=str(exc)).to_dict()],
            "counts": summarize_issues([Issue(code="format.unsupported", severity=Severity.ERROR, message=str(exc))]),
        }
    try:
        issues = run_checks(dataset, checks)
        if include_frames:
            issues.extend(_frame_checks(dataset))
        counts = summarize_issues(issues)
        report: dict[str, Any] = {
            "path": str(path),
            "format": dataset.summary.format,
            "ok": counts["errors"] == 0,
            "checks": list(checks or DEFAULT_CHECKS) + (["frames"] if include_frames else []),
            "summary": dataset.summary.to_dict(include_episodes=False),
            **counts,
            "issues": [issue.to_dict() for issue in issues],
        }
        detection = None
        try:
            detection = detect_dataset(path).to_dict()
        except Exception:
            pass
        if detection:
            report["detection"] = detection
        return report
    finally:
        dataset.close()


def _frame_checks(dataset: DatasetObject) -> list[Issue]:
    from .quality import basic_issue_scan

    try:
        _, columns = dataset.frames()
    except FramesUnavailable as exc:
        return [Issue(code="frames.unavailable", severity=Severity.INFO, message=exc.reason)]
    return basic_issue_scan(columns)
