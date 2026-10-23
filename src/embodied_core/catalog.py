"""Dataset catalog: index a tree of datasets into SQLite and query it.

The catalog stores only metadata (never frames), so it stays small enough to
ship in a repo or keep next to a shared drive. Re-indexing is incremental: a
dataset is refreshed when its marker file's mtime/size changes.

Typical use::

    embodied catalog index ./datasets --db ./catalog.db
    embodied catalog search --db ./catalog.db --format lerobot --min-episodes 20
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

from .model import DatasetSummary, to_jsonable
from .registry import UnsupportedFormat, detect_dataset, open_dataset

SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT UNIQUE NOT NULL,
    format TEXT NOT NULL,
    version TEXT,
    episodes INTEGER,
    frames INTEGER,
    duration_s REAL,
    fps REAL,
    robot_type TEXT,
    tasks TEXT,
    features TEXT,
    notes TEXT,
    meta TEXT,
    score INTEGER,
    marker TEXT,
    mtime REAL,
    size INTEGER,
    indexed_at REAL
);
CREATE INDEX IF NOT EXISTS idx_datasets_format ON datasets(format);
CREATE INDEX IF NOT EXISTS idx_datasets_robot ON datasets(robot_type);
"""

MARKER_NAMES = (
    "meta/info.json",
    "dataset_info.json",
    "dataset.json",
    "episodes.jsonl",
)


FILE_FORMAT_SUFFIXES = (".hdf5", ".h5", ".mcap", ".bag", ".db3")


def _candidate_paths(root: Path, max_depth: int = 4) -> Iterable[Path]:
    """Yield dataset roots: directories with a layout marker, or single files."""
    root = root.expanduser().resolve()
    if root.is_file():
        yield root
        return
    if not root.is_dir():
        return
    base_depth = len(root.parts)
    for current, dirs, files in os.walk(root):
        path = Path(current)
        depth = len(path.parts) - base_depth
        if depth > max_depth:
            dirs[:] = []
            continue
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        has_marker = any((path / marker).is_file() for marker in MARKER_NAMES)
        container_files = [f for f in files if f.lower().endswith(FILE_FORMAT_SUFFIXES)]
        if has_marker:
            yield path
            dirs[:] = []  # a dataset layout owns its subtree
            continue
        if container_files:
            if len(container_files) == 1:
                yield path / container_files[0]
            else:
                for name in sorted(container_files):
                    yield path / name
            dirs[:] = []
            continue
        if any(f.endswith(".tfrecord") for f in files) and any(
            (path / marker).is_file() for marker in ("dataset_info.json", "features.json")
        ):
            yield path
            dirs[:] = []


def _marker_stat(path: Path) -> tuple[str, float, int]:
    """Stamp a dataset with its newest metadata/content file.

    A single well-known file is not enough: appending episodes touches
    ``episodes.jsonl`` while ``dataset.json`` stays untouched, which used to make
    the incremental scan skip real changes. We therefore take the newest file
    among the shallow metadata surface (dataset root and ``meta/``), which stays
    cheap even for very large datasets.
    """
    marker = path
    if path.is_dir():
        candidates: list[Path] = [p for p in path.iterdir() if p.is_file()]
        meta_dir = path / "meta"
        if meta_dir.is_dir():
            candidates.extend(p for p in meta_dir.iterdir() if p.is_file())
        if not candidates:
            newest = max((p for p in path.rglob("*") if p.is_file()), key=lambda p: p.stat().st_mtime, default=None)
            candidates = [newest] if newest is not None else [path]
        marker = max(candidates, key=lambda p: p.stat().st_mtime)
    try:
        stat = marker.stat()
        return str(marker), stat.st_mtime, stat.st_size
    except OSError:
        return str(marker), 0.0, 0


def connect(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def _row_from_summary(summary: DatasetSummary, marker: str, mtime: float, size: int, score: int | None) -> dict[str, Any]:
    return {
        "path": str(Path(summary.path).resolve()),
        "format": summary.format,
        "version": summary.version,
        "episodes": summary.episode_count,
        "frames": summary.total_frames,
        "duration_s": summary.total_duration_s,
        "fps": summary.fps,
        "robot_type": summary.robot_type,
        "tasks": json.dumps(summary.tasks, ensure_ascii=False),
        "features": json.dumps([f.to_dict() for f in summary.features], ensure_ascii=False),
        "notes": json.dumps(summary.notes, ensure_ascii=False),
        "meta": json.dumps(to_jsonable(summary.meta), ensure_ascii=False),
        "score": score,
        "marker": marker,
        "mtime": mtime,
        "size": size,
        "indexed_at": time.time(),
    }


def index(
    root: str | Path,
    db_path: str | Path,
    *,
    max_depth: int = 4,
    fmt: str | None = None,
    force: bool = False,
    validate: bool = False,
) -> dict[str, Any]:
    """Scan ``root`` and upsert every dataset found into ``db_path``."""
    conn = connect(db_path)
    added = updated = skipped = failed = 0
    failures: list[dict[str, str]] = []
    try:
        for candidate in _candidate_paths(Path(root), max_depth=max_depth):
            marker, mtime, size = _marker_stat(candidate)
            resolved = str(candidate.resolve())
            existing = conn.execute("SELECT mtime, size FROM datasets WHERE path = ?", (resolved,)).fetchone()
            if existing and not force and abs(float(existing["mtime"]) - mtime) < 1e-6 and int(existing["size"]) == size:
                skipped += 1
                continue
            try:
                dataset = open_dataset(candidate, fmt)
            except UnsupportedFormat as exc:
                failed += 1
                failures.append({"path": resolved, "error": str(exc)})
                continue
            except FileNotFoundError as exc:
                failed += 1
                failures.append({"path": resolved, "error": f"missing: {exc}"})
                continue
            except Exception as exc:  # a broken dataset must not stop the scan
                failed += 1
                failures.append({"path": resolved, "error": f"{type(exc).__name__}: {exc}"})
                continue
            score = None
            summary = dataset.summary
            try:
                if validate:
                    from .model import summarize_issues
                    from .validate import run_checks

                    score = summarize_issues(run_checks(dataset))["score"]
            except Exception as exc:
                failures.append({"path": resolved, "error": f"validation: {type(exc).__name__}: {exc}"})
            finally:
                dataset.close()
            row = _row_from_summary(summary, marker, mtime, size, score)
            columns = ", ".join(row)
            placeholders = ", ".join(f":{key}" for key in row)
            conn.execute(
                f"INSERT INTO datasets ({columns}) VALUES ({placeholders}) "
                f"ON CONFLICT(path) DO UPDATE SET " + ", ".join(f"{key}=excluded.{key}" for key in row if key != "path"),
                row,
            )
            if existing:
                updated += 1
            else:
                added += 1
        conn.commit()
        total = conn.execute("SELECT COUNT(*) FROM datasets").fetchone()[0]
    finally:
        conn.close()
    return {
        "db": str(Path(db_path).resolve()),
        "root": str(Path(root).resolve()),
        "added": added,
        "updated": updated,
        "skipped": skipped,
        "failed": failed,
        "total": total,
        "failures": failures[:50],
    }


def search(
    db_path: str | Path,
    *,
    text: str | None = None,
    fmt: str | None = None,
    robot_type: str | None = None,
    min_episodes: int | None = None,
    max_episodes: int | None = None,
    min_duration: float | None = None,
    min_score: int | None = None,
    limit: int = 50,
    order_by: str = "indexed_at",
    descending: bool = True,
) -> list[dict[str, Any]]:
    if not Path(db_path).expanduser().exists():
        raise FileNotFoundError(db_path)
    clauses: list[str] = []
    params: list[Any] = []
    if text:
        like = f"%{text}%"
        clauses.append("(path LIKE ? OR tasks LIKE ? OR robot_type LIKE ? OR format LIKE ?)")
        params.extend([like, like, like, like])
    for column, value in (("format", fmt), ("robot_type", robot_type)):
        if value:
            clauses.append(f"{column} = ?")
            params.append(value)
    if min_episodes is not None:
        clauses.append("episodes >= ?")
        params.append(min_episodes)
    if max_episodes is not None:
        clauses.append("episodes <= ?")
        params.append(max_episodes)
    if min_duration is not None:
        clauses.append("duration_s >= ?")
        params.append(min_duration)
    if min_score is not None:
        clauses.append("score >= ?")
        params.append(min_score)
    allowed = {"indexed_at", "episodes", "frames", "duration_s", "score", "path", "mtime"}
    column = order_by if order_by in allowed else "indexed_at"
    direction = "DESC" if descending else "ASC"
    sql = "SELECT * FROM datasets"
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += f" ORDER BY {column} {direction} LIMIT ?"
    params.append(int(limit))
    conn = connect(db_path)
    try:
        rows = [dict(row) for row in conn.execute(sql, params)]
    finally:
        conn.close()
    for row in rows:
        for key in ("tasks", "features", "notes", "meta"):
            try:
                row[key] = json.loads(row[key]) if row.get(key) else None
            except (TypeError, json.JSONDecodeError):
                pass
    return rows


def stats(db_path: str | Path) -> dict[str, Any]:
    if not Path(db_path).expanduser().exists():
        raise FileNotFoundError(db_path)
    conn = connect(db_path)
    try:
        total = conn.execute("SELECT COUNT(*) FROM datasets").fetchone()[0]
        by_format = {row["format"]: row["n"] for row in conn.execute("SELECT format, COUNT(*) AS n FROM datasets GROUP BY format ORDER BY n DESC")}
        episodes = conn.execute("SELECT SUM(episodes), SUM(frames), SUM(duration_s) FROM datasets").fetchone()
        robots = [row["robot_type"] for row in conn.execute("SELECT DISTINCT robot_type FROM datasets WHERE robot_type IS NOT NULL")]
        last = conn.execute("SELECT MAX(indexed_at) FROM datasets").fetchone()[0]
    finally:
        conn.close()
    return {
        "total_datasets": total,
        "by_format": by_format,
        "total_episodes": episodes[0] or 0,
        "total_frames": episodes[1] or 0,
        "total_duration_s": round(float(episodes[2] or 0), 2),
        "robot_types": sorted(robots),
        "last_indexed": last,
    }


def remove(db_path: str | Path, path: str | Path) -> bool:
    conn = connect(db_path)
    try:
        cursor = conn.execute("DELETE FROM datasets WHERE path = ?", (str(Path(path).expanduser().resolve()),))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def export_manifest(db_path: str | Path, *, limit: int = 1000) -> dict[str, Any]:
    rows = search(db_path, limit=limit)
    return {
        "count": len(rows),
        "datasets": [
            {
                "path": row["path"],
                "format": row["format"],
                "version": row["version"],
                "episodes": row["episodes"],
                "frames": row["frames"],
                "duration_s": row["duration_s"],
                "fps": row["fps"],
                "robot_type": row["robot_type"],
                "tasks": row["tasks"],
                "score": row["score"],
            }
            for row in rows
        ],
    }
