"""Dataset catalog with immutable versions, runs and lineage (schema v2).

The catalog stores metadata only. Datasets are logical identities keyed by
path; every distinct file fingerprint gets an immutable version. Index and
validation runs are recorded, and a new version points at the version it
supersedes through the lineage table.

Typical use::

    embodied catalog index ./datasets --db ./catalog.db
    embodied catalog search --db ./catalog.db --format lerobot --min-episodes 20
    embodied catalog versions --db ./catalog.db --path ./datasets/night_run
    embodied catalog lineage --db ./catalog.db --path ./datasets/night_run
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Iterable, Sequence

from .cache import dataset_fingerprint
from .model import DatasetSummary, to_jsonable
from .registry import UnsupportedFormat, detect_dataset, open_dataset

SCHEMA_VERSION = 2

SCHEMA = """
CREATE TABLE IF NOT EXISTS catalog_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS datasets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT UNIQUE NOT NULL,
    current_version_id INTEGER,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_key TEXT UNIQUE NOT NULL,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at REAL NOT NULL,
    finished_at REAL,
    root TEXT,
    config TEXT,
    metrics TEXT,
    error TEXT
);
CREATE TABLE IF NOT EXISTS dataset_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id INTEGER NOT NULL,
    run_id INTEGER,
    fingerprint TEXT NOT NULL,
    format TEXT NOT NULL,
    format_version TEXT,
    episodes INTEGER,
    frames INTEGER,
    duration_s REAL,
    fps REAL,
    robot_type TEXT,
    tasks TEXT,
    features TEXT,
    notes TEXT,
    meta TEXT,
    marker TEXT,
    mtime REAL,
    size INTEGER,
    score INTEGER,
    indexed_at REAL NOT NULL,
    UNIQUE(dataset_id, fingerprint),
    FOREIGN KEY(dataset_id) REFERENCES datasets(id) ON DELETE CASCADE,
    FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE SET NULL
);
CREATE TABLE IF NOT EXISTS lineage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    parent_version_id INTEGER NOT NULL,
    child_version_id INTEGER NOT NULL,
    relation TEXT NOT NULL,
    run_id INTEGER,
    detail TEXT,
    created_at REAL NOT NULL,
    UNIQUE(parent_version_id, child_version_id, relation),
    FOREIGN KEY(parent_version_id) REFERENCES dataset_versions(id) ON DELETE CASCADE,
    FOREIGN KEY(child_version_id) REFERENCES dataset_versions(id) ON DELETE CASCADE,
    FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_datasets_updated ON datasets(updated_at);
CREATE INDEX IF NOT EXISTS idx_versions_dataset ON dataset_versions(dataset_id, indexed_at);
CREATE INDEX IF NOT EXISTS idx_versions_fingerprint ON dataset_versions(fingerprint);
CREATE INDEX IF NOT EXISTS idx_versions_format ON dataset_versions(format);
CREATE INDEX IF NOT EXISTS idx_versions_robot ON dataset_versions(robot_type);
CREATE INDEX IF NOT EXISTS idx_runs_started ON runs(started_at);
CREATE INDEX IF NOT EXISTS idx_lineage_parent ON lineage(parent_version_id);
CREATE INDEX IF NOT EXISTS idx_lineage_child ON lineage(child_version_id);
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
            dirs[:] = []
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
    """Stamp a dataset with its newest shallow metadata/content file."""
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


def _legacy_fingerprint(row: sqlite3.Row) -> str:
    payload = f"legacy\0{row['marker']}\0{row['mtime']}\0{row['size']}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _set_meta(conn: sqlite3.Connection, **values: Any) -> None:
    for key, value in values.items():
        conn.execute(
            "INSERT OR IGNORE INTO catalog_meta(key, value) VALUES (?, ?)",
            (key, json.dumps(value, ensure_ascii=False)),
        )


def _migrate_v1(conn: sqlite3.Connection) -> None:
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(datasets)")}
    if not columns or "current_version_id" in columns or "format" not in columns:
        return
    conn.execute("ALTER TABLE datasets RENAME TO datasets_v1_legacy")
    for name in ("idx_datasets_format", "idx_datasets_robot"):
        conn.execute(f"DROP INDEX IF EXISTS {name}")
    conn.executescript(SCHEMA)
    now = time.time()
    run_key = f"migrate-{uuid.uuid4().hex[:12]}"
    cursor = conn.execute(
        "INSERT INTO runs(run_key, kind, status, started_at, finished_at, config, metrics) "
        "VALUES (?, 'migrate', 'succeeded', ?, ?, ?, ?)",
        (run_key, now, now, json.dumps({"from": 1, "to": 2}), json.dumps({})),
    )
    run_id = int(cursor.lastrowid)
    migrated = 0
    for row in conn.execute("SELECT * FROM datasets_v1_legacy ORDER BY id"):
        created = float(row["indexed_at"] or now)
        dataset = conn.execute(
            "INSERT INTO datasets(path, created_at, updated_at) VALUES (?, ?, ?)",
            (row["path"], created, created),
        )
        dataset_id = int(dataset.lastrowid)
        version = conn.execute(
            "INSERT INTO dataset_versions("
            "dataset_id, run_id, fingerprint, format, format_version, episodes, frames, duration_s, fps, "
            "robot_type, tasks, features, notes, meta, marker, mtime, size, score, indexed_at"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                dataset_id,
                run_id,
                _legacy_fingerprint(row),
                row["format"],
                row["version"],
                row["episodes"],
                row["frames"],
                row["duration_s"],
                row["fps"],
                row["robot_type"],
                row["tasks"],
                row["features"],
                row["notes"],
                row["meta"],
                row["marker"],
                row["mtime"],
                row["size"],
                row["score"],
                created,
            ),
        )
        conn.execute(
            "UPDATE datasets SET current_version_id=?, updated_at=? WHERE id=?",
            (int(version.lastrowid), created, dataset_id),
        )
        migrated += 1
    conn.execute("UPDATE runs SET metrics=? WHERE id=?", (json.dumps({"migrated_datasets": migrated}), run_id))
    conn.execute("DROP TABLE datasets_v1_legacy")


def connect(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    _migrate_v1(conn)
    conn.executescript(SCHEMA)
    _set_meta(conn, schema_version=SCHEMA_VERSION, created_at=time.time())
    conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
    conn.commit()
    return conn


def _row_from_summary(summary: DatasetSummary, *, marker: str, mtime: float, size: int, score: int | None) -> dict[str, Any]:
    return {
        "format": summary.format,
        "format_version": summary.version,
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
    }


def _create_run(conn: sqlite3.Connection, kind: str, *, root: str | None = None, config: dict[str, Any] | None = None) -> tuple[int, str]:
    run_key = f"{kind}-{uuid.uuid4().hex[:12]}"
    cursor = conn.execute(
        "INSERT INTO runs(run_key, kind, status, started_at, root, config, metrics) VALUES (?, ?, 'running', ?, ?, ?, ?)",
        (run_key, kind, time.time(), root, json.dumps(config or {}, ensure_ascii=False), json.dumps({})),
    )
    return int(cursor.lastrowid), run_key


def _finish_run(
    conn: sqlite3.Connection,
    run_id: int,
    *,
    status: str,
    metrics: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    conn.execute(
        "UPDATE runs SET status=?, finished_at=?, metrics=?, error=? WHERE id=?",
        (status, time.time(), json.dumps(metrics or {}, ensure_ascii=False), error, run_id),
    )


def _fingerprint(path: Path, db_path: str | Path) -> dict[str, Any]:
    return dataset_fingerprint(path, exclude=[Path(db_path)])


def index(
    root: str | Path,
    db_path: str | Path,
    *,
    max_depth: int = 4,
    fmt: str | None = None,
    force: bool = False,
    validate: bool = False,
) -> dict[str, Any]:
    """Scan ``root``, recording a version for every changed dataset."""
    conn = connect(db_path)
    added = updated = skipped = failed = versions_added = lineage_added = 0
    failures: list[dict[str, str]] = []
    run_id, run_key = _create_run(
        conn,
        "index",
        root=str(Path(root).resolve()),
        config={"max_depth": max_depth, "fmt": fmt, "force": force, "validate": validate},
    )
    try:
        for candidate in _candidate_paths(Path(root), max_depth=max_depth):
            marker, mtime, size = _marker_stat(candidate)
            resolved = str(candidate.resolve())
            existing = conn.execute(
                "SELECT d.id, d.current_version_id, v.fingerprint, v.mtime, v.size "
                "FROM datasets d LEFT JOIN dataset_versions v ON v.id=d.current_version_id WHERE d.path=?",
                (resolved,),
            ).fetchone()
            if (
                existing
                and not force
                and abs(float(existing["mtime"] or 0.0) - mtime) < 1e-6
                and int(existing["size"] or 0) == size
            ):
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
            except Exception as exc:
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
            try:
                fingerprint = str(_fingerprint(candidate, db_path)["sha256"])
            except OSError as exc:
                failed += 1
                failures.append({"path": resolved, "error": f"fingerprint: {exc}"})
                continue

            now = time.time()
            if existing is None:
                cursor = conn.execute(
                    "INSERT INTO datasets(path, created_at, updated_at) VALUES (?, ?, ?)",
                    (resolved, now, now),
                )
                dataset_id = int(cursor.lastrowid)
                previous_version_id = None
                added += 1
            else:
                dataset_id = int(existing["id"])
                previous_version_id = existing["current_version_id"]
                updated += 1

            version = conn.execute(
                "SELECT id FROM dataset_versions WHERE dataset_id=? AND fingerprint=?",
                (dataset_id, fingerprint),
            ).fetchone()
            if version is None:
                row = _row_from_summary(summary, marker=marker, mtime=mtime, size=size, score=score)
                row.update({"dataset_id": dataset_id, "run_id": run_id, "fingerprint": fingerprint, "indexed_at": now})
                columns = ", ".join(row)
                placeholders = ", ".join(f":{key}" for key in row)
                cursor = conn.execute(f"INSERT INTO dataset_versions ({columns}) VALUES ({placeholders})", row)
                version_id = int(cursor.lastrowid)
                versions_added += 1
            else:
                version_id = int(version["id"])
                if score is not None:
                    conn.execute("UPDATE dataset_versions SET score=? WHERE id=?", (score, version_id))

            conn.execute(
                "UPDATE datasets SET current_version_id=?, updated_at=? WHERE id=?",
                (version_id, now, dataset_id),
            )
            if previous_version_id is not None and int(previous_version_id) != version_id:
                cursor = conn.execute(
                    "INSERT OR IGNORE INTO lineage(parent_version_id, child_version_id, relation, run_id, detail, created_at) "
                    "VALUES (?, ?, 'supersedes', ?, ?, ?)",
                    (
                        int(previous_version_id),
                        version_id,
                        run_id,
                        json.dumps({"marker": marker, "fingerprint": fingerprint}, ensure_ascii=False),
                        now,
                    ),
                )
                lineage_added += int(cursor.rowcount > 0)
        metrics = {
            "added": added,
            "updated": updated,
            "skipped": skipped,
            "failed": failed,
            "versions_added": versions_added,
            "lineage_added": lineage_added,
        }
        _finish_run(conn, run_id, status="succeeded", metrics=metrics)
        conn.commit()
        total = conn.execute("SELECT COUNT(*) FROM datasets").fetchone()[0]
        total_versions = conn.execute("SELECT COUNT(*) FROM dataset_versions").fetchone()[0]
    except BaseException as exc:
        _finish_run(conn, run_id, status="failed", error=f"{type(exc).__name__}: {exc}")
        conn.commit()
        raise
    finally:
        conn.close()
    return {
        "schema_version": SCHEMA_VERSION,
        "db": str(Path(db_path).resolve()),
        "root": str(Path(root).resolve()),
        "run_id": run_id,
        "run_key": run_key,
        "added": added,
        "updated": updated,
        "skipped": skipped,
        "failed": failed,
        "versions_added": versions_added,
        "lineage_added": lineage_added,
        "total": total,
        "total_versions": total_versions,
        "failures": failures[:50],
    }


def _decode_row(row: dict[str, Any]) -> dict[str, Any]:
    for key in ("tasks", "features", "notes", "meta", "config", "metrics", "detail"):
        if key not in row or row.get(key) is None:
            continue
        try:
            row[key] = json.loads(row[key])
        except (TypeError, json.JSONDecodeError):
            pass
    return row


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
        clauses.append("(d.path LIKE ? OR v.tasks LIKE ? OR v.robot_type LIKE ? OR v.format LIKE ?)")
        params.extend([like, like, like, like])
    for column, value in (("format", fmt), ("robot_type", robot_type)):
        if value:
            clauses.append(f"v.{column} = ?")
            params.append(value)
    if min_episodes is not None:
        clauses.append("v.episodes >= ?")
        params.append(min_episodes)
    if max_episodes is not None:
        clauses.append("v.episodes <= ?")
        params.append(max_episodes)
    if min_duration is not None:
        clauses.append("v.duration_s >= ?")
        params.append(min_duration)
    if min_score is not None:
        clauses.append("v.score >= ?")
        params.append(min_score)
    allowed = {
        "indexed_at": "v.indexed_at",
        "episodes": "v.episodes",
        "frames": "v.frames",
        "duration_s": "v.duration_s",
        "score": "v.score",
        "path": "d.path",
        "mtime": "v.mtime",
    }
    column = allowed.get(order_by, "v.indexed_at")
    direction = "DESC" if descending else "ASC"
    sql = (
        "SELECT d.id AS dataset_id, d.path, d.created_at AS dataset_created_at, d.updated_at, "
        "v.id AS version_id, v.fingerprint, v.format, v.format_version AS version, v.episodes, v.frames, "
        "v.duration_s, v.fps, v.robot_type, v.tasks, v.features, v.notes, v.meta, v.score, v.marker, "
        "v.mtime, v.size, v.indexed_at FROM datasets d JOIN dataset_versions v ON v.id=d.current_version_id"
    )
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += f" ORDER BY {column} {direction} LIMIT ?"
    params.append(int(limit))
    conn = connect(db_path)
    try:
        rows = [_decode_row(dict(row)) for row in conn.execute(sql, params)]
    finally:
        conn.close()
    return rows


def versions(db_path: str | Path, *, path: str | Path | None = None, limit: int = 50) -> list[dict[str, Any]]:
    if not Path(db_path).expanduser().exists():
        raise FileNotFoundError(db_path)
    sql = (
        "SELECT v.id AS version_id, d.path, v.fingerprint, v.format, v.format_version AS version, v.episodes, "
        "v.frames, v.duration_s, v.fps, v.robot_type, v.tasks, v.features, v.notes, v.meta, v.score, "
        "v.marker, v.mtime, v.size, v.indexed_at, v.run_id FROM dataset_versions v "
        "JOIN datasets d ON d.id=v.dataset_id"
    )
    params: list[Any] = []
    if path is not None:
        sql += " WHERE d.path=?"
        params.append(str(Path(path).expanduser().resolve()))
    sql += " ORDER BY v.indexed_at DESC, v.id DESC LIMIT ?"
    params.append(max(1, int(limit)))
    conn = connect(db_path)
    try:
        rows = [_decode_row(dict(row)) for row in conn.execute(sql, params)]
    finally:
        conn.close()
    return rows


def runs(db_path: str | Path, *, limit: int = 20, kind: str | None = None, status: str | None = None) -> list[dict[str, Any]]:
    if not Path(db_path).expanduser().exists():
        raise FileNotFoundError(db_path)
    clauses: list[str] = []
    params: list[Any] = []
    if kind:
        clauses.append("kind=?")
        params.append(kind)
    if status:
        clauses.append("status=?")
        params.append(status)
    sql = (
        "SELECT r.*, (SELECT COUNT(*) FROM dataset_versions v WHERE v.run_id=r.id) AS versions_written "
        "FROM runs r"
    )
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY r.started_at DESC, r.id DESC LIMIT ?"
    params.append(max(1, int(limit)))
    conn = connect(db_path)
    try:
        rows = [_decode_row(dict(row)) for row in conn.execute(sql, params)]
    finally:
        conn.close()
    return rows


def lineage(
    db_path: str | Path,
    *,
    path: str | Path | None = None,
    version_id: int | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    if not Path(db_path).expanduser().exists():
        raise FileNotFoundError(db_path)
    clauses: list[str] = []
    params: list[Any] = []
    if path is not None:
        clauses.append("(pd.path=? OR cd.path=?)")
        resolved = str(Path(path).expanduser().resolve())
        params.extend([resolved, resolved])
    if version_id is not None:
        clauses.append("(l.parent_version_id=? OR l.child_version_id=?)")
        params.extend([int(version_id), int(version_id)])
    sql = (
        "SELECT l.id, l.parent_version_id, l.child_version_id, l.relation, l.run_id, l.detail, l.created_at, "
        "pd.path AS parent_path, cd.path AS child_path, pv.fingerprint AS parent_fingerprint, "
        "cv.fingerprint AS child_fingerprint FROM lineage l "
        "JOIN dataset_versions pv ON pv.id=l.parent_version_id "
        "JOIN dataset_versions cv ON cv.id=l.child_version_id "
        "JOIN datasets pd ON pd.id=pv.dataset_id JOIN datasets cd ON cd.id=cv.dataset_id"
    )
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY l.created_at DESC, l.id DESC LIMIT ?"
    params.append(max(1, int(limit)))
    conn = connect(db_path)
    try:
        rows = [_decode_row(dict(row)) for row in conn.execute(sql, params)]
    finally:
        conn.close()
    return rows


def stats(db_path: str | Path) -> dict[str, Any]:
    if not Path(db_path).expanduser().exists():
        raise FileNotFoundError(db_path)
    conn = connect(db_path)
    try:
        total = conn.execute("SELECT COUNT(*) FROM datasets").fetchone()[0]
        total_versions = conn.execute("SELECT COUNT(*) FROM dataset_versions").fetchone()[0]
        total_runs = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        lineage_edges = conn.execute("SELECT COUNT(*) FROM lineage").fetchone()[0]
        by_format = {
            row["format"]: row["n"]
            for row in conn.execute(
                "SELECT v.format, COUNT(*) AS n FROM datasets d JOIN dataset_versions v ON v.id=d.current_version_id "
                "GROUP BY v.format ORDER BY n DESC"
            )
        }
        episodes = conn.execute(
            "SELECT SUM(v.episodes), SUM(v.frames), SUM(v.duration_s) FROM datasets d "
            "JOIN dataset_versions v ON v.id=d.current_version_id"
        ).fetchone()
        robots = [
            row["robot_type"]
            for row in conn.execute(
                "SELECT DISTINCT v.robot_type FROM datasets d JOIN dataset_versions v ON v.id=d.current_version_id "
                "WHERE v.robot_type IS NOT NULL"
            )
        ]
        last = conn.execute("SELECT MAX(indexed_at) FROM dataset_versions").fetchone()[0]
        latest_run = conn.execute("SELECT MAX(started_at) FROM runs").fetchone()[0]
        schema_version = conn.execute("SELECT value FROM catalog_meta WHERE key='schema_version'").fetchone()
    finally:
        conn.close()
    return {
        "schema_version": json.loads(schema_version[0]) if schema_version else SCHEMA_VERSION,
        "total_datasets": total,
        "total_versions": total_versions,
        "total_runs": total_runs,
        "lineage_edges": lineage_edges,
        "by_format": by_format,
        "total_episodes": episodes[0] or 0,
        "total_frames": episodes[1] or 0,
        "total_duration_s": round(float(episodes[2] or 0), 2),
        "robot_types": sorted(robots),
        "last_indexed": last,
        "last_run": latest_run,
    }


def remove(db_path: str | Path, path: str | Path) -> bool:
    conn = connect(db_path)
    try:
        cursor = conn.execute("DELETE FROM datasets WHERE path = ?", (str(Path(path).expanduser().resolve()),))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def history(db_path: str | Path, *, path: str | Path | None = None, limit: int = 20) -> dict[str, Any]:
    """Combined versions, lineage and recent runs for one dataset or the catalog."""
    return {
        "schema_version": SCHEMA_VERSION,
        "versions": versions(db_path, path=path, limit=limit),
        "lineage": lineage(db_path, path=path, limit=limit),
        "runs": runs(db_path, limit=limit),
    }


def export_manifest(db_path: str | Path, *, limit: int = 1000) -> dict[str, Any]:
    rows = search(db_path, limit=limit)
    conn = connect(db_path)
    try:
        counts = {
            row["dataset_id"]: int(row["n"])
            for row in conn.execute(
                "SELECT dataset_id, COUNT(*) AS n FROM dataset_versions GROUP BY dataset_id"
            )
        }
    finally:
        conn.close()
    return {
        "schema_version": SCHEMA_VERSION,
        "count": len(rows),
        "datasets": [
            {
                "dataset_id": row["dataset_id"],
                "version_id": row["version_id"],
                "fingerprint": row["fingerprint"],
                "versions": counts.get(row["dataset_id"], 0),
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
