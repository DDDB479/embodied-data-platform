"""Incremental cache for dataset quality reports.

Quality reports are expensive because they load every frame. The cache key
combines three things:

* the resolved dataset path;
* a fingerprint of every file below it (relative path, size and mtime);
* the normalized analysis parameters and the algorithm version.

Only metadata is cached, never frames. A metadata-only change invalidates the
entry as reliably as a data change, while a cache hit avoids opening parquet or
HDF5 files at all.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

CACHE_SCHEMA_VERSION = 1
QUALITY_ALGORITHM_VERSION = 2

_SKIP_DIRS = {".git", "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS quality_cache (
    key TEXT PRIMARY KEY,
    path TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    params TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at REAL NOT NULL,
    accessed_at REAL NOT NULL,
    hits INTEGER NOT NULL DEFAULT 0,
    compute_ms REAL
);
CREATE INDEX IF NOT EXISTS idx_quality_cache_path ON quality_cache(path);
CREATE INDEX IF NOT EXISTS idx_quality_cache_accessed ON quality_cache(accessed_at);
"""


def default_quality_cache_path() -> Path:
    """Return the per-user cache file, respecting common OS conventions.

    ``EMBODIED_QUALITY_CACHE`` wins so CI and container images can point the
    cache at a mounted volume without changing the application config.
    """

    override = os.environ.get("EMBODIED_QUALITY_CACHE")
    if override:
        return Path(override).expanduser()
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_CACHE_HOME")
    if base:
        return Path(base) / "embodied-data" / "quality-cache.sqlite3"
    return Path.home() / ".cache" / "embodied-data" / "quality-cache.sqlite3"


def _iter_dataset_files(root: Path, exclude: Sequence[Path]) -> Iterable[Path]:
    excluded = [path.resolve() for path in exclude]

    def skipped(path: Path) -> bool:
        resolved = path.resolve()
        return any(resolved == item or str(resolved).startswith(f"{item}-") for item in excluded)

    if root.is_file():
        if not skipped(root):
            yield root
        return
    for current, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in _SKIP_DIRS and not d.startswith("."))
        for name in sorted(files):
            candidate = Path(current) / name
            if not skipped(candidate):
                yield candidate


def dataset_fingerprint(path: str | Path, *, exclude: Sequence[str | Path] = ()) -> dict[str, Any]:
    """Fingerprint a dataset's on-disk shape without reading file contents.

    The fingerprint is deliberately cheap: walking large trees costs a few
    stat calls per file, while hashing every parquet shard would defeat the
    point of an incremental cache. ``refresh_cache`` is the escape hatch if a
    filesystem reports unreliable mtimes.
    """

    root = Path(path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(str(path))
    digest = hashlib.sha256()
    files = 0
    total_bytes = 0
    for item in _iter_dataset_files(root, exclude):
        try:
            stat = item.stat()
        except OSError as exc:
            raise OSError(f"cannot stat {item}: {exc}") from exc
        relative = item.relative_to(root).as_posix() if root.is_dir() else item.name
        mtime_ns = getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000))
        digest.update(f"{relative}\0{stat.st_size}\0{mtime_ns}\n".encode("utf-8"))
        files += 1
        total_bytes += int(stat.st_size)
    return {"sha256": digest.hexdigest(), "files": files, "bytes": total_bytes, "root": str(root)}


def normalize_quality_params(params: dict[str, Any]) -> dict[str, Any]:
    """Canonicalize parameters so argument order cannot fragment the cache."""

    episodes = params.get("episodes")
    features = params.get("features")
    return {
        "episodes": sorted({int(value) for value in episodes}) if episodes else None,
        "features": sorted({str(value) for value in features}) if features else None,
        "z": round(float(params.get("z", 8.0)), 6),
        "include_segments": bool(params.get("include_segments", False)),
        "skip_anomalies": bool(params.get("skip_anomalies", False)),
        "chunk_frames": int(params.get("chunk_frames", 50_000)),
        "max_frames": int(params.get("max_frames", 2_000_000)),
    }


def quality_cache_key(path: str | Path, fingerprint: str, params: dict[str, Any]) -> str:
    payload = {
        "schema": CACHE_SCHEMA_VERSION,
        "algorithm": QUALITY_ALGORITHM_VERSION,
        "path": str(Path(path).expanduser().resolve()),
        "fingerprint": fingerprint,
        "params": normalize_quality_params(params),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class QualityCache:
    """Small SQLite-backed result cache; safe to call from worker threads."""

    def __init__(self, path: str | Path | None = None, *, max_entries: int = 1000) -> None:
        self.path = Path(path).expanduser() if path is not None else default_quality_cache_path()
        self.max_entries = max(1, int(max_entries))

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.path), timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.executescript(SCHEMA)
        return conn

    def get(self, key: str) -> dict[str, Any] | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT key, path, fingerprint, params, payload, created_at, accessed_at, hits, compute_ms "
                "FROM quality_cache WHERE key = ?",
                (key,),
            ).fetchone()
            if row is None:
                return None
            now = time.time()
            conn.execute("UPDATE quality_cache SET accessed_at = ?, hits = hits + 1 WHERE key = ?", (now, key))
            conn.commit()
            return {
                "key": row["key"],
                "path": row["path"],
                "fingerprint": row["fingerprint"],
                "params": json.loads(row["params"]),
                "payload": json.loads(row["payload"]),
                "created_at": row["created_at"],
                "accessed_at": now,
                "hits": int(row["hits"]) + 1,
                "compute_ms": row["compute_ms"],
            }
        finally:
            conn.close()

    def put(
        self,
        key: str,
        *,
        path: str | Path,
        fingerprint: str,
        params: dict[str, Any],
        payload: dict[str, Any],
        compute_ms: float | None = None,
    ) -> None:
        now = time.time()
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO quality_cache (key, path, fingerprint, params, payload, created_at, accessed_at, hits, compute_ms) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?) "
                "ON CONFLICT(key) DO UPDATE SET path=excluded.path, fingerprint=excluded.fingerprint, "
                "params=excluded.params, payload=excluded.payload, created_at=excluded.created_at, "
                "accessed_at=excluded.accessed_at, compute_ms=excluded.compute_ms",
                (
                    key,
                    str(Path(path).expanduser().resolve()),
                    fingerprint,
                    json.dumps(normalize_quality_params(params), sort_keys=True),
                    encoded,
                    now,
                    now,
                    compute_ms,
                ),
            )
            count = int(conn.execute("SELECT COUNT(*) FROM quality_cache").fetchone()[0])
            if count > self.max_entries:
                conn.execute(
                    "DELETE FROM quality_cache WHERE key IN ("
                    "SELECT key FROM quality_cache ORDER BY accessed_at ASC LIMIT ?)",
                    (count - self.max_entries,),
                )
            conn.commit()
        finally:
            conn.close()

    def stats(self) -> dict[str, Any]:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT COUNT(*) AS entries, COALESCE(SUM(hits), 0) AS hits, "
                "COALESCE(SUM(LENGTH(payload)), 0) AS bytes, MIN(created_at) AS oldest, MAX(accessed_at) AS newest "
                "FROM quality_cache"
            ).fetchone()
        finally:
            conn.close()
        return {
            "path": str(self.path),
            "entries": int(row["entries"]),
            "hits": int(row["hits"]),
            "bytes": int(row["bytes"]),
            "oldest": row["oldest"],
            "newest": row["newest"],
        }

    def clear(self) -> int:
        conn = self._connect()
        try:
            cursor = conn.execute("DELETE FROM quality_cache")
            conn.commit()
            return int(cursor.rowcount)
        finally:
            conn.close()
