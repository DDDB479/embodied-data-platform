"""Object-store URI adapter with atomic, checksummed local materialization.

Supported schemes: local paths / ``file://``, ``http(s)://`` single objects and
``s3://`` prefixes when boto3 is installed. Downloads are staged in a temporary
directory, checksummed, and published atomically with a source manifest.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


class StorageError(RuntimeError):
    pass


class StorageUnavailable(StorageError):
    def __init__(self, reason: str, *, install: str | None = None) -> None:
        self.reason = reason
        self.install = install
        super().__init__(reason)


_SCHEME = re.compile(r"^([a-zA-Z][a-zA-Z0-9+.-]*)://")


@dataclass(frozen=True)
class StorageURI:
    raw: str
    scheme: str
    location: str


def parse_storage_uri(uri: str | Path) -> StorageURI:
    text = str(uri)
    match = _SCHEME.match(text)
    if not match:
        return StorageURI(text, "file", text)
    scheme = match.group(1).lower()
    if scheme == "file":
        parsed = urllib.parse.urlparse(text)
        path = urllib.request.url2pathname(parsed.path)
        if parsed.netloc and os.name == "nt":
            path = f"//{parsed.netloc}{path}"
        return StorageURI(text, "file", path)
    if scheme not in {"http", "https", "s3"}:
        raise StorageError(f"unsupported storage scheme {scheme!r}; use file, http(s) or s3")
    return StorageURI(text, scheme, text)


def _safe_relative(name: str) -> Path:
    clean = PurePosixPath(str(name).replace("\\", "/"))
    if clean.is_absolute() or ".." in clean.parts or not clean.parts:
        raise StorageError(f"unsafe object key {name!r}")
    return Path(*clean.parts)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _max_download_bytes() -> int:
    try:
        return max(1, int(os.environ.get("EMBODIED_MAX_DOWNLOAD_BYTES", 10 * 1024 * 1024 * 1024)))
    except ValueError:
        return 10 * 1024 * 1024 * 1024


def _manifest_path(destination: Path) -> Path:
    return destination / ".embodied-source.json" if destination.is_dir() else destination.parent / f".{destination.name}.embodied-source.json"


def _write_manifest(destination: Path, source: StorageURI, files: list[dict[str, Any]]) -> Path:
    manifest = {
        "schema_version": 1,
        "source": source.raw,
        "scheme": source.scheme,
        "fetched_at": time.time(),
        "files": files,
        "total_bytes": sum(int(item["size"]) for item in files),
    }
    target = _manifest_path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temp, target)
    return target


def _request(url: str, *, timeout: float, retries: int) -> bytes:
    last: Exception | None = None
    for attempt in range(max(0, retries) + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "embodied-core/0.5"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                length = int(response.headers.get("Content-Length") or 0)
                maximum = _max_download_bytes()
                if length and length > maximum:
                    raise StorageError(f"remote object is {length} bytes, above EMBODIED_MAX_DOWNLOAD_BYTES={maximum}")
                data = response.read(maximum + 1)
                if len(data) > maximum:
                    raise StorageError(f"download exceeded EMBODIED_MAX_DOWNLOAD_BYTES={maximum}")
                return data
        except (OSError, urllib.error.URLError, TimeoutError) as exc:
            last = exc
            if attempt < retries:
                time.sleep(min(2.0, 0.2 * (2**attempt)))
    raise StorageError(f"download failed after {retries + 1} attempt(s): {last}")


def _s3_client() -> Any:
    try:
        import boto3  # type: ignore
    except Exception as exc:
        raise StorageUnavailable("S3 support requires boto3", install="pip install 'embodied-core[s3]'") from exc
    return boto3.client("s3")


def _s3_parts(uri: StorageURI) -> tuple[str, str]:
    parsed = urllib.parse.urlparse(uri.raw)
    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/")
    if not bucket:
        raise StorageError("s3 URI must include a bucket: s3://bucket/prefix")
    return bucket, prefix


def _copy_local(source: Path, staged: Path) -> list[dict[str, Any]]:
    if source.is_dir():
        shutil.copytree(source, staged, dirs_exist_ok=False)
        root = staged
    elif source.is_file():
        staged.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, staged)
        root = staged
    else:
        raise FileNotFoundError(str(source))
    files: list[dict[str, Any]] = []
    candidates = [root] if root.is_file() else [path for path in root.rglob("*") if path.is_file()]
    for path in candidates:
        relative = path.name if root.is_file() else path.relative_to(root).as_posix()
        files.append({"path": relative, "size": path.stat().st_size, "sha256": _sha256(path)})
    return files


def _publish(temp_root: Path, destination: Path, *, overwrite: bool) -> None:
    if destination.exists():
        if not overwrite:
            raise FileExistsError(str(destination))
        if destination.is_dir():
            shutil.rmtree(destination)
        else:
            destination.unlink()
    os.replace(temp_root, destination)


def pull(
    uri: str | Path,
    destination: str | Path,
    *,
    overwrite: bool = False,
    retries: int = 2,
    timeout_s: float = 30.0,
) -> dict[str, Any]:
    """Materialize a local/HTTP/S3 object or prefix into ``destination``."""

    source = parse_storage_uri(uri)
    target = Path(destination).expanduser()
    if source.scheme == "file":
        source_path = Path(source.location).expanduser().resolve()
        target_path = target.resolve()
        if source_path.is_dir() and (target_path == source_path or source_path in target_path.parents):
            raise StorageError("destination cannot be inside the source directory")
    if target.exists() and not overwrite:
        raise FileExistsError(str(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=str(target.parent)))
    try:
        if source.scheme == "file":
            staged = staging / target.name
            files = _copy_local(Path(source.location).expanduser(), staged)
            _publish(staged, target, overwrite=overwrite)
        elif source.scheme in {"http", "https"}:
            staged = staging / target.name
            staged.parent.mkdir(parents=True, exist_ok=True)
            data = _request(source.raw, timeout=timeout_s, retries=retries)
            staged.write_bytes(data)
            files = [{"path": target.name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}]
            _publish(staged, target, overwrite=overwrite)
        else:
            bucket, prefix = _s3_parts(source)
            client = _s3_client()
            staged = staging / target.name
            staged.mkdir(parents=True, exist_ok=True)
            paginator = client.get_paginator("list_objects_v2")
            files = []
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                for item in page.get("Contents", []):
                    key = str(item["Key"])
                    if key.endswith("/"):
                        continue
                    relative_text = key[len(prefix) :].lstrip("/") if prefix else key
                    relative = _safe_relative(relative_text)
                    target_file = staged / relative
                    target_file.parent.mkdir(parents=True, exist_ok=True)
                    client.download_file(bucket, key, str(target_file))
                    files.append(
                        {
                            "path": relative.as_posix(),
                            "size": target_file.stat().st_size,
                            "sha256": _sha256(target_file),
                            "etag": str(item.get("ETag", "")).strip('"') or None,
                        }
                    )
            if not files:
                raise StorageError(f"s3 prefix s3://{bucket}/{prefix} contains no objects")
            _publish(staged, target, overwrite=overwrite)
        manifest_path = _write_manifest(target, source, files)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    shutil.rmtree(staging, ignore_errors=True)
    return {
        "source": source.raw,
        "scheme": source.scheme,
        "destination": str(target.resolve()),
        "files": len(files),
        "bytes": sum(int(item["size"]) for item in files),
        "manifest": str(manifest_path.resolve()),
    }
