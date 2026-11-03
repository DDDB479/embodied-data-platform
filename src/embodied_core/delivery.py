"""Versioned delivery packs with manifests and checksum verification."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence

from . import __version__
from .cache import dataset_fingerprint
from .convert import convert, to_zarr
from .gate import evaluate_quality_gate
from .media import modality_inventory
from .model import to_jsonable
from .registry import open_dataset

DELIVERY_SCHEMA_VERSION = 1


class DeliveryError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _files(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        rows.append({"path": path.relative_to(root).as_posix(), "size": path.stat().st_size, "sha256": _sha256(path)})
    return rows


def _safe_format(value: str) -> str:
    aliases = {
        "jsonl": "jsonl",
        "canonical": "jsonl",
        "lerobot": "lerobot-v2",
        "lerobot-v2": "lerobot-v2",
        "lerobot-v2.1": "lerobot-v2",
        "lerobot-v3": "lerobot-v3",
        "lerobot-v3.0": "lerobot-v3",
        "zarr": "zarr",
        "zarr-v2": "zarr",
        "mcap-inventory": "mcap-inventory",
    }
    key = str(value).strip().lower()
    if key not in aliases:
        raise DeliveryError(f"unsupported delivery format {value!r}; supported: {sorted(set(aliases.values()))}")
    return aliases[key]


def build_delivery(
    source: str | Path,
    destination: str | Path,
    *,
    formats: Sequence[str] = ("lerobot-v3",),
    name: str | None = None,
    version: str = "1",
    overwrite: bool = False,
    quality_rules: dict[str, Any] | str | Path | None = None,
    license: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an atomic delivery pack and write a checksummed manifest."""

    src = Path(source).expanduser().resolve()
    target = Path(destination).expanduser()
    if target.exists() and not overwrite:
        raise FileExistsError(str(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.delivery-", dir=str(target.parent)))
    selected = []
    for item in formats:
        normalized = _safe_format(item)
        if normalized not in selected:
            selected.append(normalized)
    if not selected:
        raise DeliveryError("at least one delivery format is required")
    dataset = open_dataset(src)
    try:
        summary = dataset.summary.to_dict(include_episodes=False)
        modalities = modality_inventory(dataset.summary)
        dataset_format = dataset.summary.format
    finally:
        dataset.close()
    fingerprint = dataset_fingerprint(src)
    delivery_id = "dl_" + hashlib.sha256(
        json.dumps(
            {
                "name": name or src.name,
                "version": version,
                "source": str(src),
                "fingerprint": fingerprint["sha256"],
                "formats": selected,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:16]
    artifacts: list[dict[str, Any]] = []
    try:
        artifacts_root = staging / "artifacts"
        artifacts_root.mkdir(parents=True, exist_ok=True)
        for item in selected:
            folder = artifacts_root / item
            if item in ("jsonl", "lerobot-v2", "lerobot-v3"):
                target_name = {"jsonl": "jsonl", "lerobot-v2": "lerobot", "lerobot-v3": "lerobot-v3"}[item]
                result = convert(src, folder, target=target_name, overwrite=False)
            elif item == "zarr":
                dataset = open_dataset(src)
                try:
                    result = to_zarr(dataset, folder, overwrite=False)
                finally:
                    dataset.close()
            else:
                folder.mkdir(parents=True, exist_ok=True)
                inventory = {
                    "source": str(src),
                    "format": dataset_format,
                    "summary": summary,
                    "modalities": modalities,
                }
                (folder / "mcap-inventory.json").write_text(json.dumps(inventory, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                result = {"target": "mcap-inventory", "destination": str(folder)}
            files = _files(folder)
            artifacts.append(
                {
                    "format": item,
                    "path": f"artifacts/{item}",
                    "result": to_jsonable(result),
                    "file_count": len(files),
                    "bytes": sum(int(entry["size"]) for entry in files),
                    "files": files,
                }
            )
        gate = None
        if quality_rules is not None:
            report = evaluate_quality_gate(src, rules=quality_rules)
            gate = {
                "ok": report.get("ok"),
                "score": report.get("score"),
                "failures": (report.get("gate") or {}).get("failures", []),
            }
        manifest = {
            "schema_version": DELIVERY_SCHEMA_VERSION,
            "delivery_id": delivery_id,
            "name": name or src.name,
            "version": str(version),
            "created_at": time.time(),
            "tool_version": __version__,
            "source": {
                "path": str(src),
                "format": dataset_format,
                "fingerprint": fingerprint,
                "summary": summary,
            },
            "formats": selected,
            "artifacts": artifacts,
            "quality_gate": gate,
            "license": license,
            "metadata": to_jsonable(metadata or {}),
        }
        (staging / "delivery-manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        os.replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    manifest_path = target / "delivery-manifest.json"
    return {
        "delivery_id": delivery_id,
        "destination": str(target.resolve()),
        "manifest": str(manifest_path.resolve()),
        "formats": selected,
        "artifacts": len(artifacts),
        "files": sum(int(item["file_count"]) for item in artifacts),
        "bytes": sum(int(item["bytes"]) for item in artifacts),
        "quality_gate": gate,
    }


def check_delivery(destination: str | Path, *, verify_hashes: bool = True) -> dict[str, Any]:
    """Verify a delivery manifest and every recorded artifact file."""

    root = Path(destination).expanduser()
    manifest_path = root / "delivery-manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(str(manifest_path))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    issues: list[dict[str, Any]] = []
    checked = 0
    recorded: set[str] = set()
    for artifact in manifest.get("artifacts") or []:
        for entry in artifact.get("files") or []:
            relative = str(entry.get("path") or "")
            parsed = PurePosixPath(relative)
            if parsed.is_absolute() or ".." in parsed.parts:
                issues.append({"code": "unsafe_path", "path": relative})
                continue
            recorded.add(f"{artifact.get('path')}/{relative}")
            path = root / str(artifact.get("path")) / relative
            if not path.is_file():
                issues.append({"code": "missing_file", "path": str(path)})
                continue
            checked += 1
            if int(path.stat().st_size) != int(entry.get("size", -1)):
                issues.append({"code": "size_mismatch", "path": str(path), "expected": entry.get("size"), "actual": path.stat().st_size})
            if verify_hashes and _sha256(path) != entry.get("sha256"):
                issues.append({"code": "sha256_mismatch", "path": str(path)})
    present = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "delivery-manifest.json"
    }
    unexpected = sorted(present - recorded)
    return {
        "ok": not issues,
        "delivery_id": manifest.get("delivery_id"),
        "name": manifest.get("name"),
        "version": manifest.get("version"),
        "formats": manifest.get("formats", []),
        "files_checked": checked,
        "issues": issues,
        "unexpected_files": unexpected,
        "quality_gate": manifest.get("quality_gate"),
    }
