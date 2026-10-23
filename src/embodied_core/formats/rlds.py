"""RLDS adapter (Google's Reinforcement Learning Datasets / TFDS layout).

RLDS stores episodes as TFRecord shards of ``tf.train.Example`` /
``tf.train.SequenceExample`` protos, described by ``dataset_info.json`` (and
optionally ``features.json``). This adapter validates the metadata and counts
the records in each shard with a dependency-free TFRecord walker; decoding the
example protos themselves needs TensorFlow and is therefore reported as an
optional capability. RLDS is a training-oriented export: episodes are steps
grouped by ``is_first``/``is_last`` flags, so one TFRecord is one *step*, not
one episode.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

from ..model import DatasetSummary, EpisodeRecord, FeatureSpec
from .base import DatasetObject, Detection, FramesUnavailable, guess_kind

FORMAT = "rlds"


def _info_file(root: Path) -> Path | None:
    if root.is_file():
        return None
    for name in ("dataset_info.json", "features.json"):
        candidate = root / name
        if candidate.is_file():
            return candidate
    return None


def _shards(root: Path) -> list[Path]:
    if root.is_file():
        return []
    patterns = ("*.tfrecord", "*.tfrecord-*", "*.tfrecords", "*.tfrecord.gz", "*.riegeli")
    found: list[Path] = []
    for pattern in patterns:
        found.extend(root.rglob(pattern))
    return sorted({p for p in found if p.is_file()})


def _is_tfrecord(path: Path) -> bool:
    if path.name.endswith(".riegeli") or path.name.endswith(".gz"):
        return False
    try:
        with path.open("rb") as handle:
            head = handle.read(12)
    except OSError:
        return False
    if len(head) < 12:
        return False
    length = struct.unpack("<Q", head[:8])[0]
    return 0 < length < 512 * 1024 * 1024


def count_tfrecords(path: Path, limit: int | None = None) -> int:
    """Count records without decoding them (length-prefixed framing only)."""
    count = 0
    size = path.stat().st_size
    with path.open("rb") as handle:
        while handle.tell() + 12 <= size:
            head = handle.read(12)
            if len(head) < 12:
                break
            length = struct.unpack("<Q", head[:8])[0]
            if length <= 0 or handle.tell() + length + 4 > size:
                break
            handle.seek(length + 4, 1)
            count += 1
            if limit is not None and count >= limit:
                break
    return count


def detect(root: Path) -> Detection | None:
    info = _info_file(root)
    if info is None:
        return None
    shards = _shards(root)
    if not shards:
        names = {"dataset_info.json", "features.json"}
        if info.name in names and any(p.suffix in {".tfrecord", ".riegeli"} for p in root.glob("*")):
            shards = [p for p in root.glob("*") if p.suffix in {".tfrecord", ".riegeli"}]
    evidence = [f"{info.name} present"]
    if shards:
        evidence.append(f"{len(shards)} shard(s)")
    if info.name == "features.json":
        return Detection(FORMAT, version=None, confidence=0.75, evidence=evidence)
    try:
        payload = json.loads(info.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return Detection(FORMAT, version=None, confidence=0.6, evidence=evidence)
    if "splits" in payload or "features" in payload:
        return Detection(FORMAT, version=str(payload.get("dataset_version") or "") or None, confidence=0.92, evidence=evidence)
    return None


def _flatten_features(
    node: Any,
    prefix: str = "",
    out: list[FeatureSpec] | None = None,
    in_sequence: bool = False,
) -> list[FeatureSpec]:
    """Walk RLDS ``features`` / ``sequence`` nesting into flat specs.

    RLDS schema::

        {"features": {"steps": {"sequence": {"feature": {
            "observation": {"features": {"state": {"shape": [7], "dtype": "tf.float32"}}},
            "action": {"shape": [7], "dtype": "tf.float32"}}}}}}

    so a ``sequence`` node is a transparent wrapper: it only marks every leaf
    below it as one-step-of-many, which is why ``steps`` never appears in the
    resulting keys.
    """
    out = out if out is not None else []
    if not isinstance(node, dict):
        return out
    sequence = node.get("sequence")
    if isinstance(sequence, dict):
        inner = sequence.get("feature")
        if isinstance(inner, dict):
            return _flatten_features(inner, prefix, out, True)
        # Some exporters inline the feature map directly in the sequence dict
        # instead of nesting it under ``feature``.
        rest = {key: value for key, value in sequence.items() if key not in ("feature", "length")}
        return _flatten_features({"features": rest}, prefix, out, True)
    children = node.get("features")
    if isinstance(children, dict):
        for key, child in children.items():
            _flatten_features(child, f"{prefix}/{key}" if prefix else str(key), out, in_sequence)
        return out
    dtype = node.get("dtype") or node.get("python_dtype") or "unknown"
    if isinstance(dtype, str):
        dtype = {
            "tf.float32": "float32",
            "tf.float64": "float64",
            "tf.int64": "int64",
            "tf.int32": "int32",
            "tf.string": "string",
            "tf.bool": "bool",
        }.get(dtype, dtype)
    shape = node.get("shape")
    dims = tuple(int(v) for v in shape if isinstance(v, (int, float))) if isinstance(shape, list) else ()
    key = prefix or str(node.get("name") or "<root>")
    out.append(
        FeatureSpec(
            key=key,
            dtype=str(dtype),
            shape=dims,
            kind=guess_kind(key),
            extra={"sequence": in_sequence, "shape": shape},
        )
    )
    return out


def _split_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    splits = payload.get("splits")
    if isinstance(splits, list):
        for split in splits:
            if not isinstance(split, dict):
                continue
            lengths = split.get("shardLengths") or split.get("shard_lengths") or []
            try:
                steps = int(sum(int(v) for v in lengths))
            except (TypeError, ValueError):
                steps = None
            rows.append({"name": split.get("name"), "shards": len(lengths), "steps": steps, "num_bytes": split.get("numBytes")})
    elif isinstance(payload.get("splits"), dict):
        for name, split in payload["splits"].items():
            if isinstance(split, dict):
                lengths = split.get("shardLengths") or []
                rows.append({"name": name, "shards": len(lengths), "steps": int(sum(int(v) for v in lengths)) if lengths else None})
    return rows


class RLDSDataset(DatasetObject):
    def __init__(self, root: Path, summary: DatasetSummary) -> None:
        super().__init__(root, summary, capabilities={"episodes", "metadata"})

    @staticmethod
    def optional_dependencies() -> dict[str, str]:
        return {"tensorflow": "decode TFRecord example protos (tensorflow / tensorflow-datasets)"}

    def frames(self, episodes=None, features=None):  # noqa: ANN001
        raise FramesUnavailable("RLDS frames live in TFRecord example protos; install tensorflow or convert with tfds")


def load(root: Path, detection: Detection | None = None) -> RLDSDataset:
    info = _info_file(root)
    payload: dict[str, Any] = {}
    notes: list[str] = []
    if info is not None:
        try:
            payload = json.loads(info.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            notes.append(f"{info.name}: {exc}")
            payload = {}
    # ``dataset_info.json`` may expose features at the top level (``features``)
    # or as the bare mapping itself; wrapping normalises both shapes so the
    # recursion starts from a container rather than from a single leaf.
    features = _flatten_features({"features": payload.get("features") or payload})
    if not features:
        notes.append("no feature descriptions found")
    shards = _shards(root)
    split_rows = _split_rows(payload)
    records: list[EpisodeRecord] = []
    total_steps = 0
    for shard in shards[:500]:
        try:
            steps = count_tfrecords(shard)
        except OSError as exc:
            notes.append(f"{shard.name}: {exc}")
            continue
        total_steps += steps
        records.append(EpisodeRecord(index=len(records), length=steps, path=shard.name, meta={"kind": "tfrecord-shard"}))
    if len(shards) > 500:
        notes.append(f"truncated shard listing at 500 of {len(shards)}")
    if not shards:
        notes.append("no TFRecord shards found next to the metadata")
    meta = {
        "dataset_name": payload.get("dataset_name"),
        "splits": split_rows,
        "steps": total_steps,
        "license": payload.get("license"),
        "description": payload.get("description"),
    }
    split = split_rows[0]["name"] if split_rows and split_rows[0].get("name") else None
    summary = DatasetSummary(
        path=str(root),
        format=FORMAT,
        version=str(payload.get("dataset_version") or "") or None,
        fps=None,
        robot_type=payload.get("robot_type") if isinstance(payload.get("robot_type"), str) else None,
        features=features,
        episodes=records,
        tasks=[],
        split=split,
        notes=notes,
        meta=meta,
        sources=[info.name] if info else [],
    )
    return RLDSDataset(root, summary)
