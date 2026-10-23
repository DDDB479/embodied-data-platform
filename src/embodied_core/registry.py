"""Adapter registry: detection order, dispatch and format documentation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .formats import hdf5, jsonl, lerobot, mcap, rlds, rosbag
from .formats.base import DatasetObject, Detection
from .model import DatasetSummary

# Order matters: the most specific layouts are probed first (LeRobot has a
# ``meta/*.json`` marker, HDF5 and MCAP are magic-byte checks, rosbag2 is a
# SQLite schema check, and JSONL is the most permissive text format).
_ADAPTERS = [lerobot, hdf5, mcap, rosbag, rlds, jsonl]

FORMAT_HELP: list[dict[str, Any]] = [
    {
        "format": "lerobot",
        "description": "LeRobot datasets, codebase_version 2.x (one parquet per episode) and 3.x (chunked file-*.parquet + meta/episodes/**.parquet)",
        "detect": "meta/info.json plus meta/episodes or data/**.parquet",
        "extensions": [".parquet", ".json", ".jsonl", ".mp4"],
        "needs": "pyarrow (parquet data and episode metadata)",
        "can_read_frames": True,
    },
    {
        "format": "hdf5",
        "description": "HDF5 / robomimic style files where each demo_* group (or generic group) is an episode",
        "detect": "HDF5 magic bytes",
        "extensions": [".hdf5", ".h5", ".hdf"],
        "needs": "h5py",
        "can_read_frames": True,
    },
    {
        "format": "mcap",
        "description": "MCAP recordings with schemas, channels and per-topic message counts",
        "detect": "MCAP magic bytes",
        "extensions": [".mcap"],
        "needs": "mcap (a stdlib fallback reader is built in)",
        "can_read_frames": False,
    },
    {
        "format": "rosbag2",
        "description": "ROS 2 rosbag2 SQLite recordings (topics + messages)",
        "detect": "*.db3 with topics/messages tables",
        "extensions": [".db3", ".mcap"],
        "needs": "none (sqlite3)",
        "can_read_frames": False,
    },
    {
        "format": "rosbag1",
        "description": "ROS 1 .bag containers with per-connection message counts",
        "detect": "#ROSBAG V2.0 magic",
        "extensions": [".bag"],
        "needs": "none",
        "can_read_frames": False,
    },
    {
        "format": "rlds",
        "description": "RLDS / TFDS datasets: dataset_info.json plus TFRecord shards, records counted per shard",
        "detect": "dataset_info.json or features.json next to shards",
        "extensions": [".tfrecord", ".json"],
        "needs": "tensorflow only to decode example protos",
        "can_read_frames": False,
    },
    {
        "format": "embodied-jsonl",
        "description": "The studio's canonical JSONL interchange layout (frames or columns per episode)",
        "detect": "dataset.json format marker or JSONL episode lines",
        "extensions": [".jsonl", ".json"],
        "needs": "none",
        "can_read_frames": True,
    },
]


class UnsupportedFormat(LookupError):
    """Raised when no adapter recognises a path."""

    def __init__(self, path: Path) -> None:
        self.path = path
        known = ", ".join(item["format"] for item in FORMAT_HELP)
        super().__init__(f"no adapter recognised {path} (supported: {known})")


#: Format names that users (or detection) may use, mapped to adapter modules.
_ALIASES = {
    "lerobot": "lerobot",
    "hdf5": "hdf5",
    "h5": "hdf5",
    "robomimic": "hdf5",
    "mcap": "mcap",
    "rosbag": "rosbag",
    "rosbag1": "rosbag",
    "rosbag2": "rosbag",
    "bag": "rosbag",
    "rlds": "rlds",
    "tfds": "rlds",
    "jsonl": "jsonl",
    "embodied-jsonl": "jsonl",
    "canonical": "jsonl",
}


def _adapter_for(name: str):
    target = _ALIASES.get(name.strip().lower(), name.strip().lower())
    for module in _ADAPTERS:
        if module.__name__.rpartition(".")[2] == target:
            return module
    raise KeyError(f"unknown format {name!r}")


def detect_dataset(path: str | Path) -> Detection:
    """Return the most confident :class:`Detection` for ``path``."""
    root = Path(path).expanduser()
    if not root.exists():
        raise FileNotFoundError(root)
    candidates: list[Detection] = []
    for module in _ADAPTERS:
        try:
            found = module.detect(root)
        except Exception:  # a broken adapter must not hide the others
            found = None
        if found is not None:
            candidates.append(found)
    if not candidates:
        raise UnsupportedFormat(root)
    candidates.sort(key=lambda det: det.confidence, reverse=True)
    return candidates[0]


def open_dataset(path: str | Path, format: str | None = None) -> DatasetObject:
    """Detect (unless ``format`` is given) and load a dataset."""
    root = Path(path).expanduser()
    if format:
        module = _adapter_for(format)
        try:
            detection = module.detect(root)
        except Exception:
            detection = None
        return module.load(root, detection)
    detection = detect_dataset(root)
    module = _adapter_for(detection.format)
    return module.load(root, detection)


def describe_formats() -> list[dict[str, Any]]:
    return [dict(item) for item in FORMAT_HELP]


def empty_summary(path: str | Path, format: str = "unknown") -> DatasetSummary:
    return DatasetSummary(path=str(path), format=format)
