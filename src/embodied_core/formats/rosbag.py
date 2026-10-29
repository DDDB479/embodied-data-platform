"""ROS recording adapters.

* **rosbag2** — the ROS 2 default: a directory (or ``.db3`` file) whose SQLite
  database holds ``topics`` and ``messages`` tables. Pure standard library.
* **rosbag1** — the ROS 1 ``.bag`` container. The reader walks the record
  stream, reads connection records for topic names and chunk-info records for
  per-topic message counts and the time window. Payload deserialisation is not
  attempted: counts and timing need no message definitions.
"""

from __future__ import annotations

import sqlite3
import struct
from pathlib import Path
from typing import Any

from ..model import DatasetSummary, EpisodeRecord, FeatureSpec
from .base import DatasetObject, Detection, FramesUnavailable, guess_kind

ROS1_MAGIC = b"#ROSBAG V2.0\n"
_OP_MSG_DATA = 0x02
_OP_CONNECTION = 0x07
_OP_CHUNK_INFO = 0x06


# --------------------------------------------------------------------------
# detection
# --------------------------------------------------------------------------
def _ros2_dbs(root: Path) -> list[Path]:
    if root.is_file() and root.suffix.lower() == ".db3":
        return [root]
    if root.is_dir():
        return sorted(p for p in root.rglob("*.db3") if p.is_file())
    return []


def _ros1_bags(root: Path) -> list[Path]:
    if root.is_file() and root.suffix.lower() == ".bag":
        return [root]
    if root.is_dir():
        return sorted(p for p in root.rglob("*.bag") if p.is_file())
    return []


def _table_names(db: Path) -> set[str]:
    """Read the table list with an explicitly closed connection.

    ``with sqlite3.connect(...)`` is a transaction scope, not a close: on
    Windows the handle would keep the .db3 locked and break tempdir cleanup.
    """
    connection = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        connection.close()


def detect(root: Path) -> Detection | None:
    dbs = _ros2_dbs(root)
    if dbs:
        try:
            names = _table_names(dbs[0])
            if {"topics", "messages"} <= names:
                return Detection(
                    "rosbag2",
                    version="2",
                    confidence=0.97,
                    evidence=[f"{dbs[0].name}: SQLite tables topics/messages"],
                )
        except sqlite3.Error:
            pass
    bags = _ros1_bags(root)
    if bags:
        try:
            with bags[0].open("rb") as handle:
                head = handle.read(len(ROS1_MAGIC))
        except OSError:
            head = b""
        if head == ROS1_MAGIC:
            return Detection("rosbag1", version="2.0", confidence=0.97, evidence=[f"{bags[0].name}: ROS bag magic"])
    return None


# --------------------------------------------------------------------------
# rosbag2
# --------------------------------------------------------------------------
class Rosbag2Dataset(DatasetObject):
    def __init__(self, root: Path, summary: DatasetSummary) -> None:
        super().__init__(root, summary, capabilities={"streams", "episodes", "sqlite"})

    def frames(self, episodes=None, features=None, *, max_frames=None):  # noqa: ANN001
        raise FramesUnavailable("rosbag2 messages are CDR-encoded; decode a topic with its ROS message type first")

    def topics(self) -> list[dict[str, Any]]:
        return [f.extra for f in self.summary.features]


def _load_ros2(root: Path) -> DatasetSummary:
    dbs = _ros2_dbs(root)
    records: list[EpisodeRecord] = []
    features: list[FeatureSpec] = []
    notes: list[str] = []
    meta: dict[str, Any] = {"files": [p.name for p in dbs]}
    topic_rows: list[dict[str, Any]] = []
    for db in dbs:
        try:
            conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        except sqlite3.Error as exc:
            notes.append(f"{db.name}: {exc}")
            continue
        try:
            rows = list(conn.execute("SELECT id, name, type FROM topics"))
            counts = dict(conn.execute("SELECT topic_id, COUNT(*) FROM messages GROUP BY topic_id"))
            window = conn.execute("SELECT MIN(timestamp), MAX(timestamp) FROM messages").fetchone()
            for topic_id, name, type_name in rows:
                count = int(counts.get(topic_id, 0))
                topic_rows.append({"topic": name, "type": type_name, "messages": count})
                features.append(
                    FeatureSpec(
                        key=str(name),
                        dtype="message",
                        shape=(),
                        kind=guess_kind(str(name) if name else ""),
                        extra={"type": type_name, "messages": count},
                    )
                )
            start_ns, end_ns = window if window else (None, None)
            duration = ((end_ns - start_ns) / 1e9) if (start_ns is not None and end_ns is not None) else None
            records.append(
                EpisodeRecord(
                    index=len(records),
                    length=sum(int(v) for v in counts.values()) or None,
                    duration_s=duration,
                    start_time_s=(start_ns / 1e9) if start_ns is not None else None,
                    path=db.name,
                    meta={"topics": [r["topic"] for r in topic_rows]},
                )
            )
        except sqlite3.Error as exc:
            notes.append(f"{db.name}: {exc}")
        finally:
            conn.close()
    meta["topics"] = topic_rows
    return DatasetSummary(
        path=str(root),
        format="rosbag2",
        version="2",
        fps=None,
        robot_type=None,
        features=features,
        episodes=records,
        tasks=[],
        notes=notes,
        meta=meta,
        sources=[p.name for p in dbs[:50]],
    )


# --------------------------------------------------------------------------
# rosbag1
# --------------------------------------------------------------------------
def _parse_fields(payload: bytes) -> dict[str, bytes]:
    fields: dict[str, bytes] = {}
    offset = 0
    while offset + 4 <= len(payload):
        length = struct.unpack_from("<I", payload, offset)[0]
        offset += 4
        chunk = payload[offset : offset + length]
        offset += length
        name, _, value = chunk.partition(b"=")
        fields[name.decode("ascii", errors="replace")] = value
    return fields


def _scan_bag(path: Path) -> dict[str, Any]:
    topics: dict[int, dict[str, Any]] = {}
    counts: dict[int, int] = {}
    first_ns: int | None = None
    last_ns: int | None = None
    with path.open("rb") as handle:
        if handle.read(len(ROS1_MAGIC)) != ROS1_MAGIC:
            raise ValueError("not a ROS 1 bag")
        while True:
            head = handle.read(4)
            if len(head) < 4:
                break
            header_len = struct.unpack("<I", head)[0]
            header = handle.read(header_len)
            data_len_raw = handle.read(4)
            if len(data_len_raw) < 4:
                break
            data_len = struct.unpack("<I", data_len_raw)[0]
            fields = _parse_fields(header)
            op = fields.get("op", b"\x00")[0]
            if op == _OP_CONNECTION:
                conn_id = struct.unpack("<I", fields.get(b"conn", b"\x00\x00\x00\x00"))[0]
                data = handle.read(data_len)
                conn_fields = _parse_fields(data)
                topic = conn_fields.get(b"topic", b"").decode("utf-8", errors="replace")
                type_name = conn_fields.get(b"type", b"").decode("utf-8", errors="replace")
                topics[conn_id] = {"topic": topic, "type": type_name}
                counts.setdefault(conn_id, 0)
                continue
            if op == _OP_CHUNK_INFO:
                data = handle.read(data_len)
                offset = 0
                if len(data) >= 24:
                    start_ns, end_ns = struct.unpack_from("<QQ", data, 8)
                    count = struct.unpack_from("<I", data, 24)[0]
                    offset = 28
                    first_ns = start_ns if first_ns is None else min(first_ns, start_ns)
                    last_ns = end_ns if last_ns is None else max(last_ns, end_ns)
                    for _ in range(count):
                        if offset + 8 > len(data):
                            break
                        conn_id, conn_count = struct.unpack_from("<II", data, offset)
                        offset += 8
                        counts[conn_id] = counts.get(conn_id, 0) + int(conn_count)
                continue
            if op == _OP_MSG_DATA:
                if data_len and first_ns is None:
                    time_ns = struct.unpack("<Q", fields.get(b"time", b"\x00" * 8))[0] if b"time" in fields else None
                    conn_id = struct.unpack("<I", fields.get(b"conn", b"\x00\x00\x00\x00"))[0]
                    counts[conn_id] = counts.get(conn_id, 0) + 1
                    if time_ns:
                        first_ns = time_ns if first_ns is None else min(first_ns, time_ns)
                        last_ns = time_ns if last_ns is None else max(last_ns, time_ns)
                handle.seek(data_len, 1)
                continue
            handle.seek(data_len, 1)
    return {"topics": topics, "counts": counts, "first_ns": first_ns, "last_ns": last_ns}


class Rosbag1Dataset(DatasetObject):
    def __init__(self, root: Path, summary: DatasetSummary) -> None:
        super().__init__(root, summary, capabilities={"streams", "episodes"})

    def frames(self, episodes=None, features=None, *, max_frames=None):  # noqa: ANN001
        raise FramesUnavailable("rosbag1 payloads are ROS-serialised; decode a topic with its message definition first")


def _load_ros1(root: Path) -> DatasetSummary:
    bags = _ros1_bags(root)
    records: list[EpisodeRecord] = []
    features: list[FeatureSpec] = []
    notes: list[str] = []
    topic_rows: list[dict[str, Any]] = []
    meta: dict[str, Any] = {"files": [p.name for p in bags]}
    for bag in bags:
        try:
            info = _scan_bag(bag)
        except Exception as exc:
            notes.append(f"{bag.name}: {exc}")
            continue
        counts: dict[int, int] = info["counts"]
        for conn_id, topic in info["topics"].items():
            count = int(counts.get(conn_id, 0))
            topic_rows.append({"topic": topic["topic"], "type": topic["type"], "messages": count})
            features.append(
                FeatureSpec(
                    key=topic["topic"],
                    dtype="message",
                    shape=(),
                    kind=guess_kind(topic["topic"]),
                    extra={"type": topic["type"], "messages": count},
                )
            )
        first_ns, last_ns = info["first_ns"], info["last_ns"]
        duration = ((last_ns - first_ns) / 1e9) if (first_ns is not None and last_ns is not None) else None
        records.append(
            EpisodeRecord(
                index=len(records),
                length=sum(int(v) for v in counts.values()) or None,
                duration_s=duration,
                start_time_s=(first_ns / 1e9) if first_ns is not None else None,
                path=bag.name,
                meta={"topics": list(info["topics"].values())},
            )
        )
    if not records:
        notes.append("no readable rosbag1 files")
    meta["topics"] = topic_rows
    return DatasetSummary(
        path=str(root),
        format="rosbag1",
        version="2.0",
        fps=None,
        robot_type=None,
        features=features,
        episodes=records,
        tasks=[],
        notes=notes,
        meta=meta,
        sources=[p.name for p in bags[:50]],
    )


def load(root: Path, detection: Detection | None = None) -> DatasetObject:
    fmt = (detection.format if detection else None) or ""
    if fmt == "rosbag2" or _ros2_dbs(root):
        return Rosbag2Dataset(root, _load_ros2(root))
    return Rosbag1Dataset(root, _load_ros1(root))
