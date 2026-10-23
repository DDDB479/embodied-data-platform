"""MCAP adapter — the modern ROS 2 / Foxglove recording container.

Uses the official ``mcap`` package when installed. Without it, a compact
standard-library reader walks the record stream well enough to report
schemas, channels, message counts and the time window of a recording. Chunked
recordings are recognised and reported, but their inner messages are not
enumerated by the fallback reader (the official reader handles them).
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any, Iterator

from ..model import DatasetSummary, EpisodeRecord, FeatureSpec
from .base import DatasetObject, Detection, FramesUnavailable, guess_kind

MAGIC = b"\x89MCAP0\r\n"
_OP_HEADER = 0x01
_OP_FOOTER = 0x02
_OP_SCHEMA = 0x03
_OP_CHANNEL = 0x04
_OP_MESSAGE = 0x05
_OP_CHUNK = 0x06
_OP_STATISTICS = 0x0B
_OP_SUMMARY_OFFSET = 0x0E
_OP_DATA_END = 0x0F


def _read_uint16(buf: bytes, offset: int) -> tuple[int, int]:
    return struct.unpack_from("<H", buf, offset)[0], offset + 2


def _read_uint32(buf: bytes, offset: int) -> tuple[int, int]:
    return struct.unpack_from("<I", buf, offset)[0], offset + 4


def _read_uint64(buf: bytes, offset: int) -> tuple[int, int]:
    return struct.unpack_from("<Q", buf, offset)[0], offset + 8


def _read_string(buf: bytes, offset: int) -> tuple[str, int]:
    length, offset = _read_uint32(buf, offset)
    value = buf[offset : offset + length].decode("utf-8", errors="replace")
    return value, offset + length


def _read_map(buf: bytes, offset: int) -> tuple[dict[str, str], int]:
    byte_length, offset = _read_uint32(buf, offset)
    end = offset + byte_length
    out: dict[str, str] = {}
    while offset < end:
        key, offset = _read_string(buf, offset)
        value, offset = _read_string(buf, offset)
        out[key] = value
    return out, end


def _mcap_module() -> Any | None:
    try:
        import mcap.reader  # type: ignore

        return mcap
    except Exception:
        return None


def _mcap_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root] if root.suffix.lower() == ".mcap" else []
    return sorted(p for p in root.rglob("*.mcap") if p.is_file())


def detect(root: Path) -> Detection | None:
    files = _mcap_files(root)
    if not files:
        return None
    evidence = [f"{files[0].name}: MCAP magic"]
    try:
        with files[0].open("rb") as handle:
            head = handle.read(8)
    except OSError:
        return None
    if head != MAGIC:
        return None
    if len(files) > 1:
        evidence.append(f"{len(files)} mcap files")
    return Detection("mcap", version=None, confidence=0.95, evidence=evidence)


def _scan_fallback(path: Path) -> dict[str, Any]:
    """Minimal record walker used when the ``mcap`` package is unavailable."""
    info: dict[str, Any] = {
        "channels": {},
        "schemas": {},
        "counts": {},
        "first_time": None,
        "last_time": None,
        "chunked": False,
        "statistics": None,
    }
    with path.open("rb") as handle:
        head = handle.read(8)
        if head != MAGIC:
            raise ValueError("not an MCAP file")
        offset = 8
        size = path.stat().st_size
        while offset < size:
            handle.seek(offset)
            opcode = handle.read(1)
            if not opcode or opcode == MAGIC[:1]:
                break
            raw_len = handle.read(8)
            if len(raw_len) < 8:
                break
            length = struct.unpack("<Q", raw_len)[0]
            payload = handle.read(length)
            offset += 9 + length
            code = opcode[0]
            if code == _OP_SCHEMA:
                sid, pos = _read_uint16(payload, 0)
                name, pos = _read_string(payload, pos)
                encoding, pos = _read_string(payload, pos)
                info["schemas"][sid] = {"name": name, "encoding": encoding}
            elif code == _OP_CHANNEL:
                cid, pos = _read_uint16(payload, 0)
                _schema_id, pos = _read_uint16(payload, pos)
                topic, pos = _read_string(payload, pos)
                info["channels"][cid] = {"topic": topic}
                info["counts"].setdefault(topic, 0)
            elif code == _OP_MESSAGE:
                _cid, pos = _read_uint16(payload, 0)
                _seq, pos = _read_uint32(payload, pos)
                log_time, pos = _read_uint64(payload, pos)
                meta = info["channels"].get(_cid)
                if meta is not None:
                    info["counts"][meta["topic"]] = info["counts"].get(meta["topic"], 0) + 1
                info["first_time"] = log_time if info["first_time"] is None else min(info["first_time"], log_time)
                info["last_time"] = log_time if info["last_time"] is None else max(info["last_time"], log_time)
            elif code == _OP_CHUNK:
                info["chunked"] = True
            elif code == _OP_STATISTICS:
                pos = 8
                message_count, pos = _read_uint64(payload, pos)
                info["statistics"] = {"message_count": message_count}
                break
            elif code == _OP_DATA_END:
                break
    return info


def _read_with_mcap(path: Path) -> dict[str, Any]:
    from mcap.reader import make_reader  # type: ignore

    info: dict[str, Any] = {
        "channels": {},
        "schemas": {},
        "counts": {},
        "first_time": None,
        "last_time": None,
        "chunked": False,
        "statistics": None,
    }
    with path.open("rb") as handle:
        reader = make_reader(handle)
        summary = reader.get_summary()
        if summary is not None:
            if summary.statistics is not None:
                info["statistics"] = {
                    "message_count": summary.statistics.message_count,
                    "channel_message_counts": dict(summary.statistics.channel_message_counts),
                }
            for cid, channel in summary.channels.items():
                info["channels"][cid] = {"topic": channel.topic, "schema_id": channel.schema_id}
            for sid, schema in summary.schemas.items():
                info["schemas"][sid] = {"name": schema.name, "encoding": schema.encoding}
        start = None
        end = None
        counts: dict[str, int] = {meta["topic"]: 0 for meta in info["channels"].values()}
        for _schema, channel, message in reader.iter_messages():
            topic = channel.topic
            info["channels"].setdefault(channel.id, {"topic": topic})
            counts[topic] = counts.get(topic, 0) + 1
            start = message.log_time if start is None else min(start, message.log_time)
            end = message.log_time if end is None else max(end, message.log_time)
        info["counts"] = counts
        info["first_time"] = start
        info["last_time"] = end
    return info


class MCAPDataset(DatasetObject):
    def __init__(self, root: Path, summary: DatasetSummary) -> None:
        super().__init__(root, summary, capabilities={"streams", "episodes"})

    @staticmethod
    def optional_dependencies() -> dict[str, str]:
        return {"mcap": "full MCAP reading (fallback reader is built in)"}

    def frames(self, episodes=None, features=None):  # noqa: ANN001 - protocol signature
        raise FramesUnavailable("MCAP stores message streams; decode topics with their ROS/CDR schema first")


def load(root: Path, detection: Detection | None = None) -> MCAPDataset:
    files = _mcap_files(root)
    module = _mcap_module()
    records: list[EpisodeRecord] = []
    features: list[FeatureSpec] = []
    notes: list[str] = []
    meta: dict[str, Any] = {"files": [f.name for f in files]}
    total_counts: dict[str, int] = {}
    start_s: float | None = None
    end_s: float | None = None
    reader_name = "mcap-python" if module is not None else "stdlib-fallback"

    for file_path in files:
        try:
            info = _read_with_mcap(file_path) if module is not None else _scan_fallback(file_path)
        except Exception as exc:
            notes.append(f"{file_path.name}: {exc}")
            continue
        first = info.get("first_time")
        last = info.get("last_time")
        if first is not None and last is not None:
            first_s, last_s = first / 1e9, last / 1e9
            start_s = first_s if start_s is None else min(start_s, first_s)
            end_s = last_s if end_s is None else max(end_s, last_s)
            duration = max(0.0, last_s - first_s)
        else:
            duration = None
        for topic, count in (info.get("counts") or {}).items():
            total_counts[topic] = total_counts.get(topic, 0) + int(count)
        topics = sorted(info.get("counts") or {})
        hint = "chunked messages not enumerated by the fallback reader" if info.get("chunked") and module is None else None
        if hint:
            notes.append(f"{file_path.name}: {hint}")
        records.append(
            EpisodeRecord(
                index=len(records),
                length=sum(int(v) for v in (info.get("counts") or {}).values()) or None,
                duration_s=duration,
                start_time_s=(first / 1e9) if first is not None else None,
                path=file_path.name,
                meta={"topics": topics, "statistics": info.get("statistics"), "reader": reader_name},
            )
        )
    for topic in sorted(total_counts):
        features.append(FeatureSpec(key=topic, dtype="message", shape=(), kind=guess_kind(topic)))
    meta["topic_counts"] = total_counts
    meta["reader"] = reader_name
    if start_s is not None and end_s is not None:
        meta["time_window_s"] = [start_s, end_s]
    summary = DatasetSummary(
        path=str(root),
        format="mcap",
        version=None,
        fps=None,
        robot_type=None,
        features=features,
        episodes=records,
        tasks=[],
        notes=notes,
        meta=meta,
        sources=[f.name for f in files[:50]],
    )
    return MCAPDataset(root, summary)
