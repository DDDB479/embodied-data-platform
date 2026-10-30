"""MCAP adapter — the modern ROS 2 / Foxglove recording container.

Uses the official ``mcap`` package when installed. Without it, a compact
standard-library reader walks the record stream well enough to report
schemas, channels, message counts and the time window of a recording. Chunked
recordings are recognised and reported, but their inner messages are not
enumerated by the fallback reader (the official reader handles them).
"""

from __future__ import annotations

import struct
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from ..model import DatasetSummary, EpisodeRecord, FeatureSpec
from ..ros2_codec import (
    CDRError,
    UnsupportedMessageType,
    decode_message,
    flatten_decoded,
    topic_feature_keys,
)
from .base import (
    DatasetObject,
    Detection,
    FrameChunk,
    FrameLimitExceeded,
    FramesUnavailable,
    collect_frame_chunks,
    guess_kind,
    resolve_chunk_frames,
    resolve_max_frames,
)

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
                schema_id, pos = _read_uint16(payload, pos)
                topic, pos = _read_string(payload, pos)
                info["channels"][cid] = {"topic": topic, "schema_id": schema_id}
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


def _select_topics(types: dict[str, str], features: Sequence[str] | None) -> list[str]:
    candidates = [topic for topic, type_name in types.items() if topic_feature_keys(type_name)]
    if features is None:
        return sorted(candidates)
    selected: list[str] = []
    for requested in features:
        for topic in candidates:
            if topic == requested or topic.startswith(f"{requested}.") or requested.startswith(f"{topic}."):
                if topic not in selected:
                    selected.append(topic)
    return selected


class MCAPDataset(DatasetObject):
    def __init__(self, root: Path, summary: DatasetSummary) -> None:
        capabilities = {"streams", "episodes"}
        if _mcap_module() is not None and summary.meta.get("topic_types"):
            capabilities.update({"frames", "cdr"})
        super().__init__(root, summary, capabilities=capabilities)

    @staticmethod
    def optional_dependencies() -> dict[str, str]:
        return {"mcap": "full MCAP reading and CDR message decoding (fallback metadata reader is built in)"}

    def frames(
        self,
        episodes: Sequence[int] | None = None,
        features: Sequence[str] | None = None,
        *,
        max_frames: int | None = None,
    ) -> tuple[list[int], dict[str, np.ndarray]]:
        return collect_frame_chunks(
            self.iter_frames(episodes, features, max_frames=max_frames),
            max_frames=max_frames,
        )

    def iter_frames(
        self,
        episodes: Sequence[int] | None = None,
        features: Sequence[str] | None = None,
        *,
        chunk_frames: int | None = None,
        max_frames: int | None = None,
    ) -> Iterator[FrameChunk]:
        module = _mcap_module()
        if module is None:
            raise FramesUnavailable(
                "MCAP frame decoding requires the mcap package",
                install="pip install 'embodied-core[mcap]'",
            )
        from mcap.reader import make_reader  # type: ignore

        encodings = {str(key): str(value).lower() for key, value in (self.summary.meta.get("topic_encodings") or {}).items()}
        types = {
            str(key): str(value)
            for key, value in (self.summary.meta.get("topic_types") or {}).items()
            if encodings.get(str(key), "cdr") in ("", "cdr", "ros2", "ros2msg")
        }
        selected = _select_topics(types, features)
        if not selected:
            raise FramesUnavailable("no CDR-decodable MCAP topics matched the request")
        wanted = set(episodes) if episodes is not None else None
        size = resolve_chunk_frames(chunk_frames)
        limit = resolve_max_frames(max_frames)
        files = _mcap_files(self.root)
        total = 0
        emitted = False
        seen: dict[int, int] = {}
        for episode_index, path in enumerate(files):
            if wanted is not None and episode_index not in wanted:
                continue
            handles = [path.open("rb") for _ in selected]
            try:
                readers = [make_reader(handle) for handle in handles]
                generators = {
                    topic: readers[position].iter_messages(topics=[topic], log_time_order=True)
                    for position, topic in enumerate(selected)
                }
                finished: set[str] = set()
                while len(finished) < len(selected):
                    buckets: dict[str, list[Any]] = {}
                    for topic in selected:
                        if topic in finished:
                            buckets[topic] = []
                            continue
                        rows = []
                        for _ in range(size):
                            try:
                                rows.append(next(generators[topic]))
                            except StopIteration:
                                finished.add(topic)
                                break
                        buckets[topic] = rows
                    if not any(buckets.values()):
                        if len(finished) >= len(selected):
                            break
                        continue
                    count = max(len(rows) for rows in buckets.values())
                    merged: dict[str, list[np.ndarray]] = defaultdict(list)
                    for topic, rows in buckets.items():
                        for _schema, _channel, message in rows:
                            try:
                                decoded = decode_message(types.get(topic, ""), message.data)
                                flat = flatten_decoded(decoded, topic)
                            except (CDRError, UnsupportedMessageType, ValueError):
                                continue
                            for key, values in flat.items():
                                merged[key].append(np.asarray(values, dtype=np.float64).reshape(-1))
                    if not merged:
                        continue
                    columns: dict[str, np.ndarray] = {}
                    for key, values in merged.items():
                        width = max((item.size for item in values), default=1)
                        matrix = np.full((count, width), np.nan, dtype=np.float64)
                        for row_index, value in enumerate(values[:count]):
                            matrix[row_index, : value.size] = value
                        columns[key] = matrix
                    total += count
                    if limit and total > limit:
                        raise FrameLimitExceeded(limit, total)
                    frame_ids = [episode_index] * count
                    offsets = {episode_index: seen.get(episode_index, 0)}
                    seen[episode_index] = seen.get(episode_index, 0) + count
                    emitted = True
                    yield FrameChunk(frame_ids, columns, offsets)
            finally:
                for handle in handles:
                    handle.close()
        if not emitted:
            raise FramesUnavailable("no MCAP messages decoded from the selected CDR topics")

    def topic_samples(
        self,
        topic: str,
        *,
        episode: int | None = None,
        limit: int = 5,
        include_arrays: bool = False,
    ) -> dict[str, Any]:
        module = _mcap_module()
        if module is None:
            raise FramesUnavailable(
                "MCAP message samples require the mcap package",
                install="pip install 'embodied-core[mcap]'",
            )
        from mcap.reader import make_reader  # type: ignore

        types = {str(key): str(value) for key, value in (self.summary.meta.get("topic_types") or {}).items()}
        type_name = types.get(topic)
        if not type_name:
            raise KeyError(f"unknown topic {topic!r}")
        encoding = str((self.summary.meta.get("topic_encodings") or {}).get(topic) or "").lower()
        if encoding not in ("", "cdr", "ros2", "ros2msg"):
            raise FramesUnavailable(f"topic {topic!r} uses message encoding {encoding!r}; only CDR is decoded")
        files = _mcap_files(self.root)
        if episode is not None:
            if episode < 0 or episode >= len(files):
                raise ValueError(f"episode {episode} is out of range")
            files = [files[episode]]
        samples = []
        for path in files:
            with path.open("rb") as handle:
                reader = make_reader(handle)
                for _schema, _channel, message in reader.iter_messages(topics=[topic], log_time_order=True):
                    decoded = decode_message(type_name, message.data)
                    body = decoded.to_dict(include_arrays=include_arrays)
                    body["log_time_ns"] = int(message.log_time)
                    samples.append(body)
                    if len(samples) >= max(1, int(limit)):
                        break
        return {"topic": topic, "type": type_name, "samples": samples}


def load(root: Path, detection: Detection | None = None) -> MCAPDataset:
    files = _mcap_files(root)
    module = _mcap_module()
    records: list[EpisodeRecord] = []
    features: list[FeatureSpec] = []
    notes: list[str] = []
    meta: dict[str, Any] = {"files": [f.name for f in files]}
    total_counts: dict[str, int] = {}
    topic_types: dict[str, str] = {}
    topic_encodings: dict[str, str] = {}
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
        schemas = info.get("schemas") or {}
        for channel in (info.get("channels") or {}).values():
            topic = str(channel.get("topic") or "")
            schema = schemas.get(channel.get("schema_id")) or {}
            if topic and schema.get("name"):
                topic_types[topic] = str(schema["name"])
                topic_encodings[topic] = str(schema.get("encoding") or "")
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
        type_name = topic_types.get(topic, "")
        decoded_features = topic_feature_keys(type_name)
        features.append(
            FeatureSpec(
                key=topic,
                dtype="message",
                shape=(),
                kind=guess_kind(topic),
                extra={
                    "type": type_name or None,
                    "encoding": topic_encodings.get(topic) or None,
                    "messages": total_counts[topic],
                    "decoded_features": decoded_features,
                    "cdr_decodable": bool(decoded_features)
                    and str(topic_encodings.get(topic) or "").lower() in ("", "cdr", "ros2", "ros2msg"),
                },
            )
        )
    meta["topic_counts"] = total_counts
    meta["topic_types"] = topic_types
    meta["topic_encodings"] = topic_encodings
    meta["topics"] = [
        {
            "topic": topic,
            "type": topic_types.get(topic),
            "encoding": topic_encodings.get(topic),
            "messages": total_counts[topic],
            "decoded_features": topic_feature_keys(topic_types.get(topic, "")),
        }
        for topic in sorted(total_counts)
    ]
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
