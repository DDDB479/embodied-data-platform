"""Fixture builders shared by the test suite.

Everything is generated at test time with the standard library plus whatever
optional dependency is present, so the suite runs on a bare checkout (the
optional-dependency tests skip instead of failing).
"""
from __future__ import annotations

import importlib.util
import json
import sqlite3
import struct
from pathlib import Path

import numpy as np

FPS = 30.0


def has_module(name: str) -> bool:
    """True when an optional reader is importable in this environment."""
    return importlib.util.find_spec(name) is not None


def has_pyarrow() -> bool:
    return has_module("pyarrow")


def _episode_rows(index: int, length: int = 30, offset: float = 0.0, spikes: bool = False) -> list[dict]:
    rows = []
    for frame in range(length):
        phase = frame / max(length - 1, 1)
        # A spike is a single-frame teleport in the middle of the episode; it is
        # far outside the MAD band, so anomaly detection must find it.
        jump = 50.0 if (spikes and frame == length // 2) else 0.0
        rows.append(
            {
                "observation.state": [float(offset + phase * 0.5 + jump), 0.1, 0.2, 0.0, 0.0, 0.0],
                "action": [float(offset + phase * 0.4 + jump), 0.1, 0.2, 0.0, 0.0, 0.0],
                "frame_index": frame,
                "episode_index": index,
                "timestamp": frame / FPS,
                "task": "pick up the cube",
            }
        )
    return rows


def write_jsonl_dataset(root: Path, episodes: int = 2, length: int = 30, spikes: bool = False) -> Path:
    """Canonical JSONL layout: dataset.json + episodes.jsonl."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "dataset.json").write_text(
        json.dumps(
            {
                "format": "embodied-jsonl",
                "version": "1.0",
                "fps": FPS,
                "robot_type": "so100",
                "tasks": ["pick up the cube"],
                "features": {
                    "observation.state": {"dtype": "float32", "shape": [6]},
                    "action": {"dtype": "float32", "shape": [6]},
                },
            }
        ),
        encoding="utf-8",
    )
    with (root / "episodes.jsonl").open("w", encoding="utf-8") as handle:
        for index in range(episodes):
            episode = {"episode_index": index, "frames": _episode_rows(index, length, offset=index * 0.1, spikes=spikes)}
            handle.write(json.dumps(episode) + "\n")
    return root


def write_lerobot_dataset(root: Path, episodes: int = 2, length: int = 30) -> Path | None:
    """LeRobot v2.1 layout; returns None when pyarrow is unavailable."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        return None

    root = root / "lerobot"
    (root / "meta").mkdir(parents=True, exist_ok=True)
    total = 0
    episode_rows = []
    for index in range(episodes):
        rows = _episode_rows(index, length)
        columns = {
            "observation.state": [np.asarray(row["observation.state"], dtype=np.float32) for row in rows],
            "action": [np.asarray(row["action"], dtype=np.float32) for row in rows],
            "timestamp": [np.float32(row["timestamp"]) for row in rows],
            "frame_index": [np.int64(row["frame_index"]) for row in rows],
            "episode_index": [np.int64(row["episode_index"]) for row in rows],
            "index": [np.int64(total + i) for i in range(len(rows))],
            "task_index": [np.int64(0) for _ in rows],
        }
        table = pa.table(
            {
                key: pa.array(
                    values,
                    type=pa.list_(pa.float32(), len(values[0])) if isinstance(values[0], np.ndarray) else None,
                )
                for key, values in columns.items()
            }
        )
        target = root / "data" / "chunk-000" / f"episode_{index:06d}.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, target)
        total += len(rows)
        episode_rows.append({"episode_index": index, "length": len(rows), "tasks": ["pick up the cube"], "task": "pick up the cube", "data/chunk_index": 0, "data/file_index": index})

    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v2.1",
                "robot_type": "so100",
                "fps": FPS,
                "total_episodes": episodes,
                "total_frames": total,
                "total_tasks": 1,
                "splits": {"train": f"0:{episodes}"},
                "features": {
                    "observation.state": {"dtype": "float32", "shape": [6], "names": None},
                    "action": {"dtype": "float32", "shape": [6], "names": None},
                },
            }
        ),
        encoding="utf-8",
    )
    with (root / "meta" / "episodes.jsonl").open("w", encoding="utf-8") as handle:
        for row in episode_rows:
            handle.write(json.dumps(row) + "\n")
    with (root / "meta" / "tasks.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"task_index": 0, "task": "pick up the cube"}) + "\n")
    return root


def write_hdf5_dataset(root: Path, demos: int = 2, length: int = 25) -> Path | None:
    """robomimic-style HDF5; returns None when h5py is unavailable."""
    try:
        import h5py
    except ImportError:
        return None

    root = root / "hdf5"
    root.mkdir(parents=True, exist_ok=True)
    path = root / "demo.hdf5"
    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        data.attrs["env_args"] = json.dumps({"env_name": "Lift", "control_freq": 20})
        for index in range(demos):
            demo = data.create_group(f"demo_{index}")
            states = np.linspace(0, 1, length, dtype=np.float64)[:, None] * np.ones((1, 7))
            demo.create_dataset("obs/object", data=states)
            demo.create_dataset("actions", data=states * 0.5)
            demo.create_dataset("rewards", data=np.zeros((length, 1)))
            demo.attrs["num_samples"] = length
    return root


def write_rosbag2(root: Path, messages: int = 40) -> Path:
    """rosbag2 SQLite layout: topics + messages."""
    root = root / "rosbag2"
    root.mkdir(parents=True, exist_ok=True)
    path = root / "bag.db3"
    connection = sqlite3.connect(path)
    try:
        connection.execute("CREATE TABLE topics (id INTEGER PRIMARY KEY, name TEXT, type TEXT, serialization_format TEXT)")
        connection.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, topic_id INTEGER, timestamp INTEGER, data BLOB)")
        connection.execute("INSERT INTO topics VALUES (1, '/joint_states', 'sensor_msgs/msg/JointState', 'cdr')")
        for index in range(messages):
            connection.execute(
                "INSERT INTO messages (topic_id, timestamp, data) VALUES (?, ?, ?)",
                (1, 1_600_000_000_000_000_000 + index * 33_000_000, b"\x00\x00"),
            )
        connection.commit()
    finally:
        connection.close()
    return root


def write_mcap(root: Path, messages: int = 20) -> Path | None:
    """MCAP file; returns None when the mcap library is unavailable."""
    try:
        from mcap.writer import Writer
    except ImportError:
        return None
    root = root / "mcap"
    root.mkdir(parents=True, exist_ok=True)
    path = root / "episode.mcap"
    with path.open("wb") as handle:
        writer = Writer(handle)
        writer.start()
        schema_id = writer.register_schema(name="std_msgs/Float64", encoding="ros1msg", data=b"float64 data")
        channel_id = writer.register_channel(topic="/cmd_vel", message_encoding="ros1", schema_id=schema_id)
        for index in range(messages):
            writer.add_message(
                channel_id=channel_id,
                log_time=1_600_000_000_000_000_000 + index * 20_000_000,
                publish_time=1_600_000_000_000_000_000 + index * 20_000_000,
                data=b"\x00" * 8,
            )
        writer.finish()
    return root


def write_rlds(root: Path, steps: int = 3) -> Path:
    """RLDS metadata + one minimally valid TFRecord shard."""
    root = root / "rlds"
    root.mkdir(parents=True, exist_ok=True)
    (root / "dataset_info.json").write_text(
        json.dumps(
            {
                "dataset_name": "toy",
                "dataset_version": "1.0.0",
                "splits": [{"name": "train", "shardLengths": [str(steps)], "numBytes": "0"}],
                "features": {
                    "steps": {
                        "sequence": {
                            "observation": {"features": {"state": {"shape": [7], "dtype": "tf.float32"}}},
                            "action": {"shape": [7], "dtype": "tf.float32"},
                            "reward": {"shape": [], "dtype": "tf.float32"},
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    payload = b"toy-record"
    frame = len(payload).to_bytes(8, "little") + (0).to_bytes(4, "little") + payload + (0).to_bytes(4, "little")
    (root / "toy-train.tfrecord-00000-of-00001").write_bytes(frame)
    return root


# --------------------------------------------------------------------------
# CDR / ROS 2 protocol fixtures
# --------------------------------------------------------------------------
def _cdr_start() -> bytearray:
    return bytearray(b"\x00\x01\x00\x00")


def _cdr_align(buf: bytearray, size: int) -> None:
    relative = len(buf) - 4
    buf.extend(b"\x00" * ((-relative) % size))


def _cdr_u32(buf: bytearray, value: int) -> None:
    _cdr_align(buf, 4)
    buf.extend(struct.pack("<I", value))


def _cdr_i32(buf: bytearray, value: int) -> None:
    _cdr_align(buf, 4)
    buf.extend(struct.pack("<i", value))


def _cdr_f64(buf: bytearray, value: float) -> None:
    _cdr_align(buf, 8)
    buf.extend(struct.pack("<d", value))


def _cdr_string(buf: bytearray, value: str) -> None:
    raw = value.encode("utf-8") + b"\x00"
    _cdr_align(buf, 4)
    buf.extend(struct.pack("<I", len(raw)))
    buf.extend(raw)


def _cdr_header(buf: bytearray, frame_id: str = "base") -> None:
    _cdr_i32(buf, 0)
    _cdr_u32(buf, 0)
    _cdr_string(buf, frame_id)


def encode_cdr_joint_state(*, positions: list[float], names: list[str] | None = None) -> bytes:
    buf = _cdr_start()
    names = names or [f"joint_{index}" for index in range(len(positions))]
    _cdr_header(buf)
    _cdr_u32(buf, len(names))
    for name in names:
        _cdr_string(buf, name)
    _cdr_u32(buf, len(positions))
    for value in positions:
        _cdr_f64(buf, float(value))
    _cdr_u32(buf, 0)
    _cdr_u32(buf, 0)
    return bytes(buf)


def encode_cdr_string(value: str) -> bytes:
    buf = _cdr_start()
    _cdr_string(buf, value)
    return bytes(buf)


def encode_cdr_depth_image(width: int = 4, height: int = 3) -> bytes:
    buf = _cdr_start()
    _cdr_header(buf, "camera/depth")
    _cdr_u32(buf, height)
    _cdr_u32(buf, width)
    _cdr_string(buf, "32FC1")
    buf.append(0)
    _cdr_u32(buf, width * 4)
    pixels = np.arange(width * height, dtype=np.float32).reshape(height, width) / 10.0
    raw = pixels.astype("<f4").tobytes()
    _cdr_u32(buf, len(raw))
    buf.extend(raw)
    return bytes(buf)


def encode_cdr_pointcloud2(point_count: int = 4) -> bytes:
    buf = _cdr_start()
    _cdr_header(buf, "lidar")
    _cdr_u32(buf, 1)
    _cdr_u32(buf, point_count)
    _cdr_u32(buf, 3)
    for name, offset in (("x", 0), ("y", 4), ("z", 8)):
        _cdr_string(buf, name)
        _cdr_u32(buf, offset)
        buf.append(7)  # FLOAT32
        _cdr_u32(buf, 1)
    buf.append(0)
    point_step = 12
    row_step = point_step * point_count
    _cdr_u32(buf, point_step)
    _cdr_u32(buf, row_step)
    points = np.stack(
        [np.arange(point_count, dtype=np.float32), np.arange(point_count, dtype=np.float32) * 2, np.ones(point_count, dtype=np.float32)],
        axis=1,
    )
    raw = points.astype("<f4").tobytes()
    _cdr_u32(buf, len(raw))
    buf.extend(raw)
    buf.append(1)
    return bytes(buf)


def write_rosbag2_cdr(root: Path, messages: int = 12) -> Path:
    """rosbag2 with real CDR JointState, String and depth topics."""

    root = root / "rosbag2_cdr"
    root.mkdir(parents=True, exist_ok=True)
    path = root / "bag.db3"
    connection = sqlite3.connect(path)
    try:
        connection.execute("CREATE TABLE topics (id INTEGER PRIMARY KEY, name TEXT, type TEXT, serialization_format TEXT)")
        connection.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, topic_id INTEGER, timestamp INTEGER, data BLOB)")
        connection.executemany(
            "INSERT INTO topics VALUES (?, ?, ?, 'cdr')",
            [
                (1, "/joint_states", "sensor_msgs/msg/JointState"),
                (2, "/instruction", "std_msgs/msg/String"),
                (3, "/depth", "sensor_msgs/msg/Image"),
                (4, "/points", "sensor_msgs/msg/PointCloud2"),
            ],
        )
        for index in range(messages):
            timestamp = 1_600_000_000_000_000_000 + index * 33_000_000
            connection.execute(
                "INSERT INTO messages (topic_id, timestamp, data) VALUES (?, ?, ?)",
                (1, timestamp, encode_cdr_joint_state(positions=[float(index), 0.1, 0.2])),
            )
            if index == 0:
                connection.execute(
                    "INSERT INTO messages (topic_id, timestamp, data) VALUES (?, ?, ?)",
                    (2, timestamp, encode_cdr_string("pick up the red cube")),
                )
            if index % 3 == 0:
                connection.execute(
                    "INSERT INTO messages (topic_id, timestamp, data) VALUES (?, ?, ?)",
                    (3, timestamp, encode_cdr_depth_image()),
                )
                connection.execute(
                    "INSERT INTO messages (topic_id, timestamp, data) VALUES (?, ?, ?)",
                    (4, timestamp, encode_cdr_pointcloud2()),
                )
        connection.commit()
    finally:
        connection.close()
    return root


def write_mcap_cdr(root: Path, messages: int = 10) -> Path | None:
    """MCAP with a CDR JointState channel; returns None without mcap."""

    try:
        from mcap.writer import Writer
    except ImportError:
        return None
    root = root / "mcap_cdr"
    root.mkdir(parents=True, exist_ok=True)
    path = root / "episode.mcap"
    with path.open("wb") as handle:
        writer = Writer(handle)
        writer.start()
        schema_id = writer.register_schema(
            name="sensor_msgs/msg/JointState",
            encoding="ros2msg",
            data=b"built-in fixture schema",
        )
        channel_id = writer.register_channel(topic="/joint_states", message_encoding="cdr", schema_id=schema_id)
        for index in range(messages):
            writer.add_message(
                channel_id=channel_id,
                log_time=1_600_000_000_000_000_000 + index * 33_000_000,
                publish_time=1_600_000_000_000_000_000 + index * 33_000_000,
                data=encode_cdr_joint_state(positions=[float(index), 1.0]),
            )
        writer.finish()
    return root


# --------------------------------------------------------------------------
# RLDS protobuf fixtures
# --------------------------------------------------------------------------
def _varint(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _pb_field(field: int, wire: int, payload: bytes | int) -> bytes:
    key = _varint((field << 3) | wire)
    if wire == 2:
        return key + _varint(len(payload)) + payload  # type: ignore[arg-type]
    return key + _varint(int(payload))


def _pb_float_feature(values: list[float]) -> bytes:
    packed = b"".join(struct.pack("<f", float(value)) for value in values)
    float_list = _pb_field(1, 2, packed)
    return _pb_field(2, 2, float_list)


def _pb_bytes_feature(values: list[bytes]) -> bytes:
    bytes_list = b"".join(_pb_field(1, 2, value) for value in values)
    return _pb_field(1, 2, bytes_list)


def _pb_feature_map(features: dict[str, bytes]) -> bytes:
    entries = []
    for key, feature in features.items():
        entry = _pb_field(1, 2, key.encode("utf-8")) + _pb_field(2, 2, feature)
        entries.append(_pb_field(1, 2, entry))
    return b"".join(entries)


def _pb_feature_list(features: list[bytes]) -> bytes:
    return b"".join(_pb_field(1, 2, feature) for feature in features)


def _pb_feature_lists(features: dict[str, list[bytes]]) -> bytes:
    entries = []
    for key, values in features.items():
        entry = _pb_field(1, 2, key.encode("utf-8")) + _pb_field(2, 2, _pb_feature_list(values))
        entries.append(_pb_field(1, 2, entry))
    return b"".join(entries)


def write_rlds_sequence(root: Path, episodes: int = 2, steps: int = 5) -> Path:
    """Create RLDS TFRecords using tensorflow.SequenceExample protos."""

    root = root / "rlds_sequence"
    root.mkdir(parents=True, exist_ok=True)
    features = {
        "steps": {
            "sequence": {
                "feature": {
                    "observation": {"features": {"state": {"shape": [3], "dtype": "tf.float32"}}},
                    "action": {"shape": [3], "dtype": "tf.float32"},
                    "timestamp": {"shape": [], "dtype": "tf.float32"},
                    "language_instruction": {"shape": [], "dtype": "tf.string"},
                }
            }
        }
    }
    (root / "dataset_info.json").write_text(
        json.dumps(
            {
                "dataset_name": "sequence-fixture",
                "dataset_version": "1.0.0",
                "splits": [{"name": "train", "shardLengths": [str(steps * episodes)]}],
                "features": features,
            }
        ),
        encoding="utf-8",
    )
    frames = []
    for episode in range(episodes):
        feature_lists = {
            "steps/observation/state": [_pb_float_feature([float(episode), float(step), 0.5]) for step in range(steps)],
            "steps/action": [_pb_float_feature([float(step) * 0.1, 0.2, 0.3]) for step in range(steps)],
            "steps/timestamp": [_pb_float_feature([step / 10.0]) for step in range(steps)],
            "steps/language_instruction": [_pb_bytes_feature([b"pick up the cube"]) for _ in range(steps)],
        }
        sequence = _pb_field(2, 2, _pb_feature_lists(feature_lists))
        frame = len(sequence).to_bytes(8, "little") + (0).to_bytes(4, "little") + sequence + (0).to_bytes(4, "little")
        frames.append(frame)
    (root / "fixture-train.tfrecord-00000-of-00001").write_bytes(b"".join(frames))
    return root
