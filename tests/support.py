"""Fixture builders shared by the test suite.

Everything is generated at test time with the standard library plus whatever
optional dependency is present, so the suite runs on a bare checkout (the
optional-dependency tests skip instead of failing).
"""
from __future__ import annotations

import importlib.util
import json
import sqlite3
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
