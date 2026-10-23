"""Generate on-disk fixtures for every supported format (no dataset downloads)."""

from __future__ import annotations

import json
import os
import sqlite3
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("EMBODIED_FIXTURES", HERE.parent / ".devtools" / "fixtures"))


def _reset(path: Path) -> Path:
    import shutil

    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def _frames(episode: int, length: int, *, bad: bool = False) -> list[dict]:
    rows = []
    for i in range(length):
        state = [0.1 * i + 0.01 * episode, -0.2 * i, 0.3, 0.0, 0.0, 0.0]
        action = [0.1 * (i + 1), -0.2 * (i + 1), 0.3, 0.0, 0.0, 0.0]
        if bad and i == length // 2:
            state[0] = float("nan")
        rows.append(
            {
                "frame_index": i,
                "timestamp": round(i / 30.0, 4),
                "observation.state": state,
                "action": action,
            }
        )
    return rows


def make_canonical(root: Path) -> Path:
    path = _reset(root / "canonical")
    dataset = {
        "format": "embodied-jsonl",
        "version": "1.0",
        "name": "so100-pick-place",
        "fps": 30,
        "robot_type": "so100",
        "features": {
            "observation.state": {"dtype": "float32", "shape": [6], "kind": "state"},
            "action": {"dtype": "float32", "shape": [6], "kind": "action"},
        },
    }
    (path / "dataset.json").write_text(json.dumps(dataset, indent=2) + "\n", encoding="utf-8")
    lines = []
    for episode, length in enumerate((60, 45)):
        lines.append(
            json.dumps(
                {
                    "episode_index": episode,
                    "task": "pick up the cube and place it in the box",
                    "frames": _frames(episode, length, bad=episode == 1),
                }
            )
        )
    (path / "episodes.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def make_lerobot(root: Path) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = _reset(root / "lerobot_v21")
    (path / "meta").mkdir(parents=True)
    (path / "data" / "chunk-000").mkdir(parents=True)
    (path / "videos" / "chunk-000" / "observation.images.top").mkdir(parents=True)
    features = {
        "observation.state": {"dtype": "float32", "shape": [6], "names": None},
        "action": {"dtype": "float32", "shape": [6], "names": None},
        "observation.images.top": {"dtype": "video", "shape": [480, 640, 3], "names": ["height", "width", "channels"]},
    }
    info = {
        "codebase_version": "v2.1",
        "robot_type": "so100",
        "total_episodes": 2,
        "total_frames": 105,
        "total_tasks": 1,
        "fps": 30,
        "features": features,
        "splits": {"train": "0:2"},
    }
    (path / "meta" / "info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    episodes = [
        {"episode_index": 0, "tasks": ["pick up the cube"], "length": 60},
        {"episode_index": 1, "tasks": ["pick up the cube"], "length": 45},
    ]
    (path / "meta" / "episodes.jsonl").write_text(
        "\n".join(json.dumps(row) for row in episodes) + "\n", encoding="utf-8"
    )
    (path / "meta" / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "pick up the cube"}) + "\n", encoding="utf-8"
    )
    offset = 0
    for episode, length in enumerate((60, 45)):
        states = [0.01 * (i + episode) for i in range(length * 6)]
        actions = [0.02 * (i + episode) for i in range(length * 6)]
        table = pa.table(
            {
                "observation.state": pa.array([states[i * 6 : (i + 1) * 6] for i in range(length)], type=pa.list_(pa.float32())),
                "action": pa.array([actions[i * 6 : (i + 1) * 6] for i in range(length)], type=pa.list_(pa.float32())),
                "timestamp": pa.array([i / 30.0 for i in range(length)], type=pa.float32()),
                "frame_index": pa.array(list(range(length)), type=pa.int64()),
                "episode_index": pa.array([episode] * length, type=pa.int64()),
                "index": pa.array(list(range(offset, offset + length)), type=pa.int64()),
                "task_index": pa.array([0] * length, type=pa.int64()),
            }
        )
        pq.write_table(table, path / "data" / "chunk-000" / f"episode_{episode:06d}.parquet")
        offset += length
    (path / "videos" / "chunk-000" / "observation.images.top" / "episode_000000.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42")
    (path / "videos" / "chunk-000" / "observation.images.top" / "episode_000001.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42")
    return path


def make_hdf5(root: Path) -> Path:
    import h5py
    import numpy as np

    path = _reset(root / "hdf5")
    target = path / "lift.hdf5"
    with h5py.File(target, "w") as handle:
        handle.attrs["env_args"] = json.dumps({"env_name": "Lift", "control_freq": 20, "type": "robomimic"})
        for demo in range(2):
            group = handle.create_group(f"data/demo_{demo}")
            length = 40 + demo * 10
            group.attrs["num_samples"] = length
            group.create_dataset("obs/object", data=np.random.rand(length, 7).astype("float32"))
            group.create_dataset("obs/robot0_eef_pos", data=np.random.rand(length, 3).astype("float32"))
            group.create_dataset("actions", data=np.random.rand(length, 7).astype("float32"))
            group.create_dataset("rewards", data=np.zeros(length, dtype="float32"))
    return path


def make_rosbag2(root: Path) -> Path:
    path = _reset(root / "rosbag2")
    target = path / "episode.db3"
    connection = sqlite3.connect(target)
    connection.executescript(
        """
        CREATE TABLE topics (id INTEGER PRIMARY KEY, name TEXT, type TEXT, serialization_format TEXT);
        CREATE TABLE messages (id INTEGER PRIMARY KEY, topic_id INTEGER, timestamp INTEGER, data BLOB);
        """
    )
    topics = [
        (1, "/joint_states", "sensor_msgs/msg/JointState", "cdr"),
        (2, "/camera/image_raw", "sensor_msgs/msg/Image", "cdr"),
    ]
    connection.executemany("INSERT INTO topics VALUES (?, ?, ?, ?)", topics)
    rows = []
    for index in range(30):
        rows.append((index * 2 + 1, 1, 1_700_000_000_000_000_000 + index * 33_000_000, b"\x00" * 64))
        if index % 3 == 0:
            rows.append((index * 2 + 2, 2, 1_700_000_000_000_000_000 + index * 33_000_000, b"\x00" * 32))
    connection.executemany("INSERT INTO messages VALUES (?, ?, ?, ?)", rows)
    connection.commit()
    connection.close()
    return path


def make_mcap(root: Path) -> Path:
    from mcap.writer import CompressionType, Writer

    path = _reset(root / "mcap")
    target = path / "episode.mcap"
    with target.open("wb") as handle:
        writer = Writer(handle, compression=CompressionType.NONE, use_chunking=False)
        writer.start(profile="ros2", library="embodied-data-studio-fixture")
        schema_id = writer.register_schema(name="geometry_msgs/msg/Twist", encoding="ros2msg", data=b"")
        channel_id = writer.register_channel(topic="/cmd_vel", message_encoding="cdr", schema_id=schema_id)
        for index in range(50):
            writer.add_message(
                channel_id=channel_id,
                log_time=1_700_000_000_000_000_000 + index * 50_000_000,
                publish_time=1_700_000_000_000_000_000 + index * 50_000_000,
                data=struct.pack("<6d", 0.1 * index, 0.0, 0.0, 0.0, 0.0, 0.1),
            )
        writer.finish()
    return path


def make_rlds(root: Path) -> Path:
    path = _reset(root / "rlds")
    info = {
        "name": "rlds_fixture",
        "fileFormat": "fileFormatVersion TFRecord",
        "splits": [{"name": "train", "shardLengths": ["3"], "numBytes": "0"}],
        "features": {
            "steps": {
                "sequence": {
                    "observation": {"features": {"state": {"shape": [4], "dtype": "tf.float32"}}},
                    "action": {"shape": [4], "dtype": "tf.float32"},
                    "reward": {"shape": [], "dtype": "tf.float32"},
                    "is_first": {"shape": [], "dtype": "tf.bool"},
                }
            }
        },
    }
    (path / "dataset_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    shard = path / "rlds_fixture-train.tfrecord-00000-of-00001"
    # three length-prefixed, CRC-terminated records (payload bytes are opaque here)
    with shard.open("wb") as handle:
        for index in range(3):
            payload = b"steps-" + bytes([index])
            handle.write(struct.pack("<Q", len(payload)) + struct.pack("<I", 0))
            handle.write(payload + struct.pack("<I", 0))
    return path


def make_broken(root: Path) -> Path:
    path = _reset(root / "broken")
    (path / "notes.txt").write_text("not a dataset\n", encoding="utf-8")
    return path


def main() -> int:
    created = {
        "canonical": make_canonical(ROOT),
        "lerobot": make_lerobot(ROOT),
        "hdf5": make_hdf5(ROOT),
        "rosbag2": make_rosbag2(ROOT),
        "rlds": make_rlds(ROOT),
        "broken": make_broken(ROOT),
    }
    try:
        created["mcap"] = make_mcap(ROOT)
    except Exception as exc:  # pragma: no cover - fixture only
        print(f"skipped mcap fixture: {exc}", file=sys.stderr)
    for name, path in created.items():
        print(f"{name:10s} {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
