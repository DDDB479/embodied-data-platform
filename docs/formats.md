# 格式支持

## 支持矩阵

| 格式 | 探测依据 | 可选依赖 | 能读帧 | 典型来源 |
| --- | --- | --- | --- | --- |
| `lerobot` v2.1 / v3.0 | `meta/info.json` + `meta/episodes*` 或 `data/**/*.parquet` | `pyarrow` | ✅ | LeRobot 训练流水线 |
| `hdf5` | HDF5 魔数 `\x89HDF\r\n\x1a\n` | `h5py` | ✅ | robomimic、ALOHA、自建遥操作 |
| `mcap` | MCAP 魔数 `\x89MCAP0\r\n` | `mcap`（内置 stdlib 回退） | ❌ | Foxglove、自动驾驶/机器人总线录制 |
| `rosbag2` | `.db3` 且有 `topics` + `messages` 表 | 无（sqlite3） | ❌ | ROS 2 |
| `rosbag1` | `#ROSBAG V2.0\n` | 无 | ❌ | 老 ROS 1 |
| `rlds` | `dataset_info.json` / `features.json` + shards | `tensorflow`（解码 example proto） | ❌ | TFDS 生态、Open X-Embodiment |
| `embodied-jsonl` | `dataset.json` 的 format 标记，或 JSONL episode 行 | 无 | ✅ | 本工具链的 canonical 交换格式 |

"能读帧"= `dataset.frames()` 返回 `(ids, dict[str, 2D ndarray])`，可以把数值列交给质量分析与转换。大文件优先用 `dataset.iter_frames(chunk_frames=...)`，它产出带 `offsets` 的 `FrameChunk`，并受 `max_frames` / `EMBODIED_MAX_FRAMES` 上限保护。"不能"= 抛 `FramesUnavailable(reason, install)`，但**元数据级能力照常**：识别、统计 episode/消息数、校验、编目都能用。

为什么 MCAP / rosbag / RLDS 读不了帧：这些格式里的"帧"要先按 schema 解码（ROS 的 `sensor_msgs/Image`、CDR 编码、TFRecord 的 Example proto）。硬解码会把工具链绑死在某个 SDK 的版本上，所以这里显式划界——需要逐帧时请先转成 LeRobot/HDF5，或者写一个项目内的解码器。

## 探测是怎么做的

统一走 `registry.detect_dataset(path)`：**所有适配器都跑一遍**，按 `confidence` 降序取第一，单个适配器抛异常不影响其它。所以扩展名撒谎也没关系——判据是目录结构与文件魔数。

```python
from embodied_core.registry import detect_dataset

detection = detect_dataset("./maybe_a_dataset")
print(detection.format, detection.version, detection.confidence, detection.evidence)
# lerobot v2.1 0.98 ['meta/info.json', 'meta/episodes.jsonl', 'data/**/*.parquet']
```

`evidence` 会出现在 `detect_dataset` 工具的输出里，模型（和人）可以据此复核判断。

## LeRobot 两种布局

**v2.1（每 episode 一个文件）**

```
meta/info.json            # codebase_version, fps, robot_type, features, total_episodes/frames
meta/episodes.jsonl
meta/tasks.jsonl
data/chunk-000/episode_000000.parquet
videos/chunk-000/<camera_key>/episode_000000.mp4
```

**v3.0（文件分块）**

```
meta/info.json
meta/stats.json
meta/tasks.parquet
meta/episodes/chunk-000/file-000.parquet
data/chunk-000/file-000.parquet
videos/<camera_key>/chunk-000/file-000.mp4
```

校验会检查 v3 的 `chunk-XXX` / `file-XXX` 编号是否从 000 连续，缺失的 metadata 会降级为 note 而不是崩溃（`lerobot.meta_missing`）。

## canonical JSONL

最小、无依赖、可 diff 的交换格式：

```
dataset.json     # {"format": "embodied-jsonl", "fps": 30, "robot_type": "...", "features": {...}}
episodes.jsonl   # 每行一个 episode
```

每行两种写法：

```json
{"episode_index": 0, "task": "pick up the cube",
 "frames": [{"observation.state": [0.1,0.2], "action": [0.0,0.1], "timestamp": 0.0}, ...]}
```

```json
{"episode_index": 0, "task": "pick up the cube",
 "columns": {"observation.state": [[0.1,0.2], ...], "action": [[0.0,0.1], ...]}}
```

`frames` 适合人写和小数据，`columns` 适合程序吐数据（列式，不用重复键名）。

## 新增一个格式适配器

两步：写模块，注册。

**1. 实现四个函数**（参照 `src/embodied_core/formats/jsonl.py`）

```python
# src/embodied_core/formats/myformat.py
from __future__ import annotations
from pathlib import Path
from .base import DatasetObject, Detection, FramesUnavailable, guess_kind
from ..model import DatasetSummary, EpisodeRecord, FeatureSpec

FORMAT = "myformat"


def detect(path: Path) -> Detection | None:
    """返回 None 表示"不是我"。confidence 0..1，evidence 给判定依据。"""
    if not (path / "manifest.yaml").is_file():
        return None
    return Detection(format=FORMAT, version="1.0", confidence=0.9, evidence=["manifest.yaml"])


def load(path: Path) -> DatasetObject:
    summary = DatasetSummary(
        path=str(path), format=FORMAT, version="1.0", fps=30.0, robot_type=None,
        features=[FeatureSpec(key="observation.state", dtype="float32", shape=[6], kind="state")],
        episodes=[EpisodeRecord(index=0, length=120)],
        tasks=[], split=None, notes=[], meta={}, sources=[],
    )
    return DatasetObject(root=path, summary=summary, capabilities={"episodes", "features"})


def frames(dataset: DatasetObject):
    """没有实现就显式声明，别返回空数组。"""
    raise FramesUnavailable("myformat frames need my-sdk", install="pip install my-sdk")
```

**2. 注册**，在 `src/embodied_core/formats/registry.py` 的 `_ADAPTERS` 列表里加 `myformat`（顺序即优先级，越靠前越先尝试），并在 `FORMAT_HELP` 里补一条说明。若名字有多种叫法，加进 `_ALIASES`。

**3. 加测试**：`embodied scaffold ./adapters/myformat --kind adapter` 会生成骨架和测试；也可以照抄 `tests/test_formats.py` 里任一用例。

约定：

- `detect` 必须**便宜**（只 stat 几个文件），因为每次识别都会调用所有适配器。
- 拿不准就返回低 confidence，不要返回 1.0。
- 缺依赖时 `load` 仍应成功（元数据级可用），只在 `frames` 里抛 `FramesUnavailable`。
- issue code 用 `命名空间.小写下划线`，例如 `myformat.bad_manifest`，并在 `docs` 或 hint 里说明怎么修。

## 特征命名约定

转换、质检、标注都依赖这套词表（`validate` 的 `features` 检查会对照）：

| kind | 约定键名 | 说明 |
| --- | --- | --- |
| `state` | `observation.state` | 本体感知（关节角、夹爪开度…） |
| `action` | `action` | 动作标签，训练必需 |
| `image` | `observation.images.<camera>` | 只做元数据描述，不解码 |
| 记账 | `timestamp` / `frame_index` / `episode_index` / `index` / `task_index` | 质检时跳过，转换时保留 |

没有 `observation.state` 或 `action` 会得到 `feature.no_state` / `feature.no_action` 警告——这是刻意的：具身数据集缺了这两样，训练侧基本用不了。
