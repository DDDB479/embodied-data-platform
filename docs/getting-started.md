# 快速上手

## 安装

三种方式，按场景选：

```bash
# 方式 1：从源码装（推荐开发时用）
git clone <this-repo> && cd 具身平台插件开发
pip install -e ".[all]"        # 核心 + parquet + hdf5 + mcap

# 方式 2：只要核心，最小依赖
pip install -e .               # 仅 numpy

# 方式 3：不安装，用完即走
PYTHONPATH=src python -m embodied_core.cli detect ./dataset
```

`[all]` 会装 `pyarrow`（parquet/LeRobot）、`h5py`（HDF5）、`mcap`（MCAP）。这些依赖是**惰性**的：没装时格式识别仍然工作（靠目录结构和魔数），只是读不到帧数据，并且会明确告诉你缺什么：

```python
from embodied_core import open_dataset
from embodied_core.formats.base import FramesUnavailable

try:
    dataset.frames()
except FramesUnavailable as exc:
    print(exc.reason, exc.install)   # parquet data needs pyarrow, pip install "embodied-core[parquet]"
```

## 第一个数据集

```bash
$ embodied detect ./datasets/aloha_pick_cube
format     lerobot
version    v2.1
confidence 0.98
evidence   meta/info.json, meta/episodes.jsonl, data/**/*.parquet

$ embodied info ./datasets/aloha_pick_cube
path             ./datasets/aloha_pick_cube
format           lerobot v2.1
episodes         24
total frames     18_432
duration         614.4 s
fps              30.0
robot type       aloha
tasks            pick up the cube, place it in the bin
features         observation.state float32[14] (state)
                 action            float32[14] (action)
                 observation.images.top video[480,640,3] (image)
```

`info` 不带 `--json` 时给人看，带 `--json` 时给程序（或模型）看。

## 一次完整的入库流程

真实平台的流程通常是"发现 → 体检 → 修 → 转 → 编目"。命令行可以一条条跑：

```bash
DATA=./datasets/aloha_pick_cube

# 1) 结构 + 元数据 + 特征词表 + 规范检查
embodied validate "$DATA"

# 2) 逐帧统计与异常窗口（需要能读帧，即装了 pyarrow）
embodied quality "$DATA" --json > quality.json
embodied segment "$DATA"                    # 按运动量切 moving / idle
embodied label "$DATA" --out tasks.json     # 生成标注任务清单

# 3) 转换：只有通过质量门的样本才进训练集
embodied convert "$DATA" ./out/lerobot --target lerobot

# 4) 编目，供后续检索
embodied catalog index ./datasets --db catalog.db
embodied catalog search --db catalog.db --min-score 80 --min-episodes 20
```

同一个流程在 Python 里：

```python
from embodied_core import open_dataset, validate, convert
from embodied_core.quality import detect_anomalies, segment_by_motion, build_labeling_tasks

path = "./datasets/aloha_pick_cube"
report = validate(path)
if report["score"] < 80 or report["errors"]:
    raise SystemExit(f"rejected: {report['score']} / {report['errors']} errors")

dataset = open_dataset(path)
ids, columns = dataset.frames()
windows = detect_anomalies(ids, columns, fps=dataset.summary.fps)
segments = segment_by_motion(ids, columns, fps=dataset.summary.fps)
manifest = build_labeling_tasks(dataset.summary, windows, include_segments=True, segments=segments)

convert(path, "./out/lerobot", target="lerobot", overwrite=True)
print(f"{len(windows)} anomaly windows, {manifest['task_count']} labeling tasks")
```

## 让模型来开车

工具本身不依赖任何特定模型。装好 MCP 之后，Codex / Claude Code / DeepSeek / Cursor 都能直接调用：

```bash
# 先确认服务能起来
embodied-mcp --list-tools

# Codex：见 docs/models.md
codex mcp add embodied -- embodied-mcp
```

接好之后可以直接对模型说：

> 帮我看下 ./datasets/new_run 是什么格式，跑一遍质检，把 score 低于 80 的 episode 列出来，然后转成 LeRobot 放到 /data/lerobot 下。

模型会依次调用 `detect_dataset` → `validate_dataset` → `quality_report` → `convert_dataset`。完整工具清单见 [mcp.md](mcp.md)。

## 新建一个平台项目

不必从空目录开始：

```bash
embodied scaffold ./my-platform --kind platform --name acme-embodied
cd my-platform && python -m unittest        # 生成的测试可以直接跑
```

生成内容包括：`README.md`、`DATA_CONTRACT.md`（帧契约、单位、质量门）、`src/<pkg>/api.py`（数据集服务接口）、`src/<pkg>/quality.py`（质量门实现）、`src/<pkg>/adapters.py`（新格式适配位）、`tests/`。

`--kind adapter` 只生成一个格式适配器骨架和它的测试；`--kind pipeline` 生成摄取流水线（扫描 → 校验 → 转换 → 编目）和测试。

## 常见问题

**为什么 `validate` 对空目录返回 1 而不是崩？** 因为对 agent 来说，"这不是一个数据集"是一种结论，不是异常。报告里 `ok=false`、`format=null`，issues 含 `format.unsupported`。只有 `detect` 会抛 `UnsupportedFormat`（它必须给一个答案）。

**nan 比例为什么是 4 位小数？** 报告要稳定可 diff，统一 round 到 4 位。

**为什么 `quality` 不报 `episode_index` 的异常？** 这些列是编码器写出来的记账列，天然单调或分段恒定，扫描它们只会制造假阳性。`quality.py` 里的 `METADATA_FEATURES` 就是干这个的。

**能读视频吗？** 不能。LeRobot 的 `observation.images.*` 只做元数据描述（分辨率、通道），不解码像素——那属于训练侧或专门的抽帧工具。
