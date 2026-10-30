# Embodied Data Studio

给具身智能数据平台开发者的一套工具：**格式识别 / 校验 / 质检 / 标注任务生成 / 格式转换 / 数据集编目 / 项目脚手架**，同时以四种形态提供——

| 形态 | 入口 | 给谁用 |
| --- | --- | --- |
| Python 库 | `import embodied_core` | 平台后端、离线脚本、CI |
| 命令行 | `embodied <command>` | 终端里的快速巡检、流水线步骤 |
| MCP 服务 | `embodied-mcp`（stdio） | Codex、Claude Code、DeepSeek、Cursor 等任意支持 MCP 的模型/客户端 |
| DSH 插件 | `dsh-embodied-data` | DeepSeek Harness，把上面 23 个 MCP 工具注册成原生 agent 工具 |

核心逻辑只有一份 Python 实现，模型侧无论走 MCP 还是 DSH 插件，用的都是同一套行为，不会出现"Codex 里能跑、DeepSeek 里报错"的分裂。

```
┌──────────────┐   ┌──────────────┐   ┌───────────────┐
│  Codex CLI   │   │ Claude Code  │   │ DeepSeek/DSH  │
└──────┬───────┘   └──────┬───────┘   └───────┬───────┘
       │ MCP (stdio)      │ MCP (stdio)       │ MCP  │ 插件工具
       └──────────┬───────┴───────────────────┘      │
                  ▼                                  ▼
        embodied_core.mcp_server  ◄──── dsh-embodied-data (JS 桥接)
                  │
                  ▼
        embodied_core (Python：formats / validate / quality / convert / catalog / scaffold)
```

## 30 秒上手

```bash
# 1) 装库（仓库内开发用 PYTHONPATH，正式用 pip install -e .）
pip install -e ".[all]"

# 2) 看一眼是什么格式
embodied detect ./datasets/aloha_pick_cube

# 3) 体检 + 打分
embodied validate ./datasets/aloha_pick_cube
embodied quality  ./datasets/aloha_pick_cube --json

# 4) 转成 LeRobot v2.1，给训练用
embodied convert ./datasets/aloha_pick_cube ./out/lerobot --target lerobot

# 5) 把整个数据盘编目，之后按条件搜
embodied catalog index /data/embodied --db catalog.db
embodied catalog search --db catalog.db --robot so100 --min-episodes 50
```

Python 里同样直接：

```python
from embodied_core import open_dataset, validate

dataset = open_dataset("./datasets/aloha_pick_cube")   # 自动识别 7 种格式
ids, columns = dataset.frames()                        # 数值列，numpy 二维
for chunk in dataset.iter_frames(chunk_frames=50_000):  # 大数据集按块读取
    print(len(chunk.ids), len(chunk.columns))
report = validate("./datasets/aloha_pick_cube")
print(report["score"], [i["code"] for i in report["issues"]])
```

## 能力一览

**格式识别**（`detect_dataset`）：LeRobot v2.1/v3.0、HDF5（robomimic / 通用组）、rosbag2、rosbag1、MCAP、RLDS/TFRecord、canonical JSONL。识别基于目录结构与文件魔数，不看扩展名。

**校验**（`validate_dataset`）：8 组检查——结构、元数据、特征词表、episode 一致性、fps 一致性、LeRobot 规范（含 v3 chunk/file 编号连续性）、parquet 可读性、HDF5 空 demo。每条问题带 `code / severity / scope / hint`，并按 error×30、warn×8、info×1 扣分，`score >= 80` 才建议入库。

**质检**（`quality_report`）：逐特征统计（nan/inf 比例、分位数、范数、近零比）、鲁棒异常检测（median + 1.4826·MAD，跳变/停滞/非有限值窗口）、按运动量切 moving/idle 段。记账列（`episode_index`、`frame_index`…）不参与异常检测，避免把编码器噪声当物理问题。报告按「数据集全量文件指纹 + 参数」增量缓存，数据集没变时第二次调用直接命中；帧数据按 `chunk_frames` 分块读取，并受 `max_frames` / `EMBODIED_MAX_FRAMES` 上限保护。

**异步作业与进度**（`submit_job` / `job_status` / `cancel_job` / `list_jobs`）：大质检或逐帧校验可后台运行，MCP 通过 `notifications/progress` 推送阶段进度，支持协作式取消。异步只开放只读分析，写入仍走同步审批路径。

**标注辅助**（`labeling_tasks`）：把异常窗口和运动段变成可交付的标注任务清单，带固定回答格式，方便回灌。

**转换**（`convert_dataset`）：任意可读格式 ⟷ canonical JSONL / LeRobot v2.1，转换时记录被丢弃的非数值特征。

**编目 v2**（`catalog_*`）：数据集、不可变版本、运行记录和版本谱系分表存储；路径作为逻辑身份，文件指纹变化时生成新版本并写 `supersedes` 谱系边。支持文本/格式/机器人型号/episode 数/时长/分数过滤，以及 `catalog versions|lineage|runs|history` 审计。

**质量门**（`quality_gate` / `embodied gate`）：规则可用 JSON/TOML 配置，覆盖期望 fps 与容差、必需特征 key/kind、最低分/episode/frame 数、最大 NaN 比例及异常窗口阈值。输出结构化 `gate.failures`，可直接作为入库或发布 CI 的退出码。

**流式协议解码**（`decode_samples`）：rosbag2 内置 CDR 读取器，覆盖 JointState、IMU、Odometry、Image/深度图、PointCloud2、LaserScan、Float/Int 数组和 String/语言指令；MCAP 安装 `mcap` 后复用同一解码器。RLDS 的 Example/SequenceExample 由内置 protobuf/TFRecord 轻量解码器读取，普通场景不需要 TensorFlow。

**模态与媒体**（`inspect_media`）：识别 image/video/depth/point_cloud/language，汇总任务指令；视频仅在 ffprobe 可用时探测宽高、时长、codec、帧率和码率，缺失时返回结构化安装提示。

**标注闭环**（`labeling_tasks` → `apply_annotations`）：任务带稳定 `task_id`；回写支持 valid/invalid/needs_fix、重复标签 last-wins、冲突日志、覆盖率与异常 episode 汇总，输出文件原子写入。

**LeRobot v3 导出**：`target=lerobot-v3` 生成 v3.0 chunk/file 布局、`meta/stats.json` 归一化统计、episode 元数据和 train/validation/test splits，并在导出后自动跑格式校验。

**平台接口**：只读 HTTP Catalog API 使用标准库，支持 Bearer/X-API-Key、CORS、分页和强 ETag；`storage_pull` 支持 file/http(s)/s3，下载带重试、SHA-256、路径穿越防护和原子发布。

**v0.5 稳定性**：CDR/TFRecord/点云均有消息、记录和点数上限；导出和标注写入原子化；解码器未知类型显式失败；`embodied doctor` 汇总依赖、解码器、资源上限和平台能力。

**脚手架**（`scaffold_project`）：`platform`（平台骨架 + 数据契约 + 质量门）/ `adapter`（新格式适配器 + 测试）/ `pipeline`（摄取流水线 + 测试），生成的文件都带类型标注，可直接跑。

## 文档

| 文档 | 内容 |
| --- | --- |
| [docs/getting-started.md](docs/getting-started.md) | 安装、第一个数据集、常见工作流 |
| [docs/formats.md](docs/formats.md) | 支持矩阵、每种格式的探测依据、如何新增适配器 |
| [docs/mcp.md](docs/mcp.md) | 23 个 MCP 工具的参数与返回结构 |
| [docs/cli.md](docs/cli.md) | 命令、退出码、JSON 输出形态 |
| [docs/models.md](docs/models.md) | Codex / Claude Code / DeepSeek / Cursor / DSH 的接入配置 |
| [docs/plugin.md](docs/plugin.md) | DSH 插件安装、配置项、自检 |
| [docs/api.md](docs/api.md) | HTTP Catalog API、对象存储与平台诊断 |
| [docs/roadmap.md](docs/roadmap.md) | v0.4 平台化与 v0.5 稳定性范围 |
| [packages/dsh-plugin/README.md](packages/dsh-plugin/README.md) | 插件包说明 |

## 目录结构

```
D:\具身平台插件开发\
├── src/embodied_core/          # Python 核心（唯一的事实来源）
│   ├── formats/                # 7 个格式适配器 + registry
│   ├── validate.py             # 8 组检查、评分
│   ├── quality.py              # 统计、异常检测、切段、标注任务
│   ├── reports.py              # 质检报告 + 增量缓存编排
│   ├── gate.py                 # JSON/TOML 质量门规则与判定
│   ├── cache.py                # 数据集指纹与 SQLite 报告缓存
│   ├── jobs.py                 # 后台作业、进度与取消
│   ├── ros2_codec.py           # 无依赖 CDR + 常见 ROS 2 消息解码
│   ├── tfrecord.py             # TFRecord / RLDS Example 轻量解码
│   ├── media.py                # ffprobe 可选探测、深度/点云/语言模态
│   ├── annotations.py          # 标注任务稳定 ID、回写与冲突处理
│   ├── storage.py              # file/http(s)/s3 原子拉取与校验
│   ├── api.py                  # 只读 HTTP Catalog API
│   ├── platform.py             # 依赖/能力/资源上限诊断
│   ├── convert.py              # JSONL / LeRobot 目标格式
│   ├── catalog.py              # schema v2：数据集/版本/运行/谱系
│   ├── scaffold.py             # 项目模板
│   ├── cli.py                  # embodied 命令
│   └── mcp_server.py           # 零依赖 MCP stdio 服务
├── packages/dsh-plugin/        # 零依赖 ESM 插件（JS 只做 MCP 桥接）
├── tests/                      # unittest（pytest 兼容），169 项
├── tools/                      # 夹具生成 + 端到端脚本
├── examples/                   # 可直接运行的示例
└── docs/
```

## 开发

```bash
python -m unittest discover -s tests -t .     # 169 项单元测试，或用 pytest
python tools/e2e.py                           # 21 项端到端（需先 tools/make_fixtures.py 生成夹具）
node packages/dsh-plugin/scripts/probe.mjs    # 插件自检（mock cordis ctx）
node --test packages/dsh-plugin/test/*.test.mjs  # 16 项插件单测（路径/并发/审批）
node tools/check_clean_install.mjs --python /abs/path/to/python  # 打包后临时安装并探测
```

夹具、离线 wheel 与本地依赖树放在 `.devtools/`（已在 `.gitignore` 中）：`tools/make_fixtures.py` 生成 `.devtools/fixtures/`，`pip install` 不可用时把 wheel 解压到 `.devtools/pydeps/` 并加进 `PYTHONPATH`。

Python 依赖全部可选：核心只需要 `numpy`；parquet 要 `pyarrow`，HDF5 要 `h5py`，MCAP 要 `mcap`。缺依赖时对应适配器会给出明确的 `FramesUnavailable(reason, install)`，而不是抛一坨 ImportError。

## 状态

- 核心库、CLI、MCP、DSH 插件均已跑通（169 项 Python 单测 + 16 项插件单测 + 21 项端到端 + 插件 probe + 干净安装）。
- 插件已装进本机 profile（`~/.dsh/profiles/desktop`）并在 Harness 会话里实测：24 个 `embodied_*` 工具可用，live 调用 detect / summarize / quality 返回真实数据。
- `quality_report` 默认写用户缓存目录；可用 `embodied cache stats|clear` 查看或清理，不占用数据集目录。
- `iter_frames()` 在 JSONL、LeRobot parquet 和 HDF5 上按块产出 `FrameChunk`；`frames()` 仍保留，但受 `max_frames` 保护，避免误把海量数据一次性读入内存。
- 编目库会自动把旧 v1 数据库迁移到 schema v2；重复索引未变化的文件只记一次 run，不产生新版本。
- 质量门规则既能内联传给 MCP，也能通过 `--rules rules.json|rules.toml` 在 CI 中复用。
- npm 发布包会在 `prepack` 时内置 `python/embodied_core`，安装后自动解析，不再依赖仓库源码路径。
- 写工具默认串行；工作区外目标需要一次性审批，避免转换/脚手架并发覆盖或越界写入。
- rosbag2 内置 CDR 解码（JointState/IMU/Odometry/Image/PointCloud2/LaserScan/标量与语言 topic）；MCAP 在安装 `mcap` 后可走同一解码器；RLDS 常见 Example/SequenceExample 无需 TensorFlow。
- 视频流（LeRobot `observation.images.*`）只做元数据描述，不解码像素。
