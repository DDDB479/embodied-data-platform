# Embodied Data Studio

给具身智能数据平台开发者的一套工具：**格式识别 / 校验 / 质检 / 标注任务生成 / 格式转换 / 数据集编目 / 项目脚手架**，同时以三种形态提供——

| 形态 | 入口 | 给谁用 |
| --- | --- | --- |
| Python 库 | `import embodied_core` | 平台后端、离线脚本、CI |
| 命令行 | `embodied <command>` | 终端里的快速巡检、流水线步骤 |
| MCP 服务 | `embodied-mcp`（stdio） | Codex、Claude Code、DeepSeek、Cursor 等任意支持 MCP 的模型/客户端 |
| DSH 插件 | `dsh-embodied-data` | DeepSeek Harness，把上面 13 个工具直接变成 agent 工具 |

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
report = validate("./datasets/aloha_pick_cube")
print(report["score"], [i["code"] for i in report["issues"]])
```

## 能力一览

**格式识别**（`detect_dataset`）：LeRobot v2.1/v3.0、HDF5（robomimic / 通用组）、rosbag2、rosbag1、MCAP、RLDS/TFRecord、canonical JSONL。识别基于目录结构与文件魔数，不看扩展名。

**校验**（`validate_dataset`）：8 组检查——结构、元数据、特征词表、episode 一致性、fps 一致性、LeRobot 规范（含 v3 chunk/file 编号连续性）、parquet 可读性、HDF5 空 demo。每条问题带 `code / severity / scope / hint`，并按 error×30、warn×8、info×1 扣分，`score >= 80` 才建议入库。

**质检**（`quality_report`）：逐特征统计（nan/inf 比例、分位数、范数、近零比）、鲁棒异常检测（median + 1.4826·MAD，跳变/停滞/非有限值窗口）、按运动量切 moving/idle 段。记账列（`episode_index`、`frame_index`…）不参与异常检测，避免把编码器噪声当物理问题。

**标注辅助**（`labeling_tasks`）：把异常窗口和运动段变成可交付的标注任务清单，带固定回答格式，方便回灌。

**转换**（`convert_dataset`）：任意可读格式 ⟷ canonical JSONL / LeRobot v2.1，转换时记录被丢弃的非数值特征。

**编目**（`catalog_*`）：SQLite 一张表，按路径增量索引（mtime+size 标记），支持文本/格式/机器人型号/episode 数/时长/分数过滤。

**脚手架**（`scaffold_project`）：`platform`（平台骨架 + 数据契约 + 质量门）/ `adapter`（新格式适配器 + 测试）/ `pipeline`（摄取流水线 + 测试），生成的文件都带类型标注，可直接跑。

## 文档

| 文档 | 内容 |
| --- | --- |
| [docs/getting-started.md](docs/getting-started.md) | 安装、第一个数据集、常见工作流 |
| [docs/formats.md](docs/formats.md) | 支持矩阵、每种格式的探测依据、如何新增适配器 |
| [docs/mcp.md](docs/mcp.md) | 13 个 MCP 工具的参数与返回结构 |
| [docs/cli.md](docs/cli.md) | 命令、退出码、JSON 输出形态 |
| [docs/models.md](docs/models.md) | Codex / Claude Code / DeepSeek / Cursor / DSH 的接入配置 |
| [docs/plugin.md](docs/plugin.md) | DSH 插件安装、配置项、自检 |
| [packages/dsh-plugin/README.md](packages/dsh-plugin/README.md) | 插件包说明 |

## 目录结构

```
D:\具身平台插件开发\
├── src/embodied_core/          # Python 核心（唯一的事实来源）
│   ├── formats/                # 7 个格式适配器 + registry
│   ├── validate.py             # 8 组检查、评分
│   ├── quality.py              # 统计、异常检测、切段、标注任务
│   ├── convert.py              # JSONL / LeRobot 目标格式
│   ├── catalog.py              # SQLite 编目与检索
│   ├── scaffold.py             # 项目模板
│   ├── cli.py                  # embodied 命令
│   └── mcp_server.py           # 零依赖 MCP stdio 服务
├── packages/dsh-plugin/        # 零依赖 ESM 插件（JS 只做 MCP 桥接）
├── tests/                      # unittest（pytest 兼容），91 项
├── tools/                      # 夹具生成 + 端到端脚本
├── examples/                   # 可直接运行的示例
└── docs/
```

## 开发

```bash
python -m unittest discover -s tests -t .     # 91 项单元测试，或用 pytest
python tools/e2e.py                           # 16 项端到端（需先 tools/make_fixtures.py 生成夹具）
node packages/dsh-plugin/scripts/probe.mjs    # 插件自检（mock cordis ctx）
```

夹具、离线 wheel 与本地依赖树放在 `.devtools/`（已在 `.gitignore` 中）：`tools/make_fixtures.py` 生成 `.devtools/fixtures/`，`pip install` 不可用时把 wheel 解压到 `.devtools/pydeps/` 并加进 `PYTHONPATH`。

Python 依赖全部可选：核心只需要 `numpy`；parquet 要 `pyarrow`，HDF5 要 `h5py`，MCAP 要 `mcap`。缺依赖时对应适配器会给出明确的 `FramesUnavailable(reason, install)`，而不是抛一坨 ImportError。

## 状态

- 核心库、CLI、MCP、DSH 插件均已跑通（91 项单元测试 + 16 项端到端 + 插件 probe）。
- 插件已装进本机 profile（`~/.dsh/profiles/desktop`）并在 Harness 会话里实测：14 个 `embodied_*` 工具可用，live 调用 detect / summarize / quality 返回真实数据。
- RLDS/MCAP 目前是 metadata 级支持：能识别、统计、校验，但读不到逐帧张量（需要 TensorFlow / CDR schema 解码），`frames()` 会显式抛 `FramesUnavailable`。
- 视频流（LeRobot `observation.images.*`）只做元数据描述，不解码像素。
