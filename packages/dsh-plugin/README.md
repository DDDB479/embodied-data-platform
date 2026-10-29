# dsh-embodied-data

DeepSeek Harness 插件：把 [embodied-data-studio](../../README.md) 的 13 个数据平台工具注册成原生 agent 工具，并注入使用它们的正确顺序。

零依赖纯 ESM——不需要 `tsc`，不在 `node_modules` 里拉任何东西。JS 只做一件事：管理 `python -m embodied_core.mcp_server` 子进程并做 JSON-RPC 桥接。

发布 tarball 会在 `prepack` 阶段自带 `python/embodied_core`，因此 `pnpm add dsh-embodied-data` 后不需要再找仓库源码；开发态 `link:` 安装仍直接使用仓库里的 `src/embodied_core`。

```bash
node scripts/probe.mjs --python /path/to/python   # 自检，不需要启动 Harness
node --test test/*.test.mjs                       # 路径、并发与审批单测
```

安装、配置项、排查见 [docs/plugin.md](../../docs/plugin.md)。

## 暴露的工具

| Harness 工具名 | 作用 |
| --- | --- |
| `embodied_self_check` | 解释器、import 路径、MCP 进程状态（不依赖 Python 也能答） |
| `embodied_detect_dataset` | 识别格式（LeRobot / HDF5 / rosbag / MCAP / RLDS / JSONL） |
| `embodied_summarize_dataset` | 特征表、episode 数、fps、机器人型号 |
| `embodied_validate_dataset` | 8 组结构检查 + 0–100 分 |
| `embodied_quality_report` | 逐特征统计 + 异常窗口 |
| `embodied_segment_episodes` | moving / idle 切段 |
| `embodied_labeling_tasks` | 生成标注任务清单 |
| `embodied_convert_dataset` | 转 canonical JSONL / LeRobot v2.1 |
| `embodied_scaffold_project` | platform / adapter / pipeline 骨架 |
| `embodied_catalog_index` | 扫描目录、增量编目 |
| `embodied_catalog_search` | 按文本/格式/型号/episode 数/分数检索 |
| `embodied_catalog_stats` | 编目汇总 |
| `embodied_list_formats` | 支持的格式与检查项 |
| `embodied_plan_windows` | 训练窗口规划 |

## 输出约定

每次调用返回 `{ tool, ok, data }`；`data` 是 MCP 工具的原始 JSON（超过 `maxOutputChars` 截断）。工具失败时 `ok=false`，`data` 里是 `{"error": {"code": "...", "message": "..."}}`。

## 安全约定

- 只有 `annotations.readOnlyHint: true` 的 MCP 工具声明为 `isConcurrencySafe`；写工具保持 DSH 默认串行。
- `convert_dataset` / `scaffold_project` / `catalog_index` / 带 `out` 的 `labeling_tasks` 在工作区外写入前调用 `approval.request`，只接受 `allowed-once`。
- 没有审批服务、审批被拒绝或 DSH 为只读模式时，写入不会发生。

## 文件

```
index.js            apply(ctx, config)：注册工具 + 提示词 section
cordis.patch.yml    bundle 声明
src/config.js       解释器/PYTHONPATH 解析
src/mcp.js          MCP stdio 客户端（spawn、分帧、超时、重启）
src/tools.js        MCP 工具 → Harness 工具
src/prompt.js       工作流与约定提示词
scripts/probe.mjs   mock ctx 的端到端自检
```
