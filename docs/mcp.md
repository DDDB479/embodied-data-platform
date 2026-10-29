# MCP 工具参考

服务名 `embodied-data-studio`，协议版本 `2024-11-05`，stdio（每行一个 JSON-RPC 消息，支持 batch）。

```bash
embodied-mcp                 # 启动服务（客户端一般自己拉起）
embodied-mcp --list-tools    # 本地打印工具清单，排查配置问题用
```

每个工具返回**一个 JSON 文本块**。成功时是业务数据本身；失败时是 `{"error": {"code": "...", "message": "..."}}`，不会抛协议级异常——模型可以读到错误并自行纠正。

`tools/list` 会为每个工具附带 MCP annotations：读工具 `readOnlyHint: true`，写工具 `readOnlyHint: false`。写工具包括 `labeling_tasks`（仅在传 `out` 时落盘）、`convert_dataset`、`scaffold_project` 和 `catalog_index`。支持审批的宿主会据此提示；DSH 插件还会额外把写工具设为串行，并对工作区外目标申请一次性审批。

---

## 1. `detect_dataset`

识别磁盘上的数据集格式。

| 参数 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `path` | string | ✓ | 数据集目录或单体文件（`.hdf5` / `.mcap` / `.db3` / `.bag` / `.jsonl`） |

返回 `{format, version, confidence, evidence[], detail}`。`evidence` 是判定依据（如 `["meta/info.json", "data/**/*.parquet"]`），方便向人解释"为什么认为是 LeRobot"。

---

## 2. `summarize_dataset`

拿到 schema 再动手。返回 `path, format, version, episodes, total_frames, total_duration_s, fps, robot_type, split, tasks[], features[], notes[], sources[], meta, capabilities[]`。

| 参数 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `path` | string | ✓ | 数据集路径 |
| `include_episodes` | boolean | | 附带每个 episode 的 `{index, length, task, duration_s}` 列表 |

`features[]` 里每项是 `{key, dtype, shape, kind, extra}`，`kind ∈ {state, action, image, frame, other}`——模型据此判断"这个数据集有没有动作标签"。

---

## 3. `validate_dataset`

结构与规范校验，返回带分数的报告。

| 参数 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `path` | string | ✓ | 数据集路径 |
| `checks` | string[] | | 只跑指定检查，见下表；默认全部（不含 `parquet`） |
| `include_frames` | boolean | | 额外跑逐帧检查（需要能读帧，慢） |

检查项：

| key | 检查内容 | 典型 issue code |
| --- | --- | --- |
| `structure` | 路径存在、能识别格式 | `dataset.empty` / `format.unsupported` |
| `metadata` | fps、机器人型号、特征表 | `meta.missing_fps` / `meta.no_features` |
| `features` | 特征词表 | `feature.no_state` / `feature.no_action` |
| `episodes` | 索引唯一、长度非零、长度离群 | `episode.duplicate_index` / `episode.length_outlier` |
| `fps` | 各 episode fps 一致性（±5%） | `episode.fps_mismatch` |
| `lerobot.spec` | meta 必需文件、版本、chunk/file 编号连续 | `lerobot.meta_missing` / `lerobot.chunk_gap` |
| `parquet` | parquet 可读、非空 | `parquet.empty` |
| `hdf5.demos` | `data/` 下是否有空 demo | `hdf5.empty_demo` |

返回：`{path, format, ok, checks[], summary, errors, warnings, infos, score, issues[], detection}`。评分 = 100 − error×30 − warn×8 − info×1（下限 0），**score ≥ 80 且 `ok=true`** 才建议入库。

未知的 check 名会直接报 `KeyError` 并列出可用项——静默忽略会让 agent 以为检查跑过了。

---

## 4. `quality_report`

逐帧统计 + 异常窗口。

| 参数 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `path` | string | ✓ | 数据集路径 |
| `episodes` | int[] | | 只分析这些 episode |
| `features` | string[] | | 只分析这些特征（默认全部信号列） |
| `z` | number | | 鲁棒 z 阈值，默认 8.0 |
| `include_segments` | boolean | | 一并返回 moving/idle 段 |
| `skip_anomalies` | boolean | | 只要统计，跳过异常检测（大数据集提速） |

返回 `{dataset, stats, anomalies[], anomaly_counts, segments?}`，`dataset` 是精简后的 summary。

`stats[key]` = `{dims, frames, nan_count, inf_count, nan_ratio, min, max, mean, std, median, p05, p95, norm_mean, norm_max, near_zero_ratio}`。

`anomalies[]` = `{code, severity, episode, feature, start_frame, end_frame, frames, detail, start_s?, end_s?}`，`code ∈ {jump, stall, non_finite}`。有时间戳/fps 时附带秒数，方便和对齐视频。

记账列（`episode_index`、`frame_index`、`index`、`task_index`、`timestamp`）不参与异常检测——它们天然单调或分段恒定，扫描只会制造假阳性。

---

## 5. `segment_episodes`

按特征速度分位数切 moving / idle 段。

| 参数 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `path` | string | ✓ | 数据集路径 |
| `episode` | int | | 只处理单个 episode |
| `feature` | string | | 用哪个特征算速度（默认 `action`，没有则取第一个信号列） |
| `min_frames` | int | | 最短段长，默认 5 |

返回 `{episode, segments}`，其中 `segments` 是**按 episode 分组的字典**（键是 episode 序号的字符串），每段是：

```json
{"start_frame": 0, "end_frame": 29, "frames": 29, "label": "moving",
 "mean_speed": 3.46, "start_s": 0.0, "end_s": 0.967}
```

`label ∈ {moving, idle}`；`episode` 参数非空时只返回那一个 episode。

---

## 6. `labeling_tasks`

把异常窗口和运动段变成标注任务清单。

| 参数 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `path` | string | ✓ | 数据集路径 |
| `max_tasks` | int | | 上限，默认 100 |
| `z` | number | | 异常检测阈值 |
| `include_segments` | boolean | | 是否为 moving/idle 段生成"描述阶段"任务 |
| `out` | string | | 同时写入该文件路径 |

返回 `{dataset, task_count, tasks[], instructions}`。`tasks[]` 每项形如：

```json
{
  "kind": "verify_window",
  "episode": 3,
  "start_frame": 412,
  "end_frame": 438,
  "cues": ["jump", "observation.state"],
  "question": "Episode 3, frames 412-438: check discontinuity ... Is the recorded behaviour valid? Answer valid / invalid / needs_fix with a short reason.",
  "suggested_label": "check discontinuity (sensor dropout / teleport)"
}
```

`instructions` 里给了固定回答格式（每行一个 JSON 对象），便于把结果回灌成结构化标注。

---

## 7. `convert_dataset`

| 参数 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `source` | string | ✓ | 源数据集 |
| `destination` | string | ✓ | 目标目录 |
| `target` | string | | `jsonl`（默认，零依赖）或 `lerobot`（v2.1，需要 pyarrow） |
| `episodes` | int[] | | 只导出这些 episode |
| `overwrite` | boolean | | 覆盖已存在目录 |

返回 `{target, destination, episodes, frames, features[], dropped_features[]}`——`target` 回显实际写出的布局名（`embodied-jsonl` / `lerobot`）。只导出数值特征；相机流和其它非数值列会被记录在 `dropped_features` 里，而不是假装转过了。

---

## 8. `scaffold_project`

| 参数 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `destination` | string | ✓ | 目标目录 |
| `kind` | string | | `platform`（默认）/ `adapter` / `pipeline` |
| `name` | string | | 项目名，默认 `embodied-platform` |
| `overwrite` | boolean | | 覆盖已存在目录 |

返回 `{kind, destination, name, package, files[], next_steps[]}`。

---

## 9–11. 编目三件套

### `catalog_index`

| 参数 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `root` | string | ✓ | 要扫描的根目录 |
| `db` | string | ✓ | SQLite 文件路径（不存在则创建） |
| `max_depth` | int | | 递归深度，默认 4 |
| `force` | boolean | | 忽略 mtime/size 缓存，强制重扫 |
| `validate` | boolean | | 索引时顺带跑校验拿分数（慢） |

返回 `{db, root, added, updated, skipped, failed, total, failures[]}`。增量依据是数据集的 marker 文件（`meta/info.json`、`episodes.jsonl` 等，取最新的 mtime）+ 目录大小。

### `catalog_search`

| 参数 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `db` | string | ✓ | 编目文件 |
| `text` | string | | 在路径/任务/机器人型号里做子串匹配 |
| `format` | string | | 格式过滤（`lerobot` / `hdf5` / …） |
| `robot_type` | string | | 机器人型号 |
| `min_episodes` | int | | episode 数下限 |
| `min_duration` | number | | 总时长下限（秒） |
| `min_score` | number | | 质量分下限 |
| `limit` | int | | 返回条数，默认 50 |

返回 `{count, results[]}`。

### `catalog_stats`

只要 `db`。返回 `{total_datasets, by_format, total_episodes, total_frames, total_duration_s, robot_types[], last_indexed}`。

---

## 12. `list_formats`

无参数。返回 `{formats[], checks}`：7 种格式各含 `{format, description, detect, needs, extensions, can_read_frames}`，`checks` 是全部检查项与默认集合。模型在不确定时先调这个。

---

## 13. `plan_windows`

| 参数 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `path` | string | ✓ | 数据集路径 |
| `window` | int | | 窗口帧数，默认 16 |
| `stride` | int | | 步长，默认等于 window |
| `limit` | int | | 最多返回多少个窗口 |

返回 `{window, stride, count, windows[]}`，每个窗口是 `{episode, start_frame, end_frame, frames}`——可直接喂给训练侧的采样器。

---

## 错误码

| code | 含义 | 怎么办 |
| --- | --- | --- |
| `unsupported_format` | 路径存在但认不出 | 先用 `list_formats` 看支持范围；新格式写 adapter |
| `frames_unavailable` | 能识别但读不到帧 | 按返回的 `install` 装可选依赖 |
| `not_found` | 路径不存在 | 检查路径 |
| `bad_arguments` | 参数类型/取值不对 | 按提示修正 |
