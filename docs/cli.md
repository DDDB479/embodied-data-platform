# 命令行参考

安装后入口是 `embodied`；没安装时用 `PYTHONPATH=src python -m embodied_core.cli`。

```text
embodied {formats,detect,info,validate,gate,quality,segment,label,convert,scaffold,windows,checks,catalog} ...
embodied --version
```

所有子命令都接受 `--json`（给程序/模型看），不带时输出人类可读的对齐文本。

## 退出码

| 码 | 含义 | 典型场景 |
| --- | --- | --- |
| 0 | 成功 / 校验通过 | `validate` 的 `ok=true` |
| 1 | 校验未通过 | 数据集有 error 级问题 |
| 2 | 用法错误或路径不存在 | 打错路径、未知 check 名 |
| 3 | 不支持的格式 | `UnsupportedFormat` |

CI 里可以直接：

```bash
embodied validate "$DATA" --json || exit $?
```

## formats / checks

```bash
embodied formats            # 7 种格式：探测依据、所需可选依赖、能否读帧
embodied checks             # 8 组检查与默认启用集合
```

两者都是纯查询，不碰数据。

## detect

```bash
embodied detect ./data/aloha --json
```

输出 `{format, version, confidence, evidence[], detail}`。认不出时退出码 3 并抛 `UnsupportedFormat`——这里必须给答案，不能像 `validate` 那样返回"不通过"。

## info

```bash
embodied info ./data/aloha [--json] [--no-episodes]
```

`--json` 时含 `episode_list`（每个 episode 的 index/length/task/duration_s）。

## validate

```bash
embodied validate ./data/aloha [--format lerobot] [--check metadata --check episodes] [--frames] [--json]
```

`--check` 可重复，取值见 `embodied checks`；`--frames` 追加逐帧检查（需要可选依赖，慢）。

## gate

```bash
embodied gate ./data/aloha [--rules rules.json|rules.toml]
                    [--expected-fps 30] [--fps-tolerance 0.5]
                    [--require-feature observation.state --require-feature action]
                    [--min-score 80] [--max-nan-ratio 0.01]
                    [--frames] [--chunk-frames 50000] [--max-frames 2000000]
                    [--no-cache] [--json]
```

规则文件示例：

```json
{
  "expected_fps": 30,
  "required_features": ["observation.state", "action"],
  "min_score": 80,
  "max_errors": 0,
  "max_nan_ratio": 0.01,
  "max_jump_windows": 0,
  "max_stall_windows": 5
}
```

退出码：`0` 通过，`1` 规则拒绝，`2` 用法/规则文件错误，`3` 不支持的格式。输出中的 `gate.failures[]` 带稳定 `code`，适合 CI 解析。

## quality

```bash
embodied quality ./data/aloha [--episodes 0 1 2] [--feature action] [--z 8] [--no-anomalies] [--segments]
                          [--no-cache] [--cache-path FILE] [--refresh-cache] [--json]
```

`--chunk-frames` 控制每块帧数（默认 50000，也可用 `EMBODIED_CHUNK_FRAMES`）；`--max-frames` 是硬上限（默认 2000000，`0` 表示不限，也可用 `EMBODIED_MAX_FRAMES`）。报告中的 `scan` 字段记录实际块数、扫描帧数和上限。

输出 `{dataset, stats, anomalies, anomaly_counts, cache}`，`--segments` 时多一个 `segments`。统计与异常都只看信号列，跳过 `episode_index` / `frame_index` / `timestamp` 这类记账列。

默认开启增量缓存：数据集文件指纹和参数不变时直接复用报告。`--no-cache` 本次禁用，`--refresh-cache` 重算并覆盖，`--cache-path` 指定 SQLite 文件（默认在用户缓存目录）。

## cache

```bash
embodied cache stats [--path FILE]
embodied cache clear [--path FILE]
```

`stats` 返回 `{path, entries, hits, bytes, oldest, newest}`；`clear` 删除所有报告并返回删除条数与剩余统计。

## segment

```bash
embodied segment ./data/aloha [--episode 0] [--feature action] [--min-frames 5] [--json]
```

输出 `{episode, segments}`，`segments` 是「episode → 段列表」的字典。

## label

```bash
embodied label ./data/aloha [--max-tasks 100] [--z 8] [--segments] [--out tasks.json] [--json]
```

`--out` 时不打印整份清单，只打印落盘路径与任务数；写出的就是 `{dataset, task_count, tasks, instructions}`。

## convert

```bash
embodied convert ./data/aloha ./out/lerobot --target lerobot [--episodes 0 1] [--overwrite] [--format hdf5] [--json]
```

- `--target jsonl`：canonical JSONL，零依赖，落盘 `dataset.json` + `episodes.jsonl`。
- `--target lerobot`：v2.1 布局（每 episode 一个 parquet），需要 `pyarrow`。
- `--format` 强制源格式，跳过自动识别。
- 只导数值特征；相机流进 `dropped_features`。

## scaffold

```bash
embodied scaffold ./my-platform [--kind platform|adapter|pipeline] [--name acme] [--overwrite] [--json]
```

返回 `{kind, destination, name, package, files[], next_steps[]}`。

## windows

```bash
embodied windows ./data/aloha --window 16 --stride 8 [--limit 100] [--json]
```

输出 `{window, stride, count, windows[]}`，窗口是 `{episode, start_frame, end_frame, frames}`。

## catalog

```bash
embodied catalog index  <root> [--db catalog.db] [--depth 4] [--force] [--validate]
embodied catalog search [--db catalog.db] [--text ...] [--format ...] [--robot ...]
                        [--min-episodes N] [--min-duration S] [--min-score N]
                        [--limit 50] [--order-by indexed_at] [--ascending]
embodied catalog stats  [--db catalog.db]
embodied catalog remove <path> [--db catalog.db]
embodied catalog export [--db catalog.db] [--limit 1000] [--out manifest.json]
embodied catalog versions [--db catalog.db] [--path DATASET] [--limit 50]
embodied catalog lineage  [--db catalog.db] [--path DATASET] [--version-id N] [--limit 100]
embodied catalog runs     [--db catalog.db] [--kind index] [--status succeeded] [--limit 20]
embodied catalog history  [--db catalog.db] [--path DATASET] [--limit 20]
```

自 v0.3 起编目为 schema v2：`datasets` 保存逻辑身份和当前版本，`dataset_versions` 保存不可变快照，`runs` 保存索引/迁移运行，`lineage` 保存 `supersedes` 等版本边。旧 v1 数据库在首次打开时自动迁移。

`index` / `stats` / `remove` 默认就输出 JSON（这些命令本来就没有"好看"的形态），`search` / `export` 带 `--json` 时输出 `{count, results[]}` / `{datasets[], count}`。

`index` 的增量逻辑：每个数据集记一个 marker 文件（`meta/info.json`、`episodes.jsonl`、`dataset.json` 等，取 mtime 最新者）的 mtime 与目录总大小，两者都没变就跳过。追加 episode 会更新 `episodes.jsonl`，因此能被识别为 `updated`。
