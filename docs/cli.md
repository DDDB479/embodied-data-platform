# 命令行参考

安装后入口是 `embodied`；没安装时用 `PYTHONPATH=src python -m embodied_core.cli`。

```text
embodied {formats,detect,info,validate,quality,segment,label,convert,scaffold,windows,checks,catalog} ...
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

## quality

```bash
embodied quality ./data/aloha [--episodes 0 1 2] [--feature action] [--z 8] [--no-anomalies] [--segments] [--json]
```

输出 `{stats, anomalies, anomaly_counts}`，`--segments` 时多一个 `segments`。统计与异常都只看信号列，跳过 `episode_index` / `frame_index` / `timestamp` 这类记账列。

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
```

`index` / `stats` / `remove` 默认就输出 JSON（这些命令本来就没有"好看"的形态），`search` / `export` 带 `--json` 时输出 `{count, results[]}` / `{datasets[], count}`。

`index` 的增量逻辑：每个数据集记一个 marker 文件（`meta/info.json`、`episodes.jsonl`、`dataset.json` 等，取 mtime 最新者）的 mtime 与目录总大小，两者都没变就跳过。追加 episode 会更新 `episodes.jsonl`，因此能被识别为 `updated`。
