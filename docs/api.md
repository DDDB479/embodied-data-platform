# HTTP Catalog API 与对象存储

## 启动

```bash
embodied api --db catalog.db --host 127.0.0.1 --port 8000
# 等价入口
embodied-api --db catalog.db
```

设置 `EMBODIED_API_TOKEN` 或 `--token` 后，所有请求都需要：

```http
Authorization: Bearer <token>
# 或
X-API-Key: <token>
```

服务是只读的；索引、转换、标注写入仍走 CLI / MCP / DSH 的写安全路径。

## 端点

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/health` | 服务名、版本、存活状态 |
| GET | `/v1/stats` | catalog 汇总 |
| GET | `/v1/datasets` | 分页/过滤查询：`limit<=200`、`offset`、`text`、`format`、`robot_type`、`min_score` |
| GET | `/v1/datasets/{id}` | 当前数据集/版本详情 |
| GET | `/v1/datasets/{id}/versions` | 不可变版本历史 |
| GET | `/v1/datasets/{id}/lineage` | 版本谱系 |
| GET | `/v1/runs` | 运行记录，可按 `kind`/`status` 过滤 |

响应带强 ETag；请求带 `If-None-Match` 时返回 `304`。CORS 用 `--cors` 配置。

## 对象存储

```bash
embodied storage info s3://bucket/prefix --json
embodied storage pull file:///data/night_run ./cache/night_run
embodied storage pull https://example.com/episode.bin ./cache/episode.bin
embodied storage pull s3://bucket/prefix ./cache/dataset --overwrite
```

- 支持 `file`、`http(s)`、`s3`；S3 需要 `pip install 'embodied-core[s3]'`。
- 拉取先写临时目录，校验 SHA-256 后原子替换目标。
- 目标目录生成 `.embodied-source.json`，记录来源、文件大小、哈希与可选 ETag。
- 路径穿越 key、超限下载、缺失 boto3 都会返回结构化错误；不会覆盖目标除非显式 `--overwrite`。

## 平台诊断

```bash
embodied doctor --json
```

输出可选依赖、ROS 2 CDR 类型、RLDS/TFRecord 上限、ffprobe、S3 可用性和 HTTP API 能力。MCP 的 `list_formats` 也会附带同一份 `platform` 信息。
