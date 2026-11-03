# v0.4–v0.6 平台化范围

## v0.4（功能平台化）

| 能力 | 交付内容 |
| --- | --- |
| ROS 2 CDR | rosbag2 内置 CDR；JointState / IMU / Odometry / Image / PointCloud2 / LaserScan / 常用标量和数组类型 |
| MCAP | 安装 `mcap` 后复用 CDR 解码器；无依赖时保留 metadata fallback |
| RLDS | 内置 TFRecord + Example/SequenceExample protobuf 子集解码，不强制 TensorFlow |
| 媒体 | ffprobe 可选探测；image/video/depth/point_cloud/language 模态清单 |
| 标注闭环 | 稳定 task ID、valid/invalid/needs_fix、去重 last-wins、冲突与覆盖率 |
| 训练导出 | LeRobot v3.0、`meta/stats.json`、episode metadata、train/validation/test splits |
| HTTP API | 标准库只读 Catalog API，`/health`、`/v1/datasets`、`/v1/runs` 等 |
| 对象存储 | file / http(s) / s3 URI，原子发布、SHA-256、来源 manifest |

## v0.5（稳定性收口）

- 解码器版本/能力矩阵由 `embodied doctor` 与 `list_formats.platform` 输出。
- CDR 消息、TFRecord、点云和下载均有可配置上限，异常输入返回稳定错误码。
- LeRobot v3 与对象存储使用 staging + 原子替换，避免半成品目录。
- 标注回写支持 previous 合并、重复标签 last-wins 和冲突记录，可重复执行。
- HTTP API 提供 Bearer / X-API-Key、分页、强 ETag；只读，不在 HTTP 路径写 catalog。
- 代码和文档均要求“有依赖走真解码、无依赖结构化降级”，不得静默返回错误数值。

## v0.6（生产操作层）

| 模块 | 交付内容 |
| --- | --- |
| DAG 流水线 | 持久化 spec/run/node output；tool/processor/human/noop 节点；依赖校验、模板变量、重试、人工等待与恢复 |
| 人工任务 | 稳定 task_id、预标/置信度、uncertainty/stratified 抽样、分配、完成、SLA 与标签导出 |
| 处理器协议 | `register_processor` / `load_processor`；内置 dataset inventory、ffprobe video probe、frame plan |
| 交付包 | `delivery-manifest.json`、逐文件 SHA-256、质量门证据、JSONL/LeRobot/Zarr/MCAP inventory |
| 训练格式 | Zarr v2 数值导出（chunk/metadata 可独立读取） |
| 可观测性 | pipeline 状态/失败节点/平均耗时；task 覆盖率与逾期；delivery check |

## 稳定性约定

- 所有可选依赖缺失时返回 `FramesUnavailable` / `StorageUnavailable`，并带安装提示。
- 解码器只承诺已注册的常见 ROS 2 类型；未知 schema 明确失败，不做猜测性字段映射。
- 数据源文件只读；写入必须提供显式 destination/out，并经过 CLI、MCP 或 DSH 的安全策略。
- 质量缓存键包含算法版本；解码/导出行为变化时递增版本，避免旧缓存污染。
