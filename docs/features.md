# 插件功能总览

`dsh-embodied-data` 是具身智能数据平台的 DSH/MCP 插件。插件本身是轻量 JS 桥接，真正能力来自统一的 `embodied_core` Python 核心，因此 DSH、Codex、Claude Code 和其他 MCP 客户端使用同一套行为。

- 当前版本：`0.5.0`
- 工具规模：23 个 MCP 工具 + 1 个插件自检工具，共 24 个 `embodied_*` 工具
- 支持格式：LeRobot v2/v3、HDF5/robomimic、rosbag1/2、MCAP、RLDS/TFRecord、canonical JSONL

## 能力矩阵

| 模块 | 能力 |
| --- | --- |
| 格式识别 | 目录结构 + 文件魔数探测；不依赖扩展名 |
| 结构校验 | 8 组检查、稳定 issue code、0–100 健康分 |
| 质量分析 | 逐特征统计、NaN/inf、常量、极值、跳变/停滞/非有限窗口、运动切段 |
| 质量门 | 期望 fps、必需特征 key/kind、最低分、episode/frame 下限、NaN 与异常窗口阈值 |
| 流式读取 | `FrameChunk`、`iter_frames()`、帧数上限、分块统计聚合 |
| ROS 2 CDR | JointState、Imu、Odometry、Pose/Twist/Vector3、Image/深度图、PointCloud2、LaserScan、语言与标量 topic |
| RLDS | TFRecord + Example/SequenceExample 轻量 protobuf 解码；普通场景不需要 TensorFlow |
| 媒体 | image/video/depth/point_cloud/language 模态清单；ffprobe 可选探测 |
| 标注闭环 | 稳定 task ID、valid/invalid/needs_fix、last-wins、冲突与覆盖率、原子写回 |
| 训练导出 | canonical JSONL、LeRobot v2.1、LeRobot v3.0（stats + splits + 导出后校验） |
| 编目 | schema v2：数据集、不可变版本、运行记录、谱系、历史与检索 |
| HTTP API | 只读 Catalog API、Bearer/X-API-Key、CORS、分页、强 ETag |
| 对象存储 | file/http(s)/s3 原子拉取、重试、SHA-256、来源 manifest |
| 异步任务 | `job_id`、进度通知、协作式取消、只读任务隔离 |
| 平台诊断 | `embodied doctor`、依赖/解码器/资源上限/能力矩阵 |

## 工具清单

### 数据集分析

- `detect_dataset`：识别格式与版本，返回 confidence/evidence。
- `summarize_dataset`：特征、episode、fps、模态、任务和来源。
- `validate_dataset`：结构、元数据、特征词表、episode、fps 和格式规范检查。
- `quality_report`：流式统计与异常窗口，按数据集指纹增量缓存。
- `quality_gate`：执行可配置入库/发布策略。

### 流与媒体

- `decode_samples`：rosbag2/MCAP CDR topic 或 RLDS episode 样本。
- `inspect_media`：深度、点云、语言、图像、视频模态与可选 ffprobe 探测。

### 标注与转换

- `segment_episodes`：moving/idle 切段。
- `labeling_tasks`：生成带稳定 `task_id` 的复核任务。
- `apply_annotations`：标签对账、冲突、覆盖率与 reviewed manifest。
- `convert_dataset`：JSONL、LeRobot v2.1、LeRobot v3.0。
- `plan_windows`：固定长度训练窗口规划。

### 编目与平台

- `catalog_index`：增量扫描并生成不可变版本。
- `catalog_search`：按文本、格式、型号、episode、时长、分数检索。
- `catalog_stats`：数据集/版本/运行/谱系统计。
- `catalog_history`：版本、运行记录和 parent/child 谱系。
- `list_formats`：格式支持矩阵、检查项、平台能力。
- `storage_pull`：file/http(s)/s3 原子拉取。

### 异步与自检

- `submit_job`、`job_status`、`cancel_job`、`list_jobs`：后台只读分析与进度。
- `self_check`：解释器、import 路径和 MCP 进程状态。

## 典型工作流

```text
新数据入库
  detect -> summarize -> validate -> inspect_media -> quality_report
  -> quality_gate -> convert -> catalog_index

机器人录制
  detect(bag/mcap) -> decode_samples -> validate -> quality_gate
  -> convert(lerobot-v3) -> catalog_index

人工复核
  segment/labeling_tasks -> reviewer labels -> apply_annotations
  -> flagged episodes / reviewed manifest

远程数据
  storage_pull -> validate -> quality_gate -> catalog_index

大数据
  submit_job -> job_status(progress) -> result -> catalog/history
```

## 安全与稳定性

- 只读工具可并发；写工具串行执行。
- 写目标在工作区外时需要一次性审批；无审批服务时失败关闭。
- 源数据集只读；转换、标注和对象存储均写入显式目标。
- LeRobot v3 导出和对象存储拉取使用 staging + 原子发布。
- 标注回写支持 previous 合并和幂等 last-wins。
- CDR、TFRecord、点云、下载大小均有环境变量上限。
- 可选依赖缺失时返回结构化安装提示，不静默产生错误结果。

## 可选依赖与边界

| 依赖 | 用途 |
| --- | --- |
| `pyarrow` | LeRobot parquet 读取与 v2/v3 导出 |
| `h5py` | HDF5 帧读取 |
| `mcap` | MCAP 消息遍历与 CDR 解码 |
| `boto3` | S3 对象存储 |
| `ffprobe` | 视频宽高、时长、codec、帧率、码率探测 |

已知边界：

- 未注册的自定义 CDR schema 会显式返回 `unsupported_message_type`，不做猜测性字段映射。
- rosbag1 仍以 metadata 统计为主。
- 视频只做元数据探测，不转码。
- S3 需要 boto3 和有效凭据；ffprobe 需要单独安装 FFmpeg。

## 验证状态

- Python 单元测试：169
- DSH 插件测试：16
- 端到端检查：21
- MCP 工具：23；DSH 工具：24
- 干净安装：`dsh-embodied-data-0.5.0.tgz`
- ffprobe 实测：2 秒 320×240 / 30fps H.264 + AAC，正确返回 60 帧、时长 2.0s、码率与音视频流信息
