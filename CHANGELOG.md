# Changelog

## Unreleased (0.5.0)

### Stability and operations

- Added `embodied doctor` and `list_formats.platform` capability reports for optional dependencies, CDR types, ffprobe, S3, HTTP API and resource limits.
- Added configurable limits for CDR messages (`EMBODIED_MAX_MESSAGE_BYTES`), point clouds (`EMBODIED_MAX_POINTS`), TFRecord records (`EMBODIED_MAX_RECORD_BYTES`) and object downloads (`EMBODIED_MAX_DOWNLOAD_BYTES`).
- LeRobot v3 export and object-store materialization now stage writes and publish atomically; v3 export runs structural validation before returning.
- Annotation feedback supports `previous` merge, stable task IDs, last-wins duplicates, conflict/coverage reporting and atomic output.
- HTTP Catalog API adds stable ETags, bounded pagination, Bearer/X-API-Key auth and read-only endpoint semantics.
- Unknown CDR schemas and malformed payloads now fail with stable `unsupported_message_type` / `decoder_error` codes instead of silent numeric output.

## 0.4.0 - 2026-10-29

### Stream decoding and platform features

- Added a dependency-free ROS 2 CDR reader and decoders for JointState, Imu, Odometry, Pose/Twist/Vector3, Image/depth, PointCloud2, LaserScan, scalar, array and String/language topics.
- rosbag2 now exposes CDR-backed `frames()` and `topic_samples()`; MCAP reuses the decoder when the `mcap` package is installed.
- Added a lightweight TFRecord + TensorFlow Example/SequenceExample decoder for RLDS, including step-aligned numeric columns and language text without TensorFlow.
- Added optional ffprobe video metadata probing and image/video/depth/point-cloud/language modality inventory.
- Added `apply_annotations` and `embodied review` for closed-loop labeling feedback.
- Added LeRobot v3.0 conversion with `meta/stats.json`, episode metadata, train/validation/test splits and post-export validation.
- Added a read-only HTTP Catalog API (`embodied api` / `embodied-api`) and atomic file/http(s)/s3 object-store pulls with SHA-256 source manifests.

## 0.3.0 - 2026-10-29

### Async jobs and progress

- Added `submit_job`, `job_status`, `cancel_job` and `list_jobs` MCP tools for background `quality_report`, `validate_dataset` and `quality_gate` runs.
- Added MCP `notifications/progress` support, driven by a `progressToken` in `tools/call` `_meta`; synchronous quality and validation calls emit progress as well.
- Jobs have bounded concurrency, cooperative cancellation, structured errors and a configurable TTL (`EMBODIED_JOB_WORKERS`, `EMBODIED_JOB_TTL_S`).
- The DSH client now parses progress notifications, retains the latest event per token and exposes `onProgress(...)`/`lastProgress(...)`.

### Incremental quality cache

- Added a SQLite quality-report cache keyed by resolved dataset path, a full file fingerprint (relative path, size and mtime), normalized parameters and an algorithm version.
- Cache hits skip opening the dataset entirely; any file or parameter change invalidates the entry. `refresh_cache` forces recomputation.
- Added `use_cache`/`refresh_cache` to `quality_report`, `EMBODIED_QUALITY_CACHE` for the path, and `embodied cache stats|clear` for operations.
- The MCP surface deliberately rejects arbitrary `cache_path` values so a read-only tool cannot become an unapproved write.

### Streaming reads and memory bounds

- Added `DatasetObject.iter_frames()` yielding bounded `FrameChunk` objects with episode-local frame offsets; JSONL, LeRobot parquet and HDF5 adapters implement native chunked readers.
- Added `EMBODIED_CHUNK_FRAMES` (default 50000) and `EMBODIED_MAX_FRAMES` (default 2000000, `0` = unlimited); `frames(max_frames=...)` raises the structured `FrameLimitExceeded` instead of growing without bound.
- Quality reports now merge per-chunk statistics and anomaly windows. Min/max/mean/std and non-finite counts stay exact; cross-chunk percentile estimates are marked explicitly.

### Catalog schema v2

- Split the catalog into logical datasets, immutable `dataset_versions`, `runs`, and parent/child `lineage` edges while preserving existing search/stats/manifest APIs.
- A changed file fingerprint creates a new version and a `supersedes` lineage edge; unchanged datasets are still skipped incrementally.
- Added `catalog versions|lineage|runs|history` CLI commands, `catalog_history` MCP tool, and automatic in-place migration from v1 databases.

### Configurable quality gate

- Added `quality_gate` MCP tool and `embodied gate` command with JSON/TOML rules for expected fps/tolerance, required feature keys or kinds, minimum score/episodes/frames, NaN ratio, and anomaly-window limits.
- Gate reports keep stable failure codes such as `gate.fps_mismatch`, `gate.required_features`, `gate.nan_ratio` and `gate.anomaly_windows`; mismatches do not raise.
- Added `quality_gate` to the read-only async job surface and DSH path-safety rules.

### Safety and compatibility

- Background jobs are restricted to read-only analyses, so they cannot bypass workspace write approvals.
- DSH normalizes nested `submit_job.arguments.path` against the session workspace.
- MCP tool surface is now 19 tools (20 with the DSH `embodied_self_check`).
- Package version is now `0.3.0`.

## 0.1.0 - 2026-10-29

### Packaging

- The DSH npm package now bundles `python/embodied_core` during `prepack`, so a registry install no longer needs a repository checkout or a separate `pip install`.
- `resolveConfig` reports whether the Python root came from config, the bundled `python/` tree, a source checkout, or a `pythonPath`/`PYTHONPATH` install.
- Added `node tools/check_clean_install.mjs`, which packs the package, installs it into a temporary app, starts the MCP server and probes a fixture.

### Safety

- DSH tool registration now propagates MCP annotations: only `readOnlyHint: true` tools are concurrency-safe; write tools remain serial.
- Write destinations outside the DSH workspace require an `allowed-once` approval. Missing approval support, a denied request or read-only mode fails closed.
- Relative path arguments resolve against the session workspace rather than the Python child process's install directory.

### Engineering

- Added a GitHub Actions matrix for Windows/Linux, Python 3.10/3.12, optional-reader and minimal installs.
- Added Node tests for bundled-root discovery, path normalization, concurrency declarations and approval branches.
- Added packaging invariants to the Python test suite.
