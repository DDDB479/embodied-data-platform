/**
 * System-prompt section for the plugin.
 *
 * The section states the canonical order, the configurable quality gate, and
 * the rules that prevent the most common mistakes (converting before
 * validating, and re-scanning datasets that already have a catalog version).
 */

/** Canonical order of operations for an embodied dataset. */
const WORKFLOW = [
  '1. `embodied_detect_dataset` - identify the format before doing anything else. Never guess a format from a file extension; adapters disagree with extensions.',
  '2. `embodied_summarize_dataset` - read fps, robot type, feature schema and episode lengths. Feature names matter downstream (observation.state / action are the training contract).',
  '3. `embodied_validate_dataset` - structural conformance (LeRobot spec, episode numbering, dtype vocabulary). Fix errors before exporting; they do not disappear on conversion.',
  '4. `embodied_quality_report` - statistical sanity plus anomaly windows. Reports are fingerprint-cached and streamed in bounded chunks; use `chunk_frames` / `max_frames` for large datasets.',
  '5. `embodied_quality_gate` - apply publish policy: expected fps, required feature keys/kinds, minimum score, NaN ratio and anomaly-window limits. This is the decision point for ingest or release.',
  '6. `embodied_segment_episodes` / `embodied_labeling_tasks` - turn motion segmentation and anomalies into human-reviewable labeling tasks.',
  '7. `embodied_convert_dataset` - export to `lerobot` or `jsonl`. Conversion only carries numeric features; images/videos are referenced, not re-encoded.',
  '8. `embodied_catalog_index` / `embodied_catalog_search` - maintain schema-v2 catalog records. `embodied_catalog_history` shows immutable versions, runs and lineage without re-scanning the disk.',
  '9. `embodied_submit_job` -> `embodied_job_status` - for large datasets, run quality_report, validate_dataset or quality_gate in the background and poll progress instead of blocking the turn. Jobs are read-only.',
].join('\n');

const CONVENTIONS = [
  '- A dataset with unfixed `error` issues must not be published; the default gate requires `score >= 80` and zero errors.',
  '- `observation.state` (proprioception) and `action` must both exist for a dataset to be trainable; add them to `required_features` or `required_feature_kinds` in gate rules.',
  '- Report `episode_index`, `frame_index` and `timestamp` in canonical JSONL so episodes stay splicable.',
  '- Prefer reading frames through the tools rather than re-implementing parquet/HDF5 parsing inline: the tool path already handles optional dependencies and chunked reads.',
  '- A cache hit is valid only for the same dataset fingerprint and parameters; change z, features, episodes or files and the report is recomputed automatically.',
  '- Catalog versions are immutable. Re-indexing a changed dataset creates a new version and a lineage edge; use `catalog_history` before assuming a path has only one revision.',
].join('\n');

export function buildPromptSection(config, toolNames) {
  const names = toolNames.length > 0 ? toolNames.join(', ') : '(none loaded - call `embodied_self_check`)';
  return [
    '## Embodied data toolbox',
    '',
    'Tools for developing embodied-AI data platforms: dataset format detection, inspection, validation, quality control and configurable gates, segmentation, labeling-task generation, conversion and versioned cataloging. Supported formats: LeRobot v2/v3, HDF5/robomimic, RLDS/TFRecord, rosbag1/rosbag2, MCAP and the canonical JSONL layout.',
    '',
    `Available: ${names}`,
    '',
    '### Workflow',
    WORKFLOW,
    '',
    '### Conventions',
    CONVENTIONS,
    '',
    'Ask before converting or indexing a path outside the session workspace. Paths may be absolute; tools never modify the source dataset.',
  ].join('\n');
}
