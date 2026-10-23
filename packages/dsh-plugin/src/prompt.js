/**
 * System-prompt section for the plugin.
 *
 * Without this the model sees thirteen `embodied_*` tools and has to infer the
 * workflow from their names. The section is intentionally short: it states the
 * canonical order, the quality gate, and the one rule that prevents the most
 * common mistake (converting before validating).
 */

/** Canonical order of operations for an embodied dataset. */
const WORKFLOW = [
  '1. `embodied_detect_dataset` — identify the format before doing anything else. Never guess a format from a file extension; adapters disagree with extensions.',
  '2. `embodied_summarize_dataset` — read fps, robot type, feature schema and episode lengths. Feature *names* matter downstream (observation.state / action are the contract).',
  '3. `embodied_validate_dataset` — structural conformance (LeRobot spec, episode numbering, dtype vocabulary). Fix errors before exporting; they do not disappear on conversion.',
  '4. `embodied_quality_report` — statistical sanity (non-finite values, constant features, suspicious ranges) plus anomaly windows (jumps, stalls).',
  '5. `embodied_segment_episodes` / `embodied_labeling_tasks` — turn motion segmentation into human-reviewable labeling tasks.',
  '6. `embodied_convert_dataset` — export to `lerobot` or `jsonl`. Conversion only carries numeric features; images/videos are referenced, not re-encoded.',
  '7. `embodied_catalog_index` / `embodied_catalog_search` — keep a queryable index of everything ingested so later sessions do not re-scan the disk.',
].join('\n');

const CONVENTIONS = [
  '- A dataset with unfixed `error` issues must not be published; `score >= 80` is the publish gate.',
  '- `observation.state` (proprioception) and `action` must both exist for a dataset to be trainable; a vision-only dataset is a red flag.',
  '- Report `episode_index`, `frame_index` and `timestamp` in the canonical JSONL so episodes stay splicable.',
  '- Prefer reading frames through the tools rather than re-implementing parquet/HDF5 parsing inline: the tool path already handles optional dependencies.',
].join('\n');

export function buildPromptSection(config, toolNames) {
  const names = toolNames.length > 0 ? toolNames.join(', ') : '(none loaded — call `embodied_self_check`)';
  return [
    '## Embodied data toolbox',
    '',
    'Tools for developing embodied-AI data platforms: dataset format detection, inspection, validation, quality control, segmentation, labeling-task generation, conversion and cataloging. Supported formats: LeRobot v2/v3, HDF5/robomimic, RLDS/TFRecord, rosbag1/rosbag2, MCAP and the canonical JSONL layout.',
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
