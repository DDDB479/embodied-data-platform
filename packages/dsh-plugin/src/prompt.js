/**
 * System-prompt section for the plugin.
 *
 * States the platform workflow: inspect, decode streams/media, gate quality,
 * close the annotation loop, export a versioned training dataset and keep the
 * catalog/history auditable.
 */

const WORKFLOW = [
  '1. `embodied_detect_dataset` - identify the format first. Never infer it from a file extension.',
  '2. `embodied_summarize_dataset` - read fps, robot type, features, modalities and episode lengths.',
  '3. `embodied_validate_dataset` - structural checks before any conversion or catalog write.',
  '4. `embodied_inspect_media` - inventory image/video/depth/point-cloud/language data; ffprobe runs only when installed.',
  '5. `embodied_quality_report` - streamed statistics and anomaly windows. Use chunk_frames/max_frames for bounded memory.',
  '6. `embodied_quality_gate` - apply expected fps, required features/kinds, score, NaN and anomaly policy.',
  '7. `embodied_decode_samples` - inspect rosbag2/MCAP CDR topics or RLDS step samples, including depth and point-cloud metadata.',
  '8. `embodied_segment_episodes` / `embodied_labeling_tasks` - create review work with stable task_id values.',
  '9. `embodied_apply_annotations` - reconcile valid/invalid/needs_fix labels, detect conflicts and write reviewed output atomically.',
  '10. `embodied_convert_dataset` - export JSONL, LeRobot v2.1 or LeRobot v3.0 (stats + train/val/test splits).',
  '11. `embodied_catalog_index` / `embodied_catalog_search` / `embodied_catalog_history` - keep versions, runs and lineage auditable.',
  '12. `embodied_storage_pull` / `embodied_submit_job` - materialize file/http(s)/s3 datasets atomically; run long read-only analyses in the background.',
].join('\n');

const CONVENTIONS = [
  '- Do not publish datasets with unfixed error issues; the default gate requires score >= 80 and zero errors.',
  '- `observation.state` and `action` are the default training contract; add required features/kinds explicitly to gate rules.',
  '- rosbag2 CDR decoding is dependency-free; MCAP decoding additionally needs the `mcap` package. RLDS common Example/SequenceExample records decode without TensorFlow.',
  '- ffprobe is optional. A missing binary is a structured install hint, not a failed dataset inspection.',
  '- CDR/RLDS readers enforce message/record/point limits. Raise them only with a concrete memory budget.',
  '- Annotation labels are last-wins with a conflict log; use task_id rather than list position for durable feedback.',
  '- LeRobot v3 exports include meta/stats.json, episode metadata and splits. Validate the exported directory after conversion.',
  '- Catalog versions are immutable. Re-indexing changed data creates a new version and lineage edge.',
  '- Object-store pulls are checksummed and source-manifested; do not edit a pulled dataset in place if you need provenance.',
].join('\n');

export function buildPromptSection(config, toolNames) {
  const names = toolNames.length > 0 ? toolNames.join(', ') : '(none loaded - call `embodied_self_check`)';
  return [
    '## Embodied data platform toolbox',
    '',
    'Tools for embodied-AI data platforms: format detection, structural validation, streamed quality control, configurable gates, CDR/RLDS decoding, media inventory, annotation writeback, LeRobot export and versioned cataloging.',
    '',
    `Available: ${names}`,
    '',
    '### Workflow',
    WORKFLOW,
    '',
    '### Conventions',
    CONVENTIONS,
    '',
    'Ask before converting, annotating or pulling into a path outside the session workspace. Tools never modify the source dataset unless the destination is explicit.',
  ].join('\n');
}
