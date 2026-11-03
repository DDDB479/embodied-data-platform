/**
 * System-prompt section for the plugin.
 *
 * v0.6 teaches the model to treat the toolkit as a data-production platform:
 * inspect, gate, decode, queue human work, run a resumable DAG, deliver a
 * checksummed pack and preserve catalog lineage.
 */

const WORKFLOW = [
  '1. `embodied_detect_dataset` - identify the format first. Never infer it from a file extension.',
  '2. `embodied_summarize_dataset` - read fps, robot type, features, modalities and episode lengths.',
  '3. `embodied_validate_dataset` - structural checks before conversion or catalog writes.',
  '4. `embodied_inspect_media` - inventory image/video/depth/point-cloud/language data; ffprobe is optional.',
  '5. `embodied_quality_report` - streamed statistics and anomaly windows with bounded memory.',
  '6. `embodied_quality_gate` - apply expected fps, required features/kinds, score, NaN and anomaly policy.',
  '7. `embodied_decode_samples` - inspect rosbag2/MCAP CDR topics or RLDS step samples.',
  '8. `embodied_labeling_tasks` - create review tasks with stable task_id values.',
  '9. `embodied_task_queue` - ingest with prelabels/uncertainty sampling, assign reviewers and track completion.',
  '10. `embodied_apply_annotations` - reconcile labels, conflicts and reviewed output.',
  '11. `embodied_pipeline_run` - execute a persistent DAG of tool/processor/human nodes; resume waiting human nodes with `run_id` and `human_results`.',
  '12. `embodied_convert_dataset` - export JSONL, LeRobot v2.1 or v3.0 with stats/splits.',
  '13. `embodied_delivery` - build or verify a versioned delivery pack with per-file SHA-256 and manifest.',
  '14. `embodied_catalog_index` / `embodied_catalog_search` / `embodied_catalog_history` - keep versions, runs and lineage auditable.',
  '15. `embodied_storage_pull` / `embodied_submit_job` - materialize object-store datasets atomically; run long read-only analyses in the background.',
].join('\n');

const CONVENTIONS = [
  '- Do not publish datasets with unfixed error issues; the default gate requires score >= 80 and zero errors.',
  '- Pipeline definitions are durable artifacts. Prefer stable node ids and explicit `depends_on` over ad-hoc shell scripts.',
  '- A human node pauses as `waiting_human`; inspect its manifest, then resume the same run with reviewer output. Do not start a duplicate run.',
  '- `pipeline_run` writes to its run store. Write tool nodes additionally require `allow_write=true`; keep read-only pipelines default.',
  '- Human labels are keyed by task_id; duplicate labels are last-wins with conflicts reported. Use the task queue for assignment and SLA tracking.',
  '- Delivery is not just conversion: verify `delivery-manifest.json`, hashes, formats, quality gate and expected artifact count before shipping.',
  '- rosbag2 CDR decoding is dependency-free; MCAP needs `mcap`. RLDS common Example/SequenceExample records decode without TensorFlow.',
  '- CDR/RLDS readers enforce message/record/point limits. Raise them only with a concrete memory budget.',
  '- Catalog versions are immutable. Re-indexing changed data creates a new version and lineage edge.',
].join('\n');

export function buildPromptSection(config, toolNames) {
  const names = toolNames.length > 0 ? toolNames.join(', ') : '(none loaded - call `embodied_self_check`)';
  return [
    '## Embodied data production platform',
    '',
    'Tools for embodied-AI data production: format detection, validation, streamed quality control, gates, CDR/RLDS decoding, media inventory, human review operations, resumable pipelines, training export, delivery packs and versioned cataloging.',
    '',
    `Available: ${names}`,
    '',
    '### Workflow',
    WORKFLOW,
    '',
    '### Conventions',
    CONVENTIONS,
    '',
    'Ask before converting, annotating, running a write pipeline or pulling into a path outside the session workspace. Tools never modify the source dataset unless the destination is explicit.',
  ].join('\n');
}
