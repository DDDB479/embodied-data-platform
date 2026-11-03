"""Persistent DAG pipeline runner for embodied-data operations."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .annotations import apply_annotation_feedback
from .catalog import history as catalog_history
from .catalog import index as catalog_index
from .catalog import search as catalog_search
from .catalog import stats as catalog_stats
from .convert import convert
from .gate import evaluate_quality_gate
from .media import probe_dataset_media
from .model import to_jsonable
from .processors import ProcessorError, list_processors, run_processor
from .quality import build_labeling_tasks, detect_anomalies
from .registry import detect_dataset, open_dataset
from .reports import quality_report
from .storage import pull as storage_pull
from .validate import validate

SCHEMA_VERSION = 1
TERMINAL = frozenset({"succeeded", "failed", "cancelled", "succeeded_with_errors"})


class PipelineError(RuntimeError):
    pass


class PipelineValidationError(PipelineError):
    pass


class PipelinePolicyError(PipelineError):
    pass


class PipelineNotFound(KeyError):
    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        super().__init__(f"unknown pipeline run {run_id!r}")


SCHEMA = """
CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id TEXT PRIMARY KEY,
    pipeline_name TEXT NOT NULL,
    spec_hash TEXT NOT NULL,
    spec_json TEXT NOT NULL,
    variables_json TEXT NOT NULL,
    status TEXT NOT NULL,
    allow_write INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    started_at REAL,
    finished_at REAL,
    current_node TEXT,
    error TEXT
);
CREATE TABLE IF NOT EXISTS node_runs (
    run_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    node_type TEXT NOT NULL,
    status TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    started_at REAL,
    finished_at REAL,
    output_json TEXT,
    error_json TEXT,
    updated_at REAL NOT NULL,
    PRIMARY KEY(run_id, node_id),
    FOREIGN KEY(run_id) REFERENCES pipeline_runs(run_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_pipeline_runs_created ON pipeline_runs(created_at);
CREATE INDEX IF NOT EXISTS idx_node_runs_status ON node_runs(status);
"""


def _load_document(source: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(source, Mapping):
        return dict(source)
    path = Path(source).expanduser()
    if not path.is_file():
        raise FileNotFoundError(str(path))
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        try:
            import tomllib
        except ImportError as exc:  # pragma: no cover
            raise PipelineValidationError(f"{path}: expected JSON pipeline") from exc
        payload = tomllib.loads(text)
    if not isinstance(payload, dict):
        raise PipelineValidationError("pipeline document must be an object")
    return payload


def validate_pipeline(source: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    spec = _load_document(source)
    nodes = spec.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        raise PipelineValidationError("pipeline.nodes must be a non-empty list")
    seen: set[str] = set()
    for position, node in enumerate(nodes):
        if not isinstance(node, dict):
            raise PipelineValidationError(f"node #{position} must be an object")
        node_id = node.get("id")
        if not isinstance(node_id, str) or not node_id:
            raise PipelineValidationError(f"node #{position} requires a non-empty id")
        if node_id in seen:
            raise PipelineValidationError(f"duplicate node id {node_id!r}")
        seen.add(node_id)
        node_type = node.get("type", "tool")
        if node_type not in {"tool", "processor", "human", "noop"}:
            raise PipelineValidationError(f"node {node_id!r} has unsupported type {node_type!r}")
        depends = node.get("depends_on") or []
        if not isinstance(depends, list) or not all(isinstance(item, str) for item in depends):
            raise PipelineValidationError(f"node {node_id!r} depends_on must be a string list")
    for node in nodes:
        for dependency in node.get("depends_on") or []:
            if dependency not in seen:
                raise PipelineValidationError(f"node {node['id']!r} depends on unknown node {dependency!r}")

    def visit(node_id: str, path: list[str]) -> None:
        if node_id in path:
            raise PipelineValidationError("pipeline cycle: " + " -> ".join(path + [node_id]))
        node = next(item for item in nodes if item["id"] == node_id)
        for dependency in node.get("depends_on") or []:
            visit(dependency, path + [node_id])

    for node in nodes:
        visit(node["id"], [])
    spec["schema_version"] = int(spec.get("schema_version", 1))
    return spec


def _topological_order(spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    nodes = {node["id"]: node for node in spec["nodes"]}
    ordered: list[dict[str, Any]] = []
    pending = set(nodes)
    while pending:
        ready = sorted(
            node_id
            for node_id in pending
            if all(dependency not in pending for dependency in nodes[node_id].get("depends_on") or [])
        )
        if not ready:
            raise PipelineValidationError("pipeline cycle detected")
        for node_id in ready:
            ordered.append(nodes[node_id])
            pending.remove(node_id)
    return ordered


def _lookup(path: str, variables: Mapping[str, Any], outputs: Mapping[str, Any]) -> Any:
    if path.startswith("nodes."):
        current: Any = outputs
        parts = path.split(".")[1:]
    else:
        current = variables
        parts = path.split(".")
    for part in parts:
        if isinstance(current, Mapping):
            if part not in current:
                raise PipelineValidationError(f"template path {path!r} is not available")
            current = current[part]
        elif isinstance(current, Sequence) and not isinstance(current, (str, bytes)):
            current = current[int(part)]
        else:
            raise PipelineValidationError(f"template path {path!r} cannot traverse {type(current).__name__}")
    return current


def _render(value: Any, variables: Mapping[str, Any], outputs: Mapping[str, Any]) -> Any:
    if isinstance(value, str):
        if value.startswith("${") and value.endswith("}") and value.count("${") == 1:
            return _lookup(value[2:-1], variables, outputs)
        out = value
        start = 0
        while True:
            left = out.find("${", start)
            if left < 0:
                break
            right = out.find("}", left + 2)
            if right < 0:
                break
            resolved = _lookup(out[left + 2 : right], variables, outputs)
            out = out[:left] + str(resolved) + out[right + 1 :]
            start = left + len(str(resolved))
        return out
    if isinstance(value, dict):
        return {key: _render(item, variables, outputs) for key, item in value.items()}
    if isinstance(value, list):
        return [_render(item, variables, outputs) for item in value]
    return value


def _summary(path: str | Path) -> dict[str, Any]:
    dataset = open_dataset(path)
    try:
        body = dataset.summary.to_dict(include_episodes=False)
        body["capabilities"] = sorted(dataset.capabilities)
        return body
    finally:
        dataset.close()


def _decode_samples(arguments: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    dataset = open_dataset(arguments["path"])
    try:
        if hasattr(dataset, "topic_samples"):
            return dataset.topic_samples(
                arguments["topic"],
                episode=arguments.get("episode"),
                limit=int(arguments.get("limit", 5)),
                include_arrays=bool(arguments.get("include_arrays", False)),
            )
        if hasattr(dataset, "episode_samples"):
            return dataset.episode_samples(episode=arguments.get("episode"), limit=int(arguments.get("limit", 1)))
        raise PipelineError(f"{dataset.summary.format} does not expose decoded samples")
    finally:
        dataset.close()


def _labeling_manifest(arguments: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    path = arguments.get("path") or context.get("input")
    if not path:
        raise ValueError("labeling task node requires path or context.input")
    dataset = open_dataset(path)
    try:
        ids, columns = dataset.frames(arguments.get("episodes") or None)
        windows = [] if arguments.get("skip_anomalies") else detect_anomalies(
            ids,
            columns,
            fps=dataset.summary.fps,
            z=float(arguments.get("z", 8.0)),
        )
        manifest = build_labeling_tasks(
            dataset.summary,
            windows,
            max_tasks=int(arguments.get("max_tasks", 100)),
            include_segments=bool(arguments.get("include_segments", False)),
        )
    finally:
        dataset.close()
    if arguments.get("out"):
        out = Path(arguments["out"]).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        manifest["written_to"] = str(out.resolve())
    return manifest


def _read_document(path: str | Path) -> Any:
    target = Path(path).expanduser()
    text = target.read_text(encoding="utf-8")
    if target.suffix.lower() in (".jsonl", ".ndjson"):
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def _write_document(path: str | Path, payload: Any) -> dict[str, Any]:
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(to_jsonable(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temp, target)
    return {"written_to": str(target.resolve())}


def _apply_annotations(arguments: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    manifest = arguments.get("manifest")
    if isinstance(manifest, str):
        manifest = _read_document(manifest)
    labels = arguments.get("labels") or []
    if isinstance(labels, str):
        labels = _read_document(labels)
    return apply_annotation_feedback(manifest, labels, out=arguments.get("out"))


ToolHandler = Callable[[dict[str, Any], dict[str, Any]], Mapping[str, Any]]
_TOOL_HANDLERS: dict[str, tuple[ToolHandler, bool]] = {}


def _register(name: str, fn: ToolHandler, *, write: bool = False) -> None:
    _TOOL_HANDLERS[name] = (fn, write)


_register("detect_dataset", lambda a, c: detect_dataset(a["path"]).to_dict())
_register("summarize_dataset", lambda a, c: _summary(a["path"]))
_register("validate_dataset", lambda a, c: validate(a["path"], checks=a.get("checks"), include_frames=bool(a.get("include_frames", False))))
_register("quality_report", lambda a, c: quality_report(
    a["path"],
    episodes=a.get("episodes"),
    features=a.get("features"),
    z=float(a.get("z", 8.0)),
    include_segments=bool(a.get("include_segments", False)),
    skip_anomalies=bool(a.get("skip_anomalies", False)),
    use_cache=bool(a.get("use_cache", True)),
    cache_path=a.get("cache_path"),
    refresh_cache=bool(a.get("refresh_cache", False)),
    chunk_frames=a.get("chunk_frames"),
    max_frames=a.get("max_frames"),
))
_register("quality_gate", lambda a, c: evaluate_quality_gate(a["path"], rules=a.get("rules")))
_register("inspect_media", lambda a, c: _media(a))
_register("decode_samples", _decode_samples)
_register("catalog_search", lambda a, c: {"results": catalog_search(a["db"], text=a.get("text"), fmt=a.get("format"), robot_type=a.get("robot_type"), min_episodes=a.get("min_episodes"), min_duration=a.get("min_duration"), min_score=a.get("min_score"), limit=int(a.get("limit", 50)), offset=int(a.get("offset", 0)))})
_register("catalog_history", lambda a, c: catalog_history(a["db"], path=a.get("path"), limit=int(a.get("limit", 20))))
_register("catalog_stats", lambda a, c: catalog_stats(a["db"]))
_register("catalog_index", lambda a, c: catalog_index(a["root"], a["db"], max_depth=int(a.get("max_depth", 4)), force=bool(a.get("force", False)), validate=bool(a.get("validate", False))), write=True)
_register("convert_dataset", lambda a, c: convert(a["source"], a["destination"], target=a.get("target", "jsonl"), overwrite=bool(a.get("overwrite", False)), episodes=a.get("episodes"), train_ratio=float(a.get("train_ratio", 0.9)), validation_ratio=float(a.get("validation_ratio", 0.0)), test_ratio=float(a.get("test_ratio", 0.0))), write=True)
_register("storage_pull", lambda a, c: storage_pull(a["uri"], a["destination"], overwrite=bool(a.get("overwrite", False)), retries=int(a.get("retries", 2))), write=True)
_register("labeling_tasks", _labeling_manifest, write=True)
_register("apply_annotations", _apply_annotations, write=True)
_register("write_json", lambda a, c: _write_document(a["path"], a.get("payload", {})), write=True)


def _media(arguments: dict[str, Any]) -> dict[str, Any]:
    dataset = open_dataset(arguments["path"])
    try:
        return probe_dataset_media(dataset, max_videos=int(arguments.get("max_videos", 4)), probe=bool(arguments.get("probe", True)))
    finally:
        dataset.close()


def list_tool_handlers() -> list[dict[str, Any]]:
    return [{"tool": name, "write": write} for name, (_, write) in sorted(_TOOL_HANDLERS.items())]


class PipelineStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.executescript(SCHEMA)
        return conn

    def create_run(self, spec: dict[str, Any], variables: Mapping[str, Any], *, allow_write: bool) -> str:
        run_id = "run_" + uuid.uuid4().hex[:16]
        encoded = json.dumps(spec, sort_keys=True, ensure_ascii=False)
        spec_hash = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        now = time.time()
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO pipeline_runs(run_id, pipeline_name, spec_hash, spec_json, variables_json, status, allow_write, created_at) "
                "VALUES (?, ?, ?, ?, ?, 'pending', ?, ?)",
                (run_id, str(spec.get("name") or "pipeline"), spec_hash, encoded, json.dumps(to_jsonable(dict(variables)), ensure_ascii=False), int(allow_write), now),
            )
            conn.commit()
        finally:
            conn.close()
        return run_id

    def get_run(self, run_id: str) -> dict[str, Any]:
        conn = self._connect()
        try:
            row = conn.execute("SELECT * FROM pipeline_runs WHERE run_id=?", (run_id,)).fetchone()
            if row is None:
                raise PipelineNotFound(run_id)
            nodes = [dict(item) for item in conn.execute("SELECT * FROM node_runs WHERE run_id=? ORDER BY updated_at", (run_id,))]
        finally:
            conn.close()
        body = dict(row)
        body["spec"] = json.loads(body.pop("spec_json"))
        body["variables"] = json.loads(body.pop("variables_json"))
        body["allow_write"] = bool(body["allow_write"])
        for node in nodes:
            node["output"] = json.loads(node.pop("output_json")) if node.get("output_json") else None
            node["error"] = json.loads(node.pop("error_json")) if node.get("error_json") else None
        body["nodes"] = nodes
        return body

    def list_runs(self, *, limit: int = 20, status: str | None = None) -> list[dict[str, Any]]:
        conn = self._connect()
        try:
            if status:
                rows = conn.execute("SELECT * FROM pipeline_runs WHERE status=? ORDER BY created_at DESC LIMIT ?", (status, max(1, int(limit)))).fetchall()
            else:
                rows = conn.execute("SELECT * FROM pipeline_runs ORDER BY created_at DESC LIMIT ?", (max(1, int(limit)),)).fetchall()
        finally:
            conn.close()
        return [self.get_run(str(row["run_id"])) for row in rows]

    def node_outputs(self, run_id: str) -> dict[str, Any]:
        run = self.get_run(run_id)
        return {node["node_id"]: node["output"] for node in run["nodes"] if node.get("status") == "succeeded" and node.get("output") is not None}

    def update_run(self, run_id: str, **values: Any) -> None:
        if not values:
            return
        columns = ", ".join(f"{key}=?" for key in values)
        conn = self._connect()
        try:
            conn.execute(f"UPDATE pipeline_runs SET {columns} WHERE run_id=?", [*values.values(), run_id])
            conn.commit()
        finally:
            conn.close()

    def upsert_node(self, run_id: str, node: Mapping[str, Any], **values: Any) -> None:
        now = time.time()
        merged = {"run_id": run_id, "node_id": node["id"], "node_type": node.get("type", "tool"), "updated_at": now, **values}
        conn = self._connect()
        try:
            existing = conn.execute("SELECT 1 FROM node_runs WHERE run_id=? AND node_id=?", (run_id, node["id"])).fetchone()
            if existing:
                assignments = ", ".join(f"{key}=?" for key in merged if key not in {"run_id", "node_id"})
                conn.execute(f"UPDATE node_runs SET {assignments} WHERE run_id=? AND node_id=?", [*[merged[key] for key in merged if key not in {"run_id", "node_id"}], run_id, node["id"]])
            else:
                columns = ", ".join(merged)
                conn.execute(f"INSERT INTO node_runs({columns}) VALUES ({', '.join('?' for _ in merged)})", list(merged.values()))
            conn.commit()
        finally:
            conn.close()

    def stats(self) -> dict[str, Any]:
        conn = self._connect()
        try:
            totals = {row["status"]: int(row["n"]) for row in conn.execute("SELECT status, COUNT(*) AS n FROM pipeline_runs GROUP BY status")}
            node_failures = {row["node_id"]: int(row["n"]) for row in conn.execute("SELECT node_id, COUNT(*) AS n FROM node_runs WHERE status='failed' GROUP BY node_id ORDER BY n DESC")}
            durations = [float(row[0]) for row in conn.execute("SELECT finished_at - started_at FROM pipeline_runs WHERE started_at IS NOT NULL AND finished_at IS NOT NULL")]
            total = conn.execute("SELECT COUNT(*) FROM pipeline_runs").fetchone()[0]
        finally:
            conn.close()
        return {
            "total_runs": int(total),
            "by_status": totals,
            "average_duration_s": round(sum(durations) / len(durations), 3) if durations else None,
            "node_failures": node_failures,
        }


def _execute_node(node: Mapping[str, Any], arguments: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    node_type = node.get("type", "tool")
    if node_type == "noop":
        return {"arguments": arguments}
    if node_type == "processor":
        return run_processor(str(node.get("processor") or node["id"]), arguments, context)
    if node_type == "tool":
        name = str(node.get("tool") or "")
        handler = _TOOL_HANDLERS.get(name)
        if handler is None:
            raise PipelineError(f"unknown pipeline tool {name!r}; available: {sorted(_TOOL_HANDLERS)}")
        fn, write = handler
        if write and not context.get("allow_write"):
            raise PipelinePolicyError(f"node {node['id']!r} uses write tool {name!r}; rerun with allow_write")
        return to_jsonable(dict(fn(arguments, context)))
    raise PipelineError(f"unsupported node type {node_type!r}")


def run_pipeline(
    spec: Mapping[str, Any] | str | Path,
    *,
    store: str | Path = "pipeline-runs.sqlite3",
    variables: Mapping[str, Any] | None = None,
    allow_write: bool = False,
) -> dict[str, Any]:
    resolved = validate_pipeline(spec)
    manager = PipelineStore(store)
    run_id = manager.create_run(resolved, variables or {}, allow_write=allow_write)
    return resume_pipeline(run_id, store=store)


def resume_pipeline(
    run_id: str,
    *,
    store: str | Path = "pipeline-runs.sqlite3",
    human_results: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    manager = PipelineStore(store)
    run = manager.get_run(run_id)
    spec = run["spec"]
    variables = run["variables"]
    outputs = manager.node_outputs(run_id)
    manager.update_run(run_id, status="running", started_at=run.get("started_at") or time.time(), finished_at=None, error=None)
    failed = False
    for node in _topological_order(spec):
        node_id = node["id"]
        existing = next((item for item in run["nodes"] if item["node_id"] == node_id), None)
        if existing and existing.get("status") == "succeeded":
            outputs[node_id] = existing.get("output") or {}
            continue
        if human_results and node_id in human_results:
            result = to_jsonable(human_results[node_id])
            manager.upsert_node(run_id, node, status="succeeded", attempts=1, started_at=time.time(), finished_at=time.time(), output_json=json.dumps(result, ensure_ascii=False), error_json=None)
            outputs[node_id] = result
            continue
        if node.get("type") == "human" and existing and existing.get("status") == "waiting_human":
            manager.update_run(run_id, status="waiting_human", current_node=node_id)
            return manager.get_run(run_id)
        try:
            arguments = _render(node.get("arguments") or {}, variables, outputs)
        except PipelineValidationError as exc:
            manager.upsert_node(run_id, node, status="failed", attempts=0, finished_at=time.time(), error_json=json.dumps({"type": type(exc).__name__, "message": str(exc)}))
            manager.update_run(run_id, status="failed", finished_at=time.time(), current_node=node_id, error=str(exc))
            return manager.get_run(run_id)
        context = {
            "variables": variables,
            "nodes": outputs,
            "run_id": run_id,
            "node_id": node_id,
            "allow_write": run["allow_write"],
            "input": variables.get("input"),
        }
        attempts = max(1, int(node.get("retries", 0)) + 1)
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            manager.upsert_node(run_id, node, status="running", attempts=attempt, started_at=time.time(), finished_at=None, error_json=None)
            try:
                if node.get("type") == "human":
                    result = _labeling_manifest(arguments, context)
                    manager.upsert_node(run_id, node, status="waiting_human", attempts=attempt, finished_at=time.time(), output_json=json.dumps(result, ensure_ascii=False))
                    manager.update_run(run_id, status="waiting_human", current_node=node_id)
                    return manager.get_run(run_id)
                result = _execute_node(node, arguments, context)
                manager.upsert_node(run_id, node, status="succeeded", attempts=attempt, finished_at=time.time(), output_json=json.dumps(result, ensure_ascii=False))
                outputs[node_id] = result
                manager.update_run(run_id, current_node=node_id)
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                manager.upsert_node(run_id, node, status="failed", attempts=attempt, finished_at=time.time(), error_json=json.dumps({"type": type(exc).__name__, "message": str(exc)}))
                if attempt < attempts:
                    time.sleep(min(0.2 * (2 ** (attempt - 1)), 1.0))
        if last_error is not None:
            if node.get("continue_on_error"):
                failed = True
                continue
            manager.update_run(run_id, status="failed", finished_at=time.time(), current_node=node_id, error=f"{type(last_error).__name__}: {last_error}")
            return manager.get_run(run_id)
    manager.update_run(run_id, status="succeeded_with_errors" if failed else "succeeded", finished_at=time.time(), current_node=None)
    return manager.get_run(run_id)
