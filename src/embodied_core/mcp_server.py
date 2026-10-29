"""MCP server (stdio, JSON-RPC 2.0) exposing the toolkit to any agent host.

The server speaks the Model Context Protocol directly over stdin/stdout, so it
runs without the ``mcp`` SDK::

    embodied-mcp                       # or: python -m embodied_core.mcp_server

Register it in a client with a command like::

    {"command": "embodied-mcp"}

or for Codex / Claude Code style configs::

    {"mcpServers": {"embodied": {"command": "embodied-mcp", "args": []}}}

The same tool surface is wrapped by the DSH plugin in ``packages/dsh-plugin``,
which lets one tooling contract serve DSH, Codex, Claude Code and any other
MCP-capable client.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import traceback
from dataclasses import dataclass
from typing import Any, Callable

from . import __version__
from .catalog import index as catalog_index
from .catalog import history as catalog_history
from .catalog import search as catalog_search
from .catalog import stats as catalog_stats
from .convert import ConversionError, convert
from .formats.base import FrameLimitExceeded, FramesUnavailable
from .gate import evaluate_quality_gate
from .jobs import Job, JobCancelled, JobManager, JobNotFound
from .model import to_jsonable
from .quality import build_labeling_tasks, compute_dataset_stats, detect_anomalies, make_windows, segment_by_motion
from .registry import UnsupportedFormat, describe_formats, detect_dataset, open_dataset
from .reports import quality_report
from .scaffold import scaffold
from .validate import CHECKS, DEFAULT_CHECKS, validate

PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "embodied-data-studio"


def _json_text(payload: Any) -> dict[str, Any]:
    return {"type": "text", "text": json.dumps(to_jsonable(payload), indent=2, ensure_ascii=False)}


# Ordered: the first matching class wins, so keep subclasses above parents.
_ERROR_CODES: tuple[tuple[type[BaseException], str], ...] = (
    (FileNotFoundError, "not_found"),
    (JobNotFound, "job_not_found"),
    (JobCancelled, "job_cancelled"),
    (UnsupportedFormat, "unsupported_format"),
    (FileExistsError, "already_exists"),
    (ConversionError, "conversion_failed"),
    (FrameLimitExceeded, "frame_limit_exceeded"),
    (FramesUnavailable, "frames_unavailable"),
    (KeyError, "bad_arguments"),
    (ValueError, "bad_arguments"),
)


ASYNC_TOOLS = frozenset({"quality_report", "validate_dataset", "quality_gate"})


@dataclass
class ToolContext:
    """Per-request transport context for progress notifications."""

    progress_token: Any = None
    emit: Callable[[dict[str, Any]], None] | None = None
    total: float = 100.0

    def notify_progress(self, progress: float, message: str) -> None:
        if self.progress_token is None or self.emit is None:
            return
        self.emit(
            {
                "jsonrpc": "2.0",
                "method": "notifications/progress",
                "params": {
                    "progressToken": self.progress_token,
                    "progress": round(float(progress), 1),
                    "total": self.total,
                    "message": message,
                },
            }
        )


def _error_code(exc: Exception) -> str:
    for cls, code in _ERROR_CODES:
        if isinstance(exc, cls):
            return code
    return "internal_error"


def _error_text(exc: Exception) -> dict[str, Any]:
    """A machine-readable error block.

    Agents recover far better from {"code": "not_found"} than from a Python
    class name, so the message carries a stable code plus any structured
    attributes the exception already exposes (reason / install / hint).
    """
    if isinstance(exc, KeyError) and exc.args:
        message = str(exc.args[0])
    else:
        message = str(exc)
    error: dict[str, Any] = {"code": _error_code(exc), "type": type(exc).__name__, "message": message}
    for attr in ("reason", "install", "hint"):
        value = getattr(exc, attr, None)
        if value:
            error[attr] = value
    return {"type": "text", "text": json.dumps({"error": error}, indent=2, ensure_ascii=False)}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _job_error(exc: BaseException) -> dict[str, Any]:
    if isinstance(exc, Exception):
        return json.loads(_error_text(exc)["text"])["error"]
    return {"code": "internal_error", "type": type(exc).__name__, "message": str(exc)}


# Jobs are process-local and intentionally read-only. Writes stay on the
# synchronous safety path so a background job can never bypass workspace
# approvals.
JOBS = JobManager(
    max_workers=_env_int("EMBODIED_JOB_WORKERS", 2),
    ttl_s=float(_env_int("EMBODIED_JOB_TTL_S", 3600)),
    max_jobs=_env_int("EMBODIED_JOB_MAX", 200),
    error_formatter=_job_error,
)


# --------------------------------------------------------------------------
# tool handlers
# --------------------------------------------------------------------------
def tool_detect(arguments: dict[str, Any]) -> Any:
    return detect_dataset(arguments["path"]).to_dict()


def tool_summarize(arguments: dict[str, Any]) -> Any:
    dataset = open_dataset(arguments["path"])
    try:
        body = dataset.summary.to_dict(include_episodes=bool(arguments.get("include_episodes", False)))
        body["capabilities"] = sorted(dataset.capabilities)
        body["optional_dependencies"] = dataset.optional_dependencies()
        return body
    finally:
        dataset.close()


def _validate_kwargs(arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "checks": arguments.get("checks") or None,
        "include_frames": bool(arguments.get("include_frames", False)),
    }


def _quality_kwargs(arguments: dict[str, Any]) -> dict[str, Any]:
    # The MCP surface never accepts an arbitrary cache path: the default cache
    # lives in the user cache directory, and a caller-supplied path would turn
    # this read-only tool into an unapproved write outside the workspace.
    if arguments.get("cache_path"):
        raise ValueError(
            "cache_path is not accepted over MCP; set EMBODIED_QUALITY_CACHE for the server process instead"
        )
    return {
        "episodes": arguments.get("episodes") or None,
        "features": arguments.get("features") or None,
        "z": float(arguments.get("z", 8.0)),
        "include_segments": bool(arguments.get("include_segments", False)),
        "skip_anomalies": bool(arguments.get("skip_anomalies", False)),
        "use_cache": bool(arguments.get("use_cache", True)),
        "refresh_cache": bool(arguments.get("refresh_cache", False)),
        "chunk_frames": arguments.get("chunk_frames"),
        "max_frames": arguments.get("max_frames"),
    }


def _gate_kwargs(arguments: dict[str, Any]) -> dict[str, Any]:
    rules = arguments.get("rules")
    if rules is not None and not isinstance(rules, dict):
        raise ValueError("rules must be a JSON object")
    return {"rules": rules}


def tool_validate(arguments: dict[str, Any], context: ToolContext | None = None) -> Any:
    return validate(
        arguments["path"],
        **_validate_kwargs(arguments),
        progress=context.notify_progress if context is not None else None,
    )


def tool_quality(arguments: dict[str, Any], context: ToolContext | None = None) -> Any:
    return quality_report(
        arguments["path"],
        **_quality_kwargs(arguments),
        progress=context.notify_progress if context is not None else None,
    )


def tool_quality_gate(arguments: dict[str, Any], context: ToolContext | None = None) -> Any:
    return evaluate_quality_gate(
        arguments["path"],
        **_gate_kwargs(arguments),
        progress=context.notify_progress if context is not None else None,
    )


def _run_job(job: Job, report: Callable[[float, str], None]) -> Any:
    arguments = job.arguments
    if job.tool == "quality_report":
        return quality_report(arguments["path"], **_quality_kwargs(arguments), progress=report)
    if job.tool == "validate_dataset":
        return validate(arguments["path"], **_validate_kwargs(arguments), progress=report)
    if job.tool == "quality_gate":
        return evaluate_quality_gate(arguments["path"], **_gate_kwargs(arguments), progress=report)
    raise KeyError(f"tool {job.tool!r} is not available as a background job")


def tool_submit_job(arguments: dict[str, Any], context: ToolContext | None = None) -> Any:
    tool = str(arguments.get("tool") or "")
    if tool not in ASYNC_TOOLS:
        raise ValueError(f"tool must be one of {sorted(ASYNC_TOOLS)}; got {tool!r}")
    job_arguments = arguments.get("arguments")
    if not isinstance(job_arguments, dict):
        raise ValueError("arguments must be a JSON object")
    if not isinstance(job_arguments.get("path"), str) or not job_arguments["path"].strip():
        raise ValueError("arguments.path is required")
    # Validate the parameter surface now, so obviously bad jobs fail at submit
    # time instead of after a round trip through the worker.
    if tool == "quality_report":
        _quality_kwargs(job_arguments)
    elif tool == "validate_dataset":
        _validate_kwargs(job_arguments)
    else:
        _gate_kwargs(job_arguments)

    emitter = None
    if context is not None and context.progress_token is not None and context.emit is not None:
        def emitter(_job: Job, progress: float, message: str) -> None:
            context.notify_progress(progress, message)

    job = JOBS.submit(
        tool,
        job_arguments,
        _run_job,
        progress_token=context.progress_token if context is not None else None,
        emit_progress=emitter,
    )
    return {
        "job": job.to_dict(),
        "poll_with": "job_status",
        "hint": "Poll job_status until status is succeeded/failed/cancelled; request the result with include_result=true.",
    }


def tool_job_status(arguments: dict[str, Any]) -> Any:
    job_id = arguments.get("job_id")
    if not isinstance(job_id, str) or not job_id:
        raise ValueError("job_id is required")
    body = JOBS.get(job_id, include_result=bool(arguments.get("include_result", False)))
    if body.get("result_ready") and not arguments.get("include_result"):
        body["hint"] = "Result is ready; call job_status again with include_result=true."
    return body


def tool_cancel_job(arguments: dict[str, Any]) -> Any:
    job_id = arguments.get("job_id")
    if not isinstance(job_id, str) or not job_id:
        raise ValueError("job_id is required")
    body = JOBS.cancel(job_id)
    body["hint"] = "Cancellation is cooperative; running work stops at the next analysis checkpoint."
    return body


def tool_list_jobs(arguments: dict[str, Any]) -> Any:
    jobs = JOBS.list(limit=int(arguments.get("limit", 20)), status=arguments.get("status"))
    return {"count": len(jobs), "jobs": jobs}


def tool_segment(arguments: dict[str, Any]) -> Any:
    dataset = open_dataset(arguments["path"])
    try:
        episode = arguments.get("episode")
        ids, columns = dataset.frames([episode] if episode is not None else None)
        return {
            "episode": episode,
            "segments": segment_by_motion(
                ids,
                columns,
                feature=arguments.get("feature"),
                fps=dataset.summary.fps,
                min_segment=int(arguments.get("min_frames", 5)),
            ),
        }
    finally:
        dataset.close()


def tool_labeling_tasks(arguments: dict[str, Any]) -> Any:
    dataset = open_dataset(arguments["path"])
    try:
        windows: list[dict[str, Any]] = []
        segments = None
        try:
            ids, columns = dataset.frames()
        except Exception:
            ids, columns = [], {}
        if columns:
            windows = detect_anomalies(ids, columns, fps=dataset.summary.fps, z=float(arguments.get("z", 8.0)))
            if arguments.get("include_segments"):
                segments = segment_by_motion(ids, columns, fps=dataset.summary.fps)
        manifest = build_labeling_tasks(
            dataset.summary,
            windows,
            max_tasks=int(arguments.get("max_tasks", 100)),
            include_segments=bool(arguments.get("include_segments")),
            segments=segments,
        )
        if arguments.get("out"):
            from pathlib import Path

            Path(arguments["out"]).expanduser().write_text(
                json.dumps(to_jsonable(manifest), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            return {"written": str(arguments["out"]), "task_count": manifest["task_count"]}
        return manifest
    finally:
        dataset.close()


def tool_convert(arguments: dict[str, Any]) -> Any:
    return convert(
        arguments["source"],
        arguments["destination"],
        target=arguments.get("target", "jsonl"),
        overwrite=bool(arguments.get("overwrite", False)),
        episodes=arguments.get("episodes") or None,
    )


def tool_scaffold(arguments: dict[str, Any]) -> Any:
    return scaffold(
        arguments["destination"],
        kind=arguments.get("kind", "platform"),
        name=arguments.get("name", "embodied-platform"),
        overwrite=bool(arguments.get("overwrite", False)),
    )


def tool_catalog_index(arguments: dict[str, Any]) -> Any:
    return catalog_index(
        arguments["root"],
        arguments["db"],
        max_depth=int(arguments.get("max_depth", 4)),
        force=bool(arguments.get("force", False)),
        validate=bool(arguments.get("validate", False)),
    )


def tool_catalog_search(arguments: dict[str, Any]) -> Any:
    rows = catalog_search(
        arguments["db"],
        text=arguments.get("text"),
        fmt=arguments.get("format"),
        robot_type=arguments.get("robot_type"),
        min_episodes=arguments.get("min_episodes"),
        min_duration=arguments.get("min_duration"),
        min_score=arguments.get("min_score"),
        limit=int(arguments.get("limit", 20)),
    )
    return {"count": len(rows), "results": rows}


def tool_catalog_stats(arguments: dict[str, Any]) -> Any:
    return catalog_stats(arguments["db"])


def tool_catalog_history(arguments: dict[str, Any]) -> Any:
    return catalog_history(
        arguments["db"],
        path=arguments.get("path"),
        limit=int(arguments.get("limit", 20)),
    )


def tool_formats(arguments: dict[str, Any]) -> Any:
    return {"formats": describe_formats(), "checks": {"default": DEFAULT_CHECKS, "available": sorted(CHECKS)}}


def tool_windows(arguments: dict[str, Any]) -> Any:
    dataset = open_dataset(arguments["path"])
    try:
        lengths = [episode.length or 0 for episode in dataset.summary.episodes]
        window = int(arguments.get("window", 16))
        plan = make_windows(lengths, window, arguments.get("stride"))
        return {"window": window, "count": len(plan), "windows": plan[: int(arguments.get("limit", 100))]}
    finally:
        dataset.close()


# --------------------------------------------------------------------------
# tool registry
# --------------------------------------------------------------------------
TOOLS: list[dict[str, Any]] = [
    {
        "name": "detect_dataset",
        "description": "Identify the on-disk format of an embodied-AI dataset (LeRobot, HDF5/robomimic, MCAP, rosbag1/2, RLDS, canonical JSONL) with confidence and evidence.",
        "inputSchema": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "File or directory to inspect"}},
            "required": ["path"],
        },
        "handler": tool_detect,
    },
    {
        "name": "summarize_dataset",
        "description": "Summarise a dataset: format, version, fps, robot type, features (key/dtype/shape/kind), episode count, frames, duration, tasks and notes. Use before any analysis to learn the schema.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "include_episodes": {"type": "boolean", "default": False, "description": "Include per-episode rows (can be long)"},
            },
            "required": ["path"],
        },
        "handler": tool_summarize,
    },
    {
        "name": "validate_dataset",
        "description": "Run structural validation: required metadata, feature vocabulary (state/action), episode indices and lengths, fps consistency, format-specific layout rules. Returns issues with severity, stable code, hint and a 0-100 score. Emits MCP progress notifications when the request carries _meta.progressToken.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "checks": {"type": "array", "items": {"type": "string"}, "description": "Subset of checks; omit for defaults"},
                "include_frames": {"type": "boolean", "default": False, "description": "Also scan numeric frames for NaN/inf (slower)"},
            },
            "required": ["path"],
        },
        "handler": tool_validate,
        "context": True,
    },
    {
        "name": "quality_report",
        "description": "Compute per-feature statistics (range, mean, std, percentiles, NaN ratio) and detect anomalous windows (jumps, stalls, non-finite spans) with robust z-scoring. Results are cached by dataset fingerprint and parameters unless use_cache=false. Emits MCP progress notifications when the request carries _meta.progressToken.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "episodes": {"type": "array", "items": {"type": "integer"}},
                "features": {"type": "array", "items": {"type": "string"}},
                "z": {"type": "number", "default": 8.0},
                "include_segments": {"type": "boolean", "default": False},
                "skip_anomalies": {"type": "boolean", "default": False},
                "use_cache": {"type": "boolean", "default": True, "description": "Reuse a fingerprint-keyed report when the dataset and parameters are unchanged."},
                "refresh_cache": {"type": "boolean", "default": False, "description": "Ignore any cached report and replace it after recomputing."},
                "chunk_frames": {"type": "integer", "description": "Frames per streaming chunk; defaults to 50000 or EMBODIED_CHUNK_FRAMES."},
                "max_frames": {"type": "integer", "description": "Hard materialization cap; 0 means unlimited; defaults to EMBODIED_MAX_FRAMES."},
            },
            "required": ["path"],
        },
        "handler": tool_quality,
        "context": True,
    },
    {
        "name": "quality_gate",
        "description": "Evaluate configurable publish rules from an inline JSON object: expected fps/tolerance, required feature keys or kinds, minimum score/episodes/frames, maximum NaN ratio and anomaly-window counts. Returns gate failures plus validation evidence. Read-only and available as a background job.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "rules": {
                    "type": "object",
                    "description": "Rule overrides; omitted keys use training-oriented defaults (min_score=80, max_errors=0).",
                    "additionalProperties": True,
                },
            },
            "required": ["path"],
        },
        "handler": tool_quality_gate,
        "context": True,
    },
    {
        "name": "segment_episodes",
        "description": "Split episodes into moving/idle segments using feature-speed percentiles. Useful for phase labelling and for trimming dead time.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "episode": {"type": "integer"},
                "feature": {"type": "string"},
                "min_frames": {"type": "integer", "default": 5},
            },
            "required": ["path"],
        },
        "handler": tool_segment,
    },
    {
        "name": "labeling_tasks",
        "description": "Generate an annotation manifest: each task names an episode, a frame range, cues and a concrete question, ready for a human reviewer or a vision-language model.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "max_tasks": {"type": "integer", "default": 100},
                "z": {"type": "number", "default": 8.0},
                "include_segments": {"type": "boolean", "default": False},
                "out": {"type": "string", "description": "Optional path to write the manifest JSON"},
            },
            "required": ["path"],
        },
        "handler": tool_labeling_tasks,
    },
    {
        "name": "convert_dataset",
        "description": "Convert a dataset to another layout: 'jsonl' (canonical, dependency-free) or 'lerobot' (v2.1 parquet, needs pyarrow). Numeric features only; camera streams are reported, not re-encoded.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "source": {"type": "string"},
                "destination": {"type": "string"},
                "target": {"type": "string", "enum": ["jsonl", "lerobot"], "default": "jsonl"},
                "episodes": {"type": "array", "items": {"type": "integer"}},
                "overwrite": {"type": "boolean", "default": False},
            },
            "required": ["source", "destination"],
        },
        "handler": tool_convert,
    },
    {
        "name": "scaffold_project",
        "description": "Write a project skeleton: 'platform' (HTTP API + quality gate + catalog + data contract), 'adapter' (new format adapter with tests) or 'pipeline' (ingest -> validate -> index).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "destination": {"type": "string"},
                "kind": {"type": "string", "enum": ["platform", "adapter", "pipeline"], "default": "platform"},
                "name": {"type": "string", "default": "embodied-platform"},
                "overwrite": {"type": "boolean", "default": False},
            },
            "required": ["destination"],
        },
        "handler": tool_scaffold,
    },
    {
        "name": "catalog_index",
        "description": "Scan a directory tree, load every dataset found and upsert its metadata into a SQLite catalog. Incremental: unchanged datasets are skipped.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "root": {"type": "string"},
                "db": {"type": "string", "description": "Catalog SQLite path"},
                "max_depth": {"type": "integer", "default": 4},
                "force": {"type": "boolean", "default": False},
                "validate": {"type": "boolean", "default": False, "description": "Also store a health score"},
            },
            "required": ["root", "db"],
        },
        "handler": tool_catalog_index,
    },
    {
        "name": "catalog_search",
        "description": "Query the dataset catalog by free text, format, robot type, episode count, duration or minimum health score.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "db": {"type": "string"},
                "text": {"type": "string"},
                "format": {"type": "string"},
                "robot_type": {"type": "string"},
                "min_episodes": {"type": "integer"},
                "min_duration": {"type": "number"},
                "min_score": {"type": "integer"},
                "limit": {"type": "integer", "default": 20},
            },
            "required": ["db"],
        },
        "handler": tool_catalog_search,
    },
    {
        "name": "catalog_stats",
        "description": "Totals for schema-v2 catalogs: datasets, immutable versions, runs, lineage edges, formats, episodes, frames and duration.",
        "inputSchema": {
            "type": "object",
            "properties": {"db": {"type": "string"}},
            "required": ["db"],
        },
        "handler": tool_catalog_stats,
    },
    {
        "name": "catalog_history",
        "description": "Inspect immutable version history, parent/child lineage and recent run records in a schema-v2 catalog, optionally filtered to one dataset path.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "db": {"type": "string"},
                "path": {"type": "string", "description": "Optional dataset root path"},
                "limit": {"type": "integer", "default": 20},
            },
            "required": ["db"],
        },
        "handler": tool_catalog_history,
    },
    {
        "name": "list_formats",
        "description": "List supported dataset formats, how each is detected, which optional dependency it needs, and the available validation checks.",
        "inputSchema": {"type": "object", "properties": {}},
        "handler": tool_formats,
    },
    {
        "name": "plan_windows",
        "description": "Plan fixed-length training windows over the episodes of a dataset (window, stride, episode/start/end).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "window": {"type": "integer", "default": 16},
                "stride": {"type": "integer"},
                "limit": {"type": "integer", "default": 100},
            },
            "required": ["path"],
        },
        "handler": tool_windows,
    },
    {
        "name": "submit_job",
        "description": "Start a long-running read-only analysis in the background. Supported tools are quality_report, validate_dataset and quality_gate. Returns a job_id immediately; poll job_status for progress and the result.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "tool": {"type": "string", "enum": ["quality_report", "validate_dataset", "quality_gate"]},
                "arguments": {
                    "type": "object",
                    "description": "Arguments for the selected tool, using the same names as its direct call.",
                    "additionalProperties": True,
                },
            },
            "required": ["tool", "arguments"],
        },
        "handler": tool_submit_job,
        "context": True,
    },
    {
        "name": "job_status",
        "description": "Read a background job's status, progress, message and structured error. Set include_result=true once status is succeeded to fetch the full report.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "job_id": {"type": "string"},
                "include_result": {"type": "boolean", "default": False},
            },
            "required": ["job_id"],
        },
        "handler": tool_job_status,
    },
    {
        "name": "cancel_job",
        "description": "Request cooperative cancellation of a background job. Running work stops at the next analysis checkpoint; use job_status to confirm.",
        "inputSchema": {
            "type": "object",
            "properties": {"job_id": {"type": "string"}},
            "required": ["job_id"],
        },
        "handler": tool_cancel_job,
    },
    {
        "name": "list_jobs",
        "description": "List recent background jobs, newest first, optionally filtered by status (queued, running, cancelling, succeeded, failed, cancelled).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "default": 20},
                "status": {"type": "string"},
            },
        },
        "handler": tool_list_jobs,
    },
]

_BY_NAME: dict[str, dict[str, Any]] = {tool["name"]: tool for tool in TOOLS}

# Hosts such as Codex treat an MCP tool without safety annotations as a write
# action and require approval before calling it. Keep these declarations close
# to the registry so the server's advertised behavior stays auditable.
_WRITE_TOOLS = {"labeling_tasks", "convert_dataset", "scaffold_project", "catalog_index"}


def _annotations(tool: dict[str, Any]) -> dict[str, bool]:
    name = tool["name"]
    read_only = name not in _WRITE_TOOLS
    return {
        "readOnlyHint": read_only,
        "destructiveHint": name in {"labeling_tasks", "convert_dataset", "scaffold_project"},
        "openWorldHint": False,
    }


def list_tools() -> list[dict[str, Any]]:
    return [
        {
            "name": tool["name"],
            "description": tool["description"],
            "inputSchema": tool["inputSchema"],
            "annotations": _annotations(tool),
        }
        for tool in TOOLS
    ]


def call_tool(
    name: str,
    arguments: dict[str, Any] | None = None,
    context: ToolContext | None = None,
) -> dict[str, Any]:
    tool = _BY_NAME.get(name)
    if tool is None:
        return {"content": [_error_text(KeyError(f"unknown tool {name!r}"))], "isError": True}
    try:
        if tool.get("context"):
            result = tool["handler"](arguments or {}, context or ToolContext())
        else:
            result = tool["handler"](arguments or {})
        return {"content": [_json_text(result)], "isError": False}
    except (FileNotFoundError, UnsupportedFormat, ConversionError, FileExistsError, ValueError, KeyError) as exc:
        return {"content": [_error_text(exc)], "isError": True}
    except Exception as exc:  # never leak a traceback to the client
        return {
            "content": [_error_text(exc), {"type": "text", "text": traceback.format_exc(limit=4)}],
            "isError": True,
        }


# --------------------------------------------------------------------------
# JSON-RPC transport
# --------------------------------------------------------------------------
def handle_message(
    message: dict[str, Any] | list[dict[str, Any]],
    *,
    emit: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any] | list[dict[str, Any]] | None:
    """Handle one message, or a JSON-RPC batch (a list of messages)."""
    if isinstance(message, list):
        responses = [handle_message(item, emit=emit) for item in message if isinstance(item, dict)]
        kept = [item for item in responses if item is not None]
        return kept or None

    method = message.get("method")
    request_id = message.get("id")
    params = message.get("params") or {}
    is_notification = "id" not in message
    meta = params.get("_meta") if isinstance(params.get("_meta"), dict) else {}
    context = ToolContext(progress_token=meta.get("progressToken"), emit=emit)

    if method == "initialize":
        result: dict[str, Any] = {
            "protocolVersion": params.get("protocolVersion") or PROTOCOL_VERSION,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": SERVER_NAME, "version": __version__},
            "instructions": (
                "Embodied-AI data platform tools. Start with detect_dataset or summarize_dataset, "
                "then validate_dataset and quality_report; quality_gate applies configurable publish rules. "
                "Use labeling_tasks to produce review work and scaffold_project for a new platform or adapter. "
                "For long scans, submit_job starts a background quality_report, validate_dataset or quality_gate "
                "run; poll job_status and fetch the result with include_result=true. quality_report is "
                "fingerprint-cached and reads frames in bounded chunks. The catalog stores immutable versions, "
                "runs and lineage; inspect them with catalog_history."
            ),
        }
    elif method in ("notifications/initialized", "initialized"):
        return None
    elif method == "ping":
        result = {}
    elif method == "tools/list":
        result = {"tools": list_tools()}
    elif method == "tools/call":
        result = call_tool(str(params.get("name")), params.get("arguments") or {}, context)
    elif method in ("resources/list", "prompts/list"):
        result = {"resources": []} if method.startswith("resources") else {"prompts": []}
    else:
        if is_notification:
            return None
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32601, "message": f"method not found: {method}"},
        }

    if is_notification:
        return None
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def serve(stdin=None, stdout=None) -> int:
    """Blocking stdio loop. Returns 0 on EOF."""
    src = stdin or sys.stdin
    dst = stdout or sys.stdout
    write_lock = threading.Lock()
    state = {"open": True}

    def emit(payload: dict[str, Any]) -> None:
        if not state["open"]:
            return
        encoded = json.dumps(payload, ensure_ascii=False)
        with write_lock:
            dst.write(encoded + "\n")
            dst.flush()

    try:
        for line in src:
            text = line.strip()
            if not text:
                continue
            try:
                message = json.loads(text)
            except json.JSONDecodeError as exc:
                response = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": f"parse error: {exc}"},
                }
            else:
                if isinstance(message, list):
                    batch = [item for item in message if isinstance(item, dict)]
                    response = handle_message(batch, emit=emit) if batch else {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {"code": -32600, "message": "invalid request"},
                    }
                elif isinstance(message, dict):
                    response = handle_message(message, emit=emit)
                else:
                    response = {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {"code": -32600, "message": "invalid request"},
                    }
            if response:
                emit(response)
    finally:
        # Background workers are daemon threads; cancel them before the process
        # exits so no notification can leak into a closed transport.
        state["open"] = False
        JOBS.shutdown()
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="embodied-mcp", description="MCP server for embodied-AI datasets")
    parser.add_argument("--version", action="version", version=f"{SERVER_NAME} {__version__}")
    parser.add_argument("--list-tools", action="store_true", help="print the tool catalogue and exit")
    args = parser.parse_args(argv)
    if args.list_tools:
        print(json.dumps(list_tools(), indent=2, ensure_ascii=False))
        return 0
    return serve()


if __name__ == "__main__":
    raise SystemExit(main())
