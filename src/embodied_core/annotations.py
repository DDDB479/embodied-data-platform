"""Annotation feedback loop for labeling-task manifests."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = 1
ANSWER_STATES = frozenset({"valid", "invalid", "needs_fix", "unknown"})


def _load_document(source: Any, *, name: str) -> Any:
    if isinstance(source, (dict, list)):
        return source
    path = Path(source).expanduser()
    if not path.is_file():
        raise FileNotFoundError(str(path))
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".jsonl", ".ndjson"):
        rows = []
        for line_no, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
        return rows
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        if not rows:
            raise ValueError(f"{path}: empty {name} document")
        return rows


def _manifest(source: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    payload = _load_document(source, name="manifest")
    if not isinstance(payload, dict) or not isinstance(payload.get("tasks"), list):
        raise ValueError("manifest must be an object with a tasks[] list")
    return dict(payload)


def _labels(source: Sequence[Mapping[str, Any]] | Mapping[str, Any] | str | Path) -> list[dict[str, Any]]:
    payload = _load_document(source, name="label")
    if isinstance(payload, dict):
        payload = payload.get("labels") or payload.get("annotations") or [payload]
    if not isinstance(payload, list):
        raise ValueError("labels must be a list or JSONL document")
    rows = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("each label must be an object")
        rows.append(dict(item))
    return rows


def normalize_answer(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if text in ANSWER_STATES:
        return text
    if text in {"pass", "passed", "ok", "yes", "true", "accept", "accepted"}:
        return "valid"
    if text in {"fail", "failed", "no", "false", "reject", "rejected"}:
        return "invalid"
    if text in {"fix", "repair", "needs_fix"}:
        return "needs_fix"
    return "unknown"


def review_annotations(
    manifest: Mapping[str, Any] | str | Path,
    labels: Sequence[Mapping[str, Any]] | Mapping[str, Any] | str | Path,
    *,
    require_complete: bool = False,
) -> dict[str, Any]:
    """Reconcile labels with a manifest. Duplicate task labels are last-wins."""

    body = _manifest(manifest)
    tasks = body["tasks"]
    by_id: dict[str, dict[str, Any]] = {}
    for position, task in enumerate(tasks):
        task_id = str(task.get("task_id") or f"task_{position}")
        task.setdefault("task_id", task_id)
        task.setdefault("status", "pending")
        by_id[task_id] = task

    latest: dict[str, dict[str, Any]] = {}
    conflicts: list[dict[str, Any]] = []
    unknown_labels: list[dict[str, Any]] = []
    for raw in _labels(labels):
        task_id = raw.get("task_id")
        if task_id is None and raw.get("id") is not None:
            try:
                task_id = str(tasks[int(raw["id"])]["task_id"])
            except (ValueError, TypeError, IndexError):
                task_id = None
        if not isinstance(task_id, str) or task_id not in by_id:
            unknown_labels.append(raw)
            continue
        answer = normalize_answer(raw.get("answer") or raw.get("label") or raw.get("status"))
        item = {
            "task_id": task_id,
            "answer": answer,
            "reason": str(raw.get("reason") or ""),
            "reviewer": str(raw.get("reviewer") or ""),
        }
        previous = latest.get(task_id)
        if previous is not None and previous["answer"] != answer:
            conflicts.append({"task_id": task_id, "previous": previous["answer"], "current": answer})
        latest[task_id] = item

    counts = {state: 0 for state in sorted(ANSWER_STATES)}
    flagged_episodes: dict[int, list[str]] = {}
    reviewed_tasks: list[dict[str, Any]] = []
    for task in tasks:
        task_id = str(task["task_id"])
        item = latest.get(task_id)
        copy = dict(task)
        if item is None:
            copy["status"] = "pending"
        else:
            copy.update(item)
            copy["status"] = item["answer"]
            counts[item["answer"]] = counts.get(item["answer"], 0) + 1
            if item["answer"] in {"invalid", "needs_fix", "unknown"}:
                episode = int(copy.get("episode", 0))
                flagged_episodes.setdefault(episode, []).append(task_id)
        reviewed_tasks.append(copy)

    reviewed = len(latest)
    total = len(tasks)
    complete = reviewed == total and not unknown_labels
    return {
        "schema_version": SCHEMA_VERSION,
        "manifest_id": body.get("manifest_id"),
        "ok": not unknown_labels and (complete or not require_complete),
        "complete": complete,
        "total_tasks": total,
        "reviewed_tasks": reviewed,
        "pending_tasks": max(0, total - reviewed),
        "coverage": round(reviewed / max(total, 1), 4),
        "counts": counts,
        "conflicts": conflicts,
        "unknown_labels": unknown_labels,
        "flagged_episodes": flagged_episodes,
        "tasks": reviewed_tasks,
        "dataset": body.get("dataset"),
    }


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="\n",
        delete=False,
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(handle.name, path)
    finally:
        try:
            if os.path.exists(handle.name):
                os.unlink(handle.name)
        except OSError:
            pass


def apply_annotation_feedback(
    manifest: Mapping[str, Any] | str | Path,
    labels: Sequence[Mapping[str, Any]] | Mapping[str, Any] | str | Path,
    *,
    out: str | Path | None = None,
    previous: str | Path | None = None,
) -> dict[str, Any]:
    """Reconcile and optionally persist a reviewed manifest atomically."""

    merged_labels = _labels(labels)
    if previous is not None:
        merged_labels = _labels(previous) + merged_labels
    reviewed = review_annotations(manifest, merged_labels)
    if out is not None:
        target = Path(out).expanduser()
        if target.suffix.lower() in (".jsonl", ".ndjson"):
            lines = [
                {
                    "task_id": task["task_id"],
                    "episode": task.get("episode"),
                    "start_frame": task.get("start_frame"),
                    "end_frame": task.get("end_frame"),
                    "status": task.get("status"),
                    "answer": task.get("answer"),
                    "reason": task.get("reason", ""),
                    "reviewer": task.get("reviewer", ""),
                }
                for task in reviewed["tasks"]
                if task.get("status") != "pending"
            ]
            _write_atomic(target, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in lines))
        else:
            _write_atomic(target, json.dumps(reviewed, indent=2, ensure_ascii=False) + "\n")
        reviewed["written_to"] = str(target.resolve())
    return reviewed
