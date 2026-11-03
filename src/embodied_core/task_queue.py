"""Persistent human-review queue with prelabels and active sampling."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .annotations import normalize_answer

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS human_tasks (
    task_id TEXT PRIMARY KEY,
    manifest_id TEXT,
    dataset_path TEXT,
    kind TEXT,
    episode INTEGER,
    start_frame INTEGER,
    end_frame INTEGER,
    question TEXT,
    suggested_label TEXT,
    status TEXT NOT NULL,
    priority INTEGER NOT NULL DEFAULT 50,
    assignee TEXT,
    prelabel TEXT,
    confidence REAL,
    model_source TEXT,
    conflict INTEGER NOT NULL DEFAULT 0,
    review_answer TEXT,
    reason TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    due_at REAL,
    payload_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_human_tasks_status ON human_tasks(status);
CREATE INDEX IF NOT EXISTS idx_human_tasks_assignee ON human_tasks(assignee);
CREATE INDEX IF NOT EXISTS idx_human_tasks_priority ON human_tasks(priority DESC);
"""


class TaskQueueError(RuntimeError):
    pass


def _load_manifest(source: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(source, Mapping):
        payload = dict(source)
    else:
        path = Path(source).expanduser()
        if not path.is_file():
            raise FileNotFoundError(str(path))
        payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("tasks"), list):
        raise ValueError("manifest must be an object with tasks[]")
    return payload


def _prelabel_map(prelabels: Sequence[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]] | None) -> dict[str, dict[str, Any]]:
    if prelabels is None:
        return {}
    if isinstance(prelabels, Mapping):
        rows = []
        for task_id, value in prelabels.items():
            if isinstance(value, Mapping):
                rows.append({"task_id": task_id, **dict(value)})
            else:
                rows.append({"task_id": task_id, "label": value})
    else:
        rows = [dict(item) for item in prelabels]
    out: dict[str, dict[str, Any]] = {}
    for item in rows:
        task_id = str(item.get("task_id") or "")
        if not task_id:
            continue
        answer = normalize_answer(item.get("label") or item.get("answer"))
        confidence = min(1.0, max(0.0, float(item.get("confidence", 0.5))))
        previous = out.get(task_id)
        conflict = previous is not None and previous["label"] != answer
        out[task_id] = {
            "label": answer,
            "confidence": confidence,
            "source": str(item.get("source") or item.get("model_source") or "prelabel"),
            "conflict": conflict or bool(previous and previous.get("conflict")),
        }
    return out


def sample_tasks(
    tasks: Sequence[Mapping[str, Any]],
    *,
    strategy: str = "priority",
    limit: int | None = None,
    prelabels: Sequence[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Select review tasks without mutating the input manifest."""

    labels = _prelabel_map(prelabels)
    enriched: list[dict[str, Any]] = []
    for task in tasks:
        item = dict(task)
        prelabel = labels.get(str(item.get("task_id")))
        if prelabel:
            item["prelabel"] = prelabel["label"]
            item["confidence"] = prelabel["confidence"]
            item["model_source"] = prelabel["source"]
            item["conflict"] = prelabel["conflict"]
            item["priority"] = 100 if prelabel["conflict"] else int(round((1.0 - prelabel["confidence"]) * 100))
        else:
            item["priority"] = int(item.get("priority", 50))
        enriched.append(item)
    if strategy in ("first", "sequential"):
        ordered = enriched
    elif strategy == "uncertainty":
        ordered = sorted(enriched, key=lambda item: (float(item.get("confidence", 1.0)), -int(item.get("priority", 0)), int(item.get("episode", 0))))
    elif strategy == "priority":
        ordered = sorted(enriched, key=lambda item: (-int(item.get("priority", 0)), int(item.get("episode", 0)), int(item.get("start_frame", 0))))
    elif strategy == "stratified":
        groups: dict[int, list[dict[str, Any]]] = {}
        for item in enriched:
            groups.setdefault(int(item.get("episode", 0)), []).append(item)
        ordered = []
        while any(groups.values()):
            for episode in sorted(groups):
                if groups[episode]:
                    ordered.append(groups[episode].pop(0))
    else:
        raise ValueError(f"unknown sampling strategy {strategy!r}")
    return ordered[: max(0, int(limit))] if limit is not None else ordered


class TaskQueueStore:
    def __init__(self, path: str | Path = "human-tasks.sqlite3") -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.executescript(SCHEMA)
        return conn

    def ingest(
        self,
        manifest: Mapping[str, Any] | str | Path,
        *,
        prelabels: Sequence[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]] | None = None,
        strategy: str = "priority",
        limit: int | None = None,
        default_due_s: float | None = None,
    ) -> dict[str, Any]:
        body = _load_manifest(manifest)
        selected = sample_tasks(body["tasks"], strategy=strategy, limit=limit, prelabels=prelabels)
        labels = _prelabel_map(prelabels)
        now = time.time()
        inserted = 0
        existing = 0
        conn = self._connect()
        try:
            for task in selected:
                task_id = str(task.get("task_id") or "")
                if not task_id:
                    continue
                prelabel = labels.get(task_id)
                priority = int(task.get("priority", 50))
                status = "prefilled" if prelabel else "pending"
                try:
                    conn.execute(
                        "INSERT INTO human_tasks(task_id, manifest_id, dataset_path, kind, episode, start_frame, end_frame, "
                        "question, suggested_label, status, priority, prelabel, confidence, model_source, conflict, "
                        "created_at, updated_at, due_at, payload_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            task_id,
                            body.get("manifest_id"),
                            (body.get("dataset") or {}).get("path"),
                            task.get("kind"),
                            task.get("episode"),
                            task.get("start_frame"),
                            task.get("end_frame"),
                            task.get("question"),
                            task.get("suggested_label"),
                            status,
                            priority,
                            prelabel["label"] if prelabel else None,
                            prelabel["confidence"] if prelabel else None,
                            prelabel["source"] if prelabel else None,
                            int(bool(prelabel and prelabel["conflict"])),
                            now,
                            now,
                            now + float(default_due_s) if default_due_s else None,
                            json.dumps(dict(task), ensure_ascii=False),
                        ),
                    )
                    inserted += 1
                except sqlite3.IntegrityError:
                    existing += 1
            conn.commit()
        finally:
            conn.close()
        return {"manifest_id": body.get("manifest_id"), "selected": len(selected), "inserted": inserted, "existing": existing, **self.stats()}

    def list(self, *, status: str | None = None, assignee: str | None = None, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if status:
            clauses.append("status=?")
            params.append(status)
        if assignee:
            clauses.append("assignee=?")
            params.append(assignee)
        sql = "SELECT * FROM human_tasks"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY priority DESC, episode ASC, start_frame ASC LIMIT ? OFFSET ?"
        params.extend([max(1, int(limit)), max(0, int(offset))])
        conn = self._connect()
        try:
            rows = [dict(row) for row in conn.execute(sql, params)]
        finally:
            conn.close()
        for row in rows:
            row["conflict"] = bool(row["conflict"])
            row["payload"] = json.loads(row.pop("payload_json")) if row.get("payload_json") else None
        return rows

    def assign(self, assignee: str, *, limit: int = 20, strategy: str = "priority") -> dict[str, Any]:
        tasks = self.list(status="pending", limit=100000)
        prefilled = self.list(status="prefilled", limit=100000)
        candidates = sample_tasks(tasks + prefilled, strategy=strategy, limit=limit)
        if not candidates:
            return {"assignee": assignee, "assigned": 0, "tasks": []}
        now = time.time()
        conn = self._connect()
        try:
            for task in candidates:
                conn.execute("UPDATE human_tasks SET assignee=?, status='assigned', updated_at=? WHERE task_id=? AND status IN ('pending','prefilled')", (assignee, now, task["task_id"]))
            conn.commit()
        finally:
            conn.close()
        return {"assignee": assignee, "assigned": len(candidates), "tasks": [task["task_id"] for task in candidates]}

    def complete(self, labels: Sequence[Mapping[str, Any]], *, reviewer: str | None = None) -> dict[str, Any]:
        now = time.time()
        updated = 0
        unknown: list[str] = []
        conn = self._connect()
        try:
            for label in labels:
                task_id = str(label.get("task_id") or label.get("id") or "")
                if not task_id:
                    unknown.append(task_id)
                    continue
                answer = normalize_answer(label.get("answer") or label.get("label"))
                cursor = conn.execute(
                    "UPDATE human_tasks SET status='reviewed', review_answer=?, reason=?, assignee=COALESCE(?, assignee), updated_at=? WHERE task_id=?",
                    (answer, str(label.get("reason") or ""), reviewer, now, task_id),
                )
                if cursor.rowcount:
                    updated += 1
                else:
                    unknown.append(task_id)
            conn.commit()
        finally:
            conn.close()
        return {"updated": updated, "unknown": unknown, **self.stats()}

    def export_labels(self, *, status: str = "reviewed", limit: int = 100000) -> list[dict[str, Any]]:
        rows = self.list(status=status, limit=limit)
        return [
            {
                "task_id": row["task_id"],
                "answer": row["review_answer"],
                "reason": row["reason"] or "",
                "reviewer": row["assignee"] or "",
            }
            for row in rows
            if row.get("review_answer")
        ]

    def stats(self) -> dict[str, Any]:
        now = time.time()
        conn = self._connect()
        try:
            by_status = {row["status"]: int(row["n"]) for row in conn.execute("SELECT status, COUNT(*) AS n FROM human_tasks GROUP BY status")}
            by_assignee = {row["assignee"] or "unassigned": int(row["n"]) for row in conn.execute("SELECT assignee, COUNT(*) AS n FROM human_tasks GROUP BY assignee")}
            overdue = conn.execute("SELECT COUNT(*) FROM human_tasks WHERE due_at IS NOT NULL AND due_at < ? AND status != 'reviewed'", (now,)).fetchone()[0]
            total = conn.execute("SELECT COUNT(*) FROM human_tasks").fetchone()[0]
            reviewed = by_status.get("reviewed", 0)
        finally:
            conn.close()
        return {
            "total_tasks": int(total),
            "by_status": by_status,
            "by_assignee": by_assignee,
            "overdue": int(overdue),
            "completion_ratio": round(reviewed / max(int(total), 1), 4),
        }
