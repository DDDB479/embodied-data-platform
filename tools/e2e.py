"""End-to-end exercise of the whole toolkit against generated fixtures."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIXTURES = Path(os.environ.get("EMBODIED_FIXTURES", HERE.parent / ".devtools" / "fixtures"))
PYTHON = sys.executable

results: list[tuple[str, bool, str]] = []


def check(label: str, fn):
    try:
        detail = fn()
        results.append((label, True, "" if detail is None else str(detail)))
    except Exception as exc:
        results.append((label, False, f"{type(exc).__name__}: {exc}"))
        traceback.print_exc(limit=3)
    return results[-1][1]


from embodied_core import open_dataset, describe_formats, detect_dataset, validate  # noqa: E402
from embodied_core.quality import compute_feature_stats, detect_anomalies, segment_by_motion  # noqa: E402

EXPECTED = {
    "canonical": "embodied-jsonl",
    "lerobot_v21": "lerobot",
    "hdf5": "hdf5",
    "rosbag2": "rosbag2",
    "rlds": "rlds",
    "mcap": "mcap",
}


def t_detect():
    seen = {}
    for name, expected in EXPECTED.items():
        path = FIXTURES / name
        detection = detect_dataset(path)
        seen[name] = detection.format
        assert detection.format == expected, f"{name}: detected {detection.format}, expected {expected}"
    return seen


def t_unknown():
    from embodied_core.registry import UnsupportedFormat

    try:
        detect_dataset(FIXTURES / "broken")
    except UnsupportedFormat as exc:
        return str(exc)[:60]
    raise AssertionError("expected UnsupportedFormat")


def t_summaries():
    out = {}
    for name in EXPECTED:
        dataset = open_dataset(FIXTURES / name)
        try:
            summary = dataset.summary
            assert summary.episode_count >= 1, f"{name}: no episodes"
            keys = summary.feature_keys()
            assert keys, f"{name}: no features"
            out[name] = {
                "episodes": summary.episode_count,
                "frames": summary.total_frames,
                "fps": summary.fps,
                "robot": summary.robot_type,
                "features": keys[:4],
                "caps": sorted(dataset.capabilities),
            }
        finally:
            dataset.close()
    return json.dumps(out)


def t_frames():
    dataset = open_dataset(FIXTURES / "canonical")
    ids, columns = dataset.frames()
    assert set(ids) == {0, 1}, ids
    assert "observation.state" in columns, list(columns)
    assert columns["observation.state"].shape == (105, 6), columns["observation.state"].shape
    dataset.close()
    return f"{len(columns)} columns, {sum(len(v) for v in columns.values())} values"


def t_validate_canonical():
    report = validate(FIXTURES / "canonical", include_frames=True)
    codes = [issue["code"] for issue in report["issues"]]
    assert "dataset.empty" not in codes
    assert report["ok"] is True, report["issues"]
    return f"score={report['score']} issues={len(report['issues'])}"


def t_validate_lerobot():
    report = validate(FIXTURES / "lerobot_v21", include_frames=False)
    codes = {issue["code"] for issue in report["issues"]}
    assert "meta.missing_fps" not in codes, codes
    assert "dataset.empty" not in codes, codes
    return f"ok={report['ok']} codes={sorted(codes)}"


def t_quality():
    dataset = open_dataset(FIXTURES / "canonical")
    ids, columns = dataset.frames()
    stats = compute_feature_stats(columns)
    windows = detect_anomalies(ids, columns, fps=dataset.summary.fps, z=6.0)
    dataset.close()
    assert stats["observation.state"]["frames"] == 105, stats["observation.state"]
    return f"nan_ratio={stats['observation.state']['nan_ratio']:.3f} windows={len(windows)}"


def t_quality_cache():
    from embodied_core.reports import quality_report

    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp) / "quality-cache.sqlite3"
        first = quality_report(FIXTURES / "canonical", cache_path=cache)
        second = quality_report(FIXTURES / "canonical", cache_path=cache)
        assert first["cache"]["hit"] is False, first["cache"]
        assert second["cache"]["hit"] is True, second["cache"]
        assert second["cache"]["hits"] == 1, second["cache"]
    return "miss -> hit"


def t_segment():
    dataset = open_dataset(FIXTURES / "canonical")
    ids, columns = dataset.frames([0])
    segments = segment_by_motion(ids, columns, fps=dataset.summary.fps)
    dataset.close()
    assert segments, "no segments"
    labels = [segment["label"] for segment in segments.get(0, [])]
    return f"{sum(len(v) for v in segments.values())} segments in {len(segments)} episodes: {labels[:6]}"


def t_label():
    from embodied_core.quality import build_labeling_tasks

    dataset = open_dataset(FIXTURES / "canonical")
    ids, columns = dataset.frames()
    windows = detect_anomalies(ids, columns, fps=dataset.summary.fps, z=6.0)
    manifest = build_labeling_tasks(dataset.summary, windows, max_tasks=20, include_segments=True)
    dataset.close()
    assert manifest["task_count"] >= 1, manifest
    return f"tasks={manifest['task_count']}"


def t_async_jobs():
    import time

    from embodied_core import mcp_server

    with tempfile.TemporaryDirectory() as tmp:
        previous = os.environ.get("EMBODIED_QUALITY_CACHE")
        os.environ["EMBODIED_QUALITY_CACHE"] = str(Path(tmp) / "quality-cache.sqlite3")
        events = []
        try:
            response = mcp_server.handle_message(
                {
                    "jsonrpc": "2.0",
                    "id": 99,
                    "method": "tools/call",
                    "params": {
                        "name": "submit_job",
                        "arguments": {"tool": "quality_report", "arguments": {"path": str(FIXTURES / "canonical")}},
                        "_meta": {"progressToken": "e2e-job"},
                    },
                },
                emit=events.append,
            )
            submitted = json.loads(response["result"]["content"][0]["text"])
            job_id = submitted["job"]["job_id"]
            deadline = time.time() + 30
            status = {}
            while time.time() < deadline:
                status_response = mcp_server.handle_message(
                    {
                        "jsonrpc": "2.0",
                        "id": 100,
                        "method": "tools/call",
                        "params": {"name": "job_status", "arguments": {"job_id": job_id, "include_result": True}},
                    }
                )
                status = json.loads(status_response["result"]["content"][0]["text"])
                if status["status"] in {"succeeded", "failed", "cancelled"}:
                    break
                time.sleep(0.05)
            assert status["status"] == "succeeded", status
            assert "observation.state" in status["result"]["stats"], status["result"]
            progress = [event for event in events if event.get("method") == "notifications/progress"]
            assert progress, events
            assert all(item["params"]["progressToken"] == "e2e-job" for item in progress)
        finally:
            if previous is None:
                os.environ.pop("EMBODIED_QUALITY_CACHE", None)
            else:
                os.environ["EMBODIED_QUALITY_CACHE"] = previous
    return f"job={job_id} progress_events={len(progress)}"


def t_streaming_and_gate():
    from embodied_core import evaluate_quality_gate
    from embodied_core.reports import quality_report

    dataset = open_dataset(FIXTURES / "canonical")
    try:
        chunks = list(dataset.iter_frames(chunk_frames=13, max_frames=0))
        assert len(chunks) > 1, len(chunks)
        assert sum(len(chunk.ids) for chunk in chunks) == 105
    finally:
        dataset.close()
    quality = quality_report(FIXTURES / "canonical", use_cache=False, chunk_frames=13, max_frames=0)
    assert quality["scan"]["chunks"] > 1, quality["scan"]
    assert quality["stats"]["observation.state"]["frames"] == 105, quality["stats"]
    gate = evaluate_quality_gate(
        FIXTURES / "canonical",
        rules={"expected_fps": 30, "required_features": ["observation.state", "action"], "max_nan_ratio": 0.01},
    )
    assert gate["ok"], gate["gate"]
    rejected = evaluate_quality_gate(FIXTURES / "canonical", rules={"expected_fps": 60})
    assert not rejected["ok"], rejected["gate"]
    return f"chunks={len(chunks)} gate=pass reject={rejected['gate']['failures'][0]['code']}"


def t_convert_jsonl_roundtrip():
    from embodied_core.convert import convert

    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "out"
        result = convert(FIXTURES / "lerobot_v21", target, target="jsonl")
        assert result["episodes"] == 2, result
        again = open_dataset(target)
        try:
            assert again.summary.episode_count == 2
        finally:
            again.close()
    return "lerobot -> jsonl ok"


def t_convert_lerobot():
    from embodied_core.convert import convert

    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "out"
        result = convert(FIXTURES / "canonical", target, target="lerobot")
        assert (target / "meta" / "info.json").is_file(), result
        parquet = list(target.rglob("*.parquet"))
        assert parquet, result
        back = open_dataset(target)
        try:
            assert back.summary.episode_count == 2, back.summary.to_dict(include_episodes=False)
        finally:
            back.close()
    return f"jsonl -> lerobot ok ({result.get('frames')} frames)"


def t_catalog():
    from embodied_core.catalog import export_manifest, history, index, search, stats

    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "catalog.db"
        report = index(FIXTURES, db, max_depth=2, validate=True)
        assert report["total"] >= 6, report
        rows = search(db, text="pick")
        assert rows, "search returned nothing"
        info = stats(db)
        assert info["total_datasets"] >= 6, info
        assert info["schema_version"] == 2, info
        assert info["total_versions"] >= info["total_datasets"], info
        assert info["total_runs"] >= 1, info
        manifest = export_manifest(db)
        assert manifest["datasets"], manifest
        combined = history(db, limit=5)
        assert combined["versions"] and combined["runs"], combined
    return f"schema=2 indexed={report['total']} versions={info['total_versions']} runs={info['total_runs']}"


def t_scaffold():
    from embodied_core.scaffold import scaffold

    with tempfile.TemporaryDirectory() as tmp:
        for kind in ("platform", "adapter", "pipeline"):
            dest = Path(tmp) / kind
            result = scaffold(dest, kind=kind, name=f"demo-{kind}")
            assert result["files"], (kind, result)
            assert (dest / "README.md").is_file(), (kind, result["files"])
    return "platform/adapter/pipeline ok"


def t_cli():
    commands = [
        ["formats"],
        ["detect", str(FIXTURES / "lerobot_v21"), "--json"],
        ["validate", str(FIXTURES / "canonical"), "--json"],
        ["gate", str(FIXTURES / "canonical"), "--expected-fps", "30", "--json"],
        ["info", str(FIXTURES / "hdf5"), "--json"],
    ]
    codes = []
    for argv in commands:
        proc = subprocess.run(
            [PYTHON, "-m", "embodied_core.cli", *argv],
            capture_output=True,
            text=True,
            cwd=str(HERE.parent),
        )
        codes.append((argv[0], proc.returncode))
        assert proc.returncode == 0, f"{argv}: {proc.returncode}\n{proc.stderr[-800:]}"
        if argv[0] != "formats":
            json.loads(proc.stdout)
    return codes


def t_mcp_stdio():
    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05"}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "detect_dataset", "arguments": {"path": str(FIXTURES / "rosbag2")}},
        },
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {"name": "summarize_dataset", "arguments": {"path": str(FIXTURES / "canonical")}},
        },
    ]
    payload = "\n".join(json.dumps(item) for item in requests) + "\n"
    proc = subprocess.run(
        [PYTHON, "-m", "embodied_core.mcp_server"],
        input=payload,
        capture_output=True,
        text=True,
        cwd=str(HERE.parent),
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr[-800:]
    responses = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
    assert len(responses) == 4, [r.get("id") for r in responses]
    tools = responses[1]["result"]["tools"]
    assert len(tools) == 19, len(tools)
    detect_payload = json.loads(responses[2]["result"]["content"][0]["text"])
    assert detect_payload["format"] == "rosbag2", detect_payload
    summary_payload = json.loads(responses[3]["result"]["content"][0]["text"])
    assert summary_payload["episodes"] == 2, summary_payload
    return f"tools={len(tools)} detect={detect_payload['format']}"


def t_readme_examples():
    formats = describe_formats()
    assert any(item["format"] == "lerobot" for item in formats)
    for item in formats:
        assert item["extensions"], item
    return f"{len(formats)} formats documented"


for label, fn in [
    ("detect fixtures", t_detect),
    ("unknown format", t_unknown),
    ("summaries", t_summaries),
    ("frames", t_frames),
    ("validate canonical", t_validate_canonical),
    ("validate lerobot", t_validate_lerobot),
    ("quality", t_quality),
    ("quality cache", t_quality_cache),
    ("segment", t_segment),
    ("labeling tasks", t_label),
    ("async jobs", t_async_jobs),
    ("streaming + gate", t_streaming_and_gate),
    ("convert lerobot->jsonl", t_convert_jsonl_roundtrip),
    ("convert jsonl->lerobot", t_convert_lerobot),
    ("catalog", t_catalog),
    ("scaffold", t_scaffold),
    ("cli", t_cli),
    ("mcp stdio", t_mcp_stdio),
    ("format docs", t_readme_examples),
]:
    check(label, fn)

print("\n=== results ===")
failed = 0
for label, ok, detail in results:
    status = "PASS" if ok else "FAIL"
    print(f"{status}  {label:26s} {detail}")
    failed += 0 if ok else 1
print(f"\n{len(results) - failed}/{len(results)} passed")
raise SystemExit(1 if failed else 0)
